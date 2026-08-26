import argparse
import os, sys
import json
from huggingface_hub import HfApi, CommitOperationAdd
from collections import defaultdict

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

try:
    import morfessor
except ImportError:
    morfessor = None

try:
    from myte_tokenizer import MyteTokenizer
except ImportError:
    MyteTokenizer = None

# Repo root: these scripts live in scripts/tokenization/, while the external
# tokenizer clones (superbpe/, boundlessbpe/, picky_bpe/, SaGe/) and the
# sibling dataset/ directory are resolved relative to the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_segments_sp(sp, sentence, enable_bpe_dropout=False, dropout_prob=0.1):
    if enable_bpe_dropout:
        pieces = sp.encode_as_pieces(sentence, enable_sampling=True, alpha=dropout_prob, nbest_size=-1)
    else:
        pieces = sp.encode_as_pieces(sentence)
    words = []
    current_word = []
    
    for piece in pieces:
        # Check for U+2581 (SentencePiece space)
        is_new = piece.startswith('\u2581')
        clean = piece.replace('\u2581', '')
        
        if is_new:
            if current_word:
                words.append(current_word)
            current_word = []
            if clean:
                current_word.append(clean)
        else:
            if clean:
                current_word.append(clean)
                
    if current_word:
        words.append(current_word)
        
    return words

def get_segments_wp(tokenizer, sentence):
    encoded = tokenizer.encode(sentence)
    words = []
    current_word = []
    last_word_id = None
    
    for token, word_id in zip(encoded.tokens, encoded.word_ids):
        if word_id is None:
            continue
            
        clean = token.replace('##', '')
        if not clean:
            continue
            
        if word_id != last_word_id:
            if current_word:
                words.append(current_word)
            current_word = [clean]
            last_word_id = word_id
        else:
            current_word.append(clean)
            
    if current_word:
        words.append(current_word)
        
    return words

def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

def get_segments_superbpe(tokenizer, sentence):
    encoded = tokenizer.encode(sentence)
    
    words = []
    current_word_subwords = []
    byte_buffer = bytearray()
    
    # Check if the tokenizer uses byte-level BPE
    is_bytelevel = False
    if hasattr(tokenizer, 'get_vocab'):
        is_bytelevel = 'Ġ' in tokenizer.get_vocab()
        
    for token_str in encoded.tokens:
        is_new = token_str.startswith('Ġ') or token_str.startswith(' ')

        # Check if the model is a new version (real spaces) or old version (Ġ)
        if token_str.startswith('Ġ'):
            clean_token = token_str.replace('Ġ', '', 1)
        elif token_str.startswith(' '):
            clean_token = token_str.lstrip(' ')
        else:
            clean_token = token_str
        
        if is_new and (byte_buffer or current_word_subwords):
            if byte_buffer:
                current_word_subwords.append(byte_buffer.decode('utf-8', errors='replace'))
                byte_buffer.clear()
            if current_word_subwords:
                words.append(current_word_subwords)
            current_word_subwords = []
            
        if clean_token:
            if is_bytelevel:
                # If the token contains byte-mapped characters (backward compatibility)
                try:
                    token_bytes = bytearray([byte_decoder.get(c, ord(c)) for c in clean_token])
                except ValueError:
                    token_bytes = bytearray(clean_token.encode('utf-8'))
                
                byte_buffer.extend(token_bytes)
                
                try:
                    decoded_str = byte_buffer.decode('utf-8')
                    if decoded_str:
                        current_word_subwords.append(decoded_str)
                    byte_buffer.clear()
                except UnicodeDecodeError:
                    pass
            else:
                current_word_subwords.append(clean_token)
            
    if byte_buffer:
        current_word_subwords.append(byte_buffer.decode('utf-8', errors='replace'))
    if current_word_subwords:
        words.append(current_word_subwords)
        
    return words

def get_segments_morfessor(model, sentence):
    words = []
    for word in sentence.split():
        segments = model.viterbi_segment(word)[0]
        words.append(segments)
    return words

def get_segments_boundlessbpe(tokenizer, sentence):
    # encode_ordinary_chunks returns list of bytes
    tokens_bytes = tokenizer.encode_ordinary_chunks(sentence, blowup=True)
    
    words = []
    current_word = []
    byte_buffer = bytearray()
    
    for piece_bytes in tokens_bytes:
        is_new = piece_bytes.startswith(b' ') or piece_bytes.startswith(b'\n') or piece_bytes.startswith(b'\r')
        
        # We strip the leading whitespace to mimic BPE behavior
        if is_new:
            if byte_buffer:
                current_word.append(byte_buffer.decode('utf-8', errors='replace'))
                byte_buffer.clear()
            if current_word:
                words.append(current_word)
            current_word = []
            
            clean_bytes = piece_bytes.lstrip(b' \n\r')
        else:
            clean_bytes = piece_bytes
            
        byte_buffer.extend(clean_bytes)
        
        try:
            decoded_str = byte_buffer.decode('utf-8')
            if decoded_str:
                current_word.append(decoded_str)
            byte_buffer.clear()
        except UnicodeDecodeError:
            pass

    if byte_buffer:
        current_word.append(byte_buffer.decode('utf-8', errors='replace'))
    if current_word:
        words.append(current_word)
        
    return words

def get_segments_pickybpe(model, sentence):
    from utils import WHITESPACE
    words = []
    for word in sentence.split():
        tokens = model._encode_word_by_events(WHITESPACE + word)
        current_word = []
        for token in tokens:
            clean = token.str.replace(WHITESPACE, '')
            if clean:
                current_word.append(clean)
        if current_word:
            words.append(current_word)
    return words

def get_segments_pathpiece(tokenizer, sentence):
    encoded = tokenizer(sentence)  # returns dict with 'input_ids'
    ids = tokenizer.get_ids()  # unjoined strings
    # We do not use decode because we want subword strings.
    # ids is a dict of index -> string token
    # It contains regular spaces ' ' from the hex-encoded vocabulary mapping.
    
    words = []
    current_word_subwords = []
    byte_buffer = bytearray()
    
    for token_id in encoded['input_ids']:
        # pathpiece ids are bytes
        piece_bytes = ids[token_id]
        
        # Check for regular space
        is_new = piece_bytes.startswith(b' ')
        clean_bytes = piece_bytes.replace(b' ', b'')
        
        if is_new and (byte_buffer or current_word_subwords):
            if byte_buffer:
                current_word_subwords.append(byte_buffer.decode('utf-8', errors='replace'))
                byte_buffer.clear()
            if current_word_subwords:
                words.append(current_word_subwords)
            current_word_subwords = []
            
        byte_buffer.extend(clean_bytes)
        
        try:
            # If the current buffer decodes successfully, it's a complete UTF-8 sequence
            decoded_str = byte_buffer.decode('utf-8')
            if decoded_str:
                current_word_subwords.append(decoded_str)
            byte_buffer.clear()
        except UnicodeDecodeError:
            # Buffer waits for the next token bytes to complete the character
            pass
            
    if byte_buffer:
        current_word_subwords.append(byte_buffer.decode('utf-8', errors='replace'))
    if current_word_subwords:
        words.append(current_word_subwords)
        
    return words

def get_segments_myte(tokenizer, sentence):
    return tokenizer.get_segments(sentence)

    
def get_segments_sage(tokenizer, sentence):
    tokens_bytes = tokenizer.tokenize_to_bytes(sentence)
    
    words = []
    current_word = []
    byte_buffer = bytearray()
    
    for piece_bytes in tokens_bytes:
        is_new = piece_bytes.startswith(b' ')
        clean_bytes = piece_bytes.replace(b' ', b'')
        
        if is_new and (byte_buffer or current_word):
            if byte_buffer:
                current_word.append(byte_buffer.decode('utf-8', errors='replace'))
                byte_buffer.clear()
            if current_word:
                words.append(current_word)
            current_word = []
            
        byte_buffer.extend(clean_bytes)
        
        try:
            # If the current buffer decodes successfully, it's a complete UTF-8 sequence
            decoded_str = byte_buffer.decode('utf-8')
            if decoded_str:
                current_word.append(decoded_str)
            byte_buffer.clear()
        except UnicodeDecodeError:
            pass
            
    if byte_buffer:
        current_word.append(byte_buffer.decode('utf-8', errors='replace'))
    if current_word:
        words.append(current_word)
        
    return words

def main():
    parser = argparse.ArgumentParser(description="Segment test data and upload to Hugging Face")
    parser.add_argument("--langs", nargs="+", required=True, help="List of language codes")
    parser.add_argument("--model_types", nargs="+", default=["bpe", "unigram", "wordpiece"], help="List of model types")
    parser.add_argument("--vocab_size", type=int, default=25000, help="Vocabulary size (T: final vocab size)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--hf_repo", type=str, default="SakethVemula/fixed-tokenizer-segments", help="Hugging Face repo ID")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of files to commit in one API call")
    parser.add_argument("--bpe_dropout", action="store_true", help="Enable BPE dropout for BPE tokenizer")
    parser.add_argument("--bpe_dropout_prob", type=float, default=0.1, help="Dropout probability for BPE dropout (alpha)")
    parser.add_argument("--superbpe_base_vocab_size", type=int, default=10000, help="Base vocab size for SuperBPE (t: vocab size of subword stage)")
    
    args = parser.parse_args()

    api = HfApi()
    created_repos = set()
    operations_by_repo = defaultdict(list)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(REPO_ROOT), 'dataset')
    tokenizers_dir = os.path.join(REPO_ROOT, 'tokenizers-bin')
    output_base_dir = os.path.join(REPO_ROOT, 'segmented_outputs')
    os.makedirs(output_base_dir, exist_ok=True)

    def flush_repo(repo_id):
        if operations_by_repo[repo_id]:
            print(f"Uploading {len(operations_by_repo[repo_id])} files to {repo_id}...")
            if repo_id not in created_repos:
                try:
                    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
                    created_repos.add(repo_id)
                except Exception as e:
                    print(f"Note: Error checking/creating repo {repo_id}: {e}")
            
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations_by_repo[repo_id],
                    commit_message=f"Upload segmented {args.split} subset"
                )
                print(f"Successfully uploaded to {repo_id}")
            except Exception as e:
                print(f"Failed to commit to {repo_id}: {e}")
            
            operations_by_repo[repo_id] = []

    repo_id = args.hf_repo
    for model_type in args.model_types:
        for lang in args.langs:
            # allow bpe dropout for morphbpe as well
            output_model_type = "bpe-dropout" if (args.bpe_dropout and model_type == 'bpe') else ("morphbpe-dropout" if args.bpe_dropout and model_type == 'morphbpe' else model_type)
            
            print(f"\nProcessing {lang} - {output_model_type} (from {model_type})")
            
            input_file = os.path.join(dataset_dir, lang, f"{args.split}.{lang}")
            if not os.path.exists(input_file):
                print(f"Warning: Dataset file {input_file} not found. Skipping.")
                continue

            if model_type in ['wordpiece', 'morphwp']:
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.json")
            elif model_type == 'morfessor':
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.bin")
            elif model_type == 'superbpe':
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.superbpe_base_vocab_size}_{args.vocab_size}.json")
            elif model_type == 'boundlessbpe':
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.model")
            elif model_type == 'pickybpe':
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.json")
            elif model_type == 'pathpiece':
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.vocab")
            elif model_type == 'myte':
                model_json_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.json")
                model_bin_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_morfessor_{args.vocab_size}.bin")
                model_path = model_json_path # Primary path for check
            elif model_type == 'sage':
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.vocab")
            else:
                model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.vocab_size}.model")

            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found. Skipping.")
                continue
                
            out_dir_lang = os.path.join(output_base_dir, lang)
            os.makedirs(out_dir_lang, exist_ok=True)
            if model_type == 'superbpe':
                out_file = os.path.join(out_dir_lang, f"{output_model_type}_v{args.superbpe_base_vocab_size}t-{args.vocab_size}T_{args.split}.jsonl")
            else:
                out_file = os.path.join(out_dir_lang, f"{output_model_type}_v{args.vocab_size}_{args.split}.jsonl")

            # Load model
            if model_type in ['wordpiece', 'superbpe', 'morphwp']:
                if Tokenizer is None:
                    print("Error: tokenizers library not installed.")
                    continue
                tokenizer = Tokenizer.from_file(model_path)
            elif model_type == 'morfessor':
                if morfessor is None:
                    print("Error: morfessor library not installed.")
                    continue
                io = morfessor.MorfessorIO()
                model = io.read_binary_model_file(model_path)
            elif model_type == 'boundlessbpe':
                boundlessbpe_dir = os.path.join(REPO_ROOT, "boundlessbpe")
                if boundlessbpe_dir not in sys.path:
                    sys.path.insert(0, boundlessbpe_dir)
                from boundlessbpe import FasterRegexInference
                tokenizer = FasterRegexInference()
                tokenizer.load(model_path)
            elif model_type == 'pickybpe':
                pickybpe_dir = os.path.join(REPO_ROOT, "picky_bpe")
                if pickybpe_dir not in sys.path:
                    sys.path.insert(0, pickybpe_dir)
                from picky_tokenize import BPEModel
                tokenizer = BPEModel(model_path)
            elif model_type == 'pathpiece':
                try:
                    import pathpiece
                except ImportError:
                    print("Error: pathpiece library not installed/in environment.")
                    continue
                tokenizer = pathpiece.Tokenizer(model_path)

            elif model_type == 'myte':
                if MyteTokenizer is None:
                    print("Error: myte_tokenizer.py not found.")
                    continue
                tokenizer = MyteTokenizer()
                tokenizer.load(model_json_path, model_bin_path)
            elif model_type == 'sage':
                sage_dir = os.path.join(REPO_ROOT, "SaGe", "src")
                if sage_dir not in sys.path:
                    sys.path.insert(0, sage_dir)
                try:
                    from sage_tokenizer.model import SaGeTokenizer
                except ImportError:
                    print("Error: SaGeTokenizer not found.")
                    continue
                
                vocab_bytes = []
                with open(model_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip('\n')
                        if not line: continue
                        try:
                            token_bytes = bytes.fromhex(line)
                        except ValueError:
                            # Fallback just in case
                            try:
                                token_str = json.loads(line)
                            except json.JSONDecodeError:
                                token_str = line
                            if isinstance(token_str, int):
                                token_str = str(token_str)
                            token_bytes = token_str.encode('utf-8')
                        vocab_bytes.append(token_bytes)
                
                # SaGe requires all 256 single bytes to exist in the vocabulary
                existing_bytes = set(vocab_bytes)
                for i in range(256):
                    b = bytes([i])
                    if b not in existing_bytes:
                        vocab_bytes.append(b)
                        
                tokenizer = SaGeTokenizer(vocab_bytes)
            else:
                if spm is None:
                    print("Error: sentencepiece library not installed.")
                    continue
                sp = spm.SentencePieceProcessor()
                sp.load(model_path)
            
            print(f"Segmenting {input_file}...")
            # Segment
            try:
                with open(input_file, 'r', encoding='utf-8') as f_in, \
                     open(out_file, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        if model_type in ['wordpiece', 'morphwp']:
                            words = get_segments_wp(tokenizer, line)
                        elif model_type == 'superbpe':
                            words = get_segments_superbpe(tokenizer, line)
                        elif model_type == 'boundlessbpe':
                            words = get_segments_boundlessbpe(tokenizer, line)
                        elif model_type == 'pickybpe':
                            words = get_segments_pickybpe(tokenizer, line)
                        elif model_type == 'pathpiece':
                            words = get_segments_pathpiece(tokenizer, line)

                        elif model_type == 'myte':
                            words = get_segments_myte(tokenizer, line)
                        elif model_type == 'sage':
                            words = get_segments_sage(tokenizer, line)
                        elif model_type == 'morfessor':
                            words = get_segments_morfessor(model, line)
                        else:
                            words = get_segments_sp(
                                sp, line, 
                                enable_bpe_dropout=(args.bpe_dropout and model_type in ['bpe', 'morphbpe']), 
                                dropout_prob=args.bpe_dropout_prob
                            )
                        f_out.write(json.dumps(words, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error during segmentation: {e}")
                continue
            
            # Add to HF commit operations
            if model_type == 'superbpe':
                path_in_repo = f"{lang}/{output_model_type}_v{args.superbpe_base_vocab_size}t-{args.vocab_size}T_{args.split}.jsonl"
            else:
                path_in_repo = f"{lang}/{output_model_type}_v{args.vocab_size}_{args.split}.jsonl"
            op = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=out_file)
            operations_by_repo[repo_id].append(op)
            
            if len(operations_by_repo[repo_id]) >= args.batch_size:
                flush_repo(repo_id)
                
    # Final flush
    flush_repo(repo_id)

if __name__ == "__main__":
    main()
