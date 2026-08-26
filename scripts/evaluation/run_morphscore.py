import argparse
import os, sys
import json
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, CommitOperationAdd
from collections import defaultdict

# Add morphscore directory to path
# Repo root: scripts live two levels below it, while data directories
# (tokenizers-bin/, evaluation_results/, ...) and external tokenizer
# clones live at the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

script_dir = os.path.dirname(os.path.abspath(__file__))
morphscore_dir = os.path.join(REPO_ROOT, 'morphscore')
if morphscore_dir not in sys.path:
    sys.path.insert(0, morphscore_dir)

from morphscore import MorphScore

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


def get_segments_sp(sp, sentence, enable_bpe_dropout=False, dropout_prob=0.1):
    if enable_bpe_dropout:
        pieces = sp.encode_as_pieces(sentence, enable_sampling=True, alpha=dropout_prob, nbest_size=-1)
    else:
        pieces = sp.encode_as_pieces(sentence)
    words = []
    current_word = []
    
    for piece in pieces:
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
    
    is_bytelevel = False
    if hasattr(tokenizer, 'get_vocab'):
        is_bytelevel = 'Ġ' in tokenizer.get_vocab()
        
    for token_str in encoded.tokens:
        is_new = token_str.startswith('Ġ') or token_str.startswith(' ')

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
    tokens_bytes = tokenizer.encode_ordinary_chunks(sentence, blowup=True)
    words = []
    current_word = []
    byte_buffer = bytearray()
    
    for piece_bytes in tokens_bytes:
        is_new = piece_bytes.startswith(b' ') or piece_bytes.startswith(b'\n') or piece_bytes.startswith(b'\r')
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
    encoded = tokenizer(sentence)
    ids = tokenizer.get_ids()
    words = []
    current_word_subwords = []
    byte_buffer = bytearray()
    
    for token_id in encoded['input_ids']:
        piece_bytes = ids[token_id]
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
            decoded_str = byte_buffer.decode('utf-8')
            if decoded_str:
                current_word_subwords.append(decoded_str)
            byte_buffer.clear()
        except UnicodeDecodeError:
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


LANG_MAP = {
    'fin': 'finnish',
    'hun': 'hungarian',
    'mal': 'malayalam',
    'tam': 'tamil',
    'tel': 'telugu',
    'kir': 'kirghiz',
    'tur': 'turkish',
    'ind': 'indonesian',
    'san': 'sanskrit',
    'hin': 'hindi',
    'snd': 'sindhi',
    'hrv': 'croatian',
    'rus': 'russian',
    'fas': 'persian',
    'eng': 'english',
    'swe': 'swedish',
    'heb': 'hebrew',
    'afr': 'afrikaans',
    'kat': 'georgian',
    'ell': 'greek',
    'isl': 'icelandic',
    'gle': 'irish',
    'ita': 'italian',
    'kor': 'korean',
    'lav': 'latvian',
    'spa': 'spanish',
    'mon': 'mongolian' # fallback mapping just in case
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--model_types", nargs="+", required=True)
    parser.add_argument("--vocab_sizes", nargs="+", type=int, required=True)
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--bpe_dropout", action="store_true")
    parser.add_argument("--bpe_dropout_prob", type=float, default=0.1)
    parser.add_argument("--superbpe_base_vocab_size", type=int, default=10000)
    
    args = parser.parse_args()

    api = HfApi(token=args.hf_token)
    created_repos = set()
    operations_by_repo = defaultdict(list)

    dataset_dir = os.path.join(REPO_ROOT, 'morphscore', 'data')
    tokenizers_dir = os.path.join(REPO_ROOT, 'tokenizers-bin')
    output_base_dir = os.path.join(REPO_ROOT, 'morphscore_outputs')
    os.makedirs(output_base_dir, exist_ok=True)

    eval_out_dir = os.path.join(REPO_ROOT, 'evaluation_results', 'morphscore')
    os.makedirs(eval_out_dir, exist_ok=True)

    
    # Initialize morph scorer to use its morph_eval
    morph_scorer = MorphScore(exclude_single_tok=True, exclude_single_morpheme=True)

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
                    commit_message="Upload segmented morphscore evaluations"
                )
                print(f"Successfully uploaded to {repo_id}")
            except Exception as e:
                print(f"Failed to commit to {repo_id}: {e}")
            
            operations_by_repo[repo_id] = []

    repo_id = args.hf_repo

    for lang in args.langs:
        lang_full = LANG_MAP.get(lang, lang)
        data_path = os.path.join(dataset_dir, f"{lang_full}_data.csv")
        
        if not os.path.exists(data_path):
            print(f"Warning: Morphscore dataset for {lang} ({data_path}) not found. Skipping.")
            continue
            
        # 1. Load the dataset
        df = pd.read_csv(data_path)
        
        # 2. Filter out wordforms for which combining the segments doesn't give rise to the wordform
        filtered_rows = []
        for _, row in df.iterrows():
            wordform = row.get('wordform', '')
            if pd.isna(wordform): continue
            wordform = str(wordform).strip()
            if not wordform: continue
            
            prefix = row.get('preceding_part', '')
            prefix = '' if pd.isna(prefix) else str(prefix)
            
            stem = row.get('stem', '')
            stem = '' if pd.isna(stem) else str(stem)
            
            suffix = row.get('following_part', '')
            suffix = '' if pd.isna(suffix) else str(suffix)
            
            gold_str = prefix + stem + suffix
            if gold_str != wordform:
                continue
                
            morphemes = []
            if prefix: morphemes.append(prefix)
            if stem: morphemes.append(stem)
            if suffix: morphemes.append(suffix)
            
            # ensure morphemes is non-empty
            if not morphemes:
                continue
                
            # Exclude single morpheme instances if flag is true
            if morph_scorer.config.get('exclude_single_morpheme', True) and len(morphemes) <= 1:
                continue
                
            filtered_rows.append({
                'wordform': wordform,
                'morphemes': morphemes
            })
            
        print(f"Language {lang}: {len(filtered_rows)} words retained after filtering.")
        if not filtered_rows:
            continue

        for model_type in args.model_types:
            for vocab_size in args.vocab_sizes:
                output_model_type = "bpe-dropout" if (args.bpe_dropout and model_type == 'bpe') else ("morphbpe-dropout" if args.bpe_dropout and model_type == 'morphbpe' else model_type)
                
                print(f"\nProcessing {lang} - {output_model_type} (vocab: {vocab_size})")
                
                if model_type in ['wordpiece', 'morphwp']:
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.json")
                elif model_type == 'morfessor':
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.bin")
                elif model_type == 'superbpe':
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{args.superbpe_base_vocab_size}_{vocab_size}.json")
                elif model_type == 'boundlessbpe':
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.model")
                elif model_type == 'pickybpe':
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.json")
                elif model_type == 'pathpiece':
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.vocab")
                elif model_type == 'myte':
                    model_json_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.json")
                    model_bin_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_morfessor_{vocab_size}.bin")
                    model_path = model_json_path 
                elif model_type == 'sage':
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.vocab")
                else:
                    model_path = os.path.join(tokenizers_dir, f"{lang}_{model_type}_{vocab_size}.model")

                if not os.path.exists(model_path):
                    print(f"Warning: Model file {model_path} not found. Skipping.")
                    continue
                    
                out_dir_lang = os.path.join(output_base_dir, lang)
                os.makedirs(out_dir_lang, exist_ok=True)
                if model_type == 'superbpe':
                    out_file = os.path.join(out_dir_lang, f"{output_model_type}_v{args.superbpe_base_vocab_size}t-{vocab_size}T.csv")
                    path_in_repo = f"{lang}/{output_model_type}_v{args.superbpe_base_vocab_size}t-{vocab_size}T.csv"
                else:
                    out_file = os.path.join(out_dir_lang, f"{output_model_type}_v{vocab_size}.csv")
                    path_in_repo = f"{lang}/{output_model_type}_v{vocab_size}.csv"

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
                        print("Error: pathpiece disabled.")
                        continue
                    tokenizer = pathpiece.Tokenizer(model_path)
                elif model_type == 'myte':
                    if MyteTokenizer is None:
                        print("Error: myte disabled.")
                        continue
                    tokenizer = MyteTokenizer()
                    tokenizer.load(model_json_path, model_bin_path)
                elif model_type == 'sage':
                    sage_dir = os.path.join(REPO_ROOT, "SaGe", "src")
                    if sage_dir not in sys.path:
                        sys.path.insert(0, sage_dir)
                    from sage_tokenizer.model import SaGeTokenizer
                    vocab_bytes = []
                    with open(model_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip('\n')
                            if not line: continue
                            try:
                                token_bytes = bytes.fromhex(line)
                            except ValueError:
                                try:
                                    token_str = json.loads(line)
                                except json.JSONDecodeError:
                                    token_str = line
                                if isinstance(token_str, int):
                                    token_str = str(token_str)
                                token_bytes = token_str.encode('utf-8')
                            vocab_bytes.append(token_bytes)
                    existing_bytes = set(vocab_bytes)
                    for i in range(256):
                        b = bytes([i])
                        if b not in existing_bytes:
                            vocab_bytes.append(b)
                    tokenizer = SaGeTokenizer(vocab_bytes)
                else:
                    if spm is None:
                        print("Error: sentencepiece disabled.")
                        continue
                    sp = spm.SentencePieceProcessor()
                    sp.load(model_path)
                
                # 3. Perform segmentation of filtered wordforms
                # 4. Calculate precision, recall, f1 scores
                results = []
                for item in filtered_rows:
                    wordform = item['wordform']
                    morphemes = item['morphemes']
                    
                    try:
                        if model_type in ['wordpiece', 'morphwp']:
                            word_segments = get_segments_wp(tokenizer, wordform)
                        elif model_type == 'superbpe':
                            word_segments = get_segments_superbpe(tokenizer, wordform)
                        elif model_type == 'boundlessbpe':
                            word_segments = get_segments_boundlessbpe(tokenizer, wordform)
                        elif model_type == 'pickybpe':
                            word_segments = get_segments_pickybpe(tokenizer, wordform)
                        elif model_type == 'pathpiece':
                            word_segments = get_segments_pathpiece(tokenizer, wordform)
                        elif model_type == 'myte':
                            word_segments = get_segments_myte(tokenizer, wordform)
                        elif model_type == 'sage':
                            word_segments = get_segments_sage(tokenizer, wordform)
                        elif model_type == 'morfessor':
                            word_segments = get_segments_morfessor(model, wordform)
                        else:
                            word_segments = get_segments_sp(
                                sp, wordform, 
                                enable_bpe_dropout=(args.bpe_dropout and model_type in ['bpe', 'morphbpe']), 
                                dropout_prob=args.bpe_dropout_prob
                            )
                    except Exception as e:
                        print(f"Error segmenting {wordform}: {e}")
                        continue
                        
                    # get_segments returns a list of words where each word is a list of subwords.
                    # Since we feed a single wordform (which usually does not contain spaces), 
                    # word_segments should have length 1 if it doesn't split on spaces.
                    if len(word_segments) == 0:
                        pred_segments = []
                    elif len(word_segments) == 1:
                        pred_segments = word_segments[0]
                    else:
                        # flatten if it somehow split into multiple words
                        pred_segments = [seg for word in word_segments for seg in word]
                        
                    # Exclude single token predictions if flag is true
                    if morph_scorer.config.get('exclude_single_tok', True) and len(pred_segments) <= 1:
                        continue
                    
                    point_recall, point_precision = morph_scorer.morph_eval(morphemes, pred_segments)
                    
                    if pd.isna(point_recall) or pd.isna(point_precision):
                        point_recall = 0.0
                        point_precision = 0.0
                        
                    if point_precision + point_recall > 0:
                        f1 = 2 * point_precision * point_recall / (point_precision + point_recall)
                    else:
                        f1 = 0.0
                        
                    results.append({
                        'wordform': wordform,
                        'gold_segment': ' '.join(morphemes),
                        'predicted_segment': ' '.join(pred_segments),
                        'precision': float(point_precision),
                        'recall': float(point_recall),
                        'f1': float(f1)
                    })
                    
                # 5. Save the necessary segmented and evaluated data as a CSV
                out_df = pd.DataFrame(results)
                out_df.to_csv(out_file, index=False)
                
                # Calculate and save overall averages to evaluation_results/morphscore/{lang}
                if len(out_df) > 0:
                    avg_precision = out_df['precision'].mean()
                    avg_recall = out_df['recall'].mean()
                    avg_f1 = out_df['f1'].mean()
                else:
                    avg_precision = avg_recall = avg_f1 = 0.0
                    
                lang_eval_dir = os.path.join(eval_out_dir, lang)
                os.makedirs(lang_eval_dir, exist_ok=True)
                
                eval_filename = f"{output_model_type}_v{vocab_size}.json"
                
                summary_file = os.path.join(lang_eval_dir, eval_filename)
                
                eval_metrics = {
                    "precision": float(avg_precision),
                    "recall": float(avg_recall),
                    "f1": float(avg_f1)
                }
                
                with open(summary_file, 'w', encoding='utf-8') as sf:
                    json.dump(eval_metrics, sf, indent=2, ensure_ascii=False)
                
                # Add to HF commit operations
                op = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=out_file)
                operations_by_repo[repo_id].append(op)
                
                if len(operations_by_repo[repo_id]) >= args.batch_size:
                    flush_repo(repo_id)
                    
    # Final flush
    flush_repo(repo_id)

if __name__ == "__main__":
    main()
