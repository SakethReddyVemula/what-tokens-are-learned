import argparse
import os
import sentencepiece as spm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
import shutil
import subprocess
import sys
from myte_tokenizer import MyteTokenizer

# Repo root: these scripts live in scripts/tokenization/, while the external
# tokenizer clones (superbpe/, boundlessbpe/, picky_bpe/, SaGe/) and the
# sibling dataset/ directory are resolved relative to the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_input_file(lang, model_type, input_file):
    """Resolves the input file path, using default convention if not provided."""
    if input_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(os.path.dirname(REPO_ROOT), 'dataset')
        # if sslm-bpe / sslm-ulm / sslm-wp is model_type, use presegmented dataset as input
        if model_type in ["sslm-bpe", "sslm-ulm", "sslm-wp"]:
            dataset_dir = os.path.join(dataset_dir, 'presegmented_dataset')
        elif model_type in ["morphbpe", "morphulm", "morphwp"] and lang == "eng":
            dataset_dir = os.path.join(dataset_dir, 'morfessor_presegmented_dataset')
        input_file = os.path.join(dataset_dir, lang, f'train.{lang}')

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    return input_file


def train_sentencepiece_tokenizer(lang, vocab_size, model_type, sp_model_type, input_file, output_dir, character_coverage):
    """
    Trains a SentencePiece tokenizer (BPE or Unigram).
    """
    model_prefix = os.path.join(output_dir, f'{lang}_{model_type}_{vocab_size}')

    print(f"Training SentencePiece tokenizer for {lang}...")
    print(f"  Input: {input_file}")
    print(f"  Model Type: {model_type}")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Character Coverage: {character_coverage}")
    print(f"  Output Prefix: {model_prefix}")

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=sp_model_type,
            character_coverage=character_coverage,
        )
        print(f"Tokenizer trained successfully. Saved to {output_dir}")
    except Exception as e:
        print(f"Error training tokenizer: {e}")
        raise


def train_pathpiece_tokenizer(lang, vocab_size, input_file, output_dir, character_coverage):
    """
    Trains a PathPiece tokenizer by training a SentencePiece BPE model and using its vocab file.
    """
    model_prefix = os.path.join(output_dir, f'{lang}_pathpiece_{vocab_size}')

    print(f"Training PathPiece tokenizer for {lang} (via SentencePiece BPE)...")
    print(f"  Input: {input_file}")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Output Prefix: {model_prefix}")

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=character_coverage,
        )
        # Now convert the SPM `.vocab` to PathPiece hex format
        vocab_file = f"{model_prefix}.vocab"
        hex_tokens = []
        byte_tokens_set = set()
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split('\t')[0]
                
                # Replace SentencePiece space with real space
                token_str = token.replace('\u2581', ' ')
                token_bytes = token_str.encode('utf-8')
                
                if token_bytes not in byte_tokens_set:
                    byte_tokens_set.add(token_bytes)
                    hex_tokens.append(token_bytes.hex())
                    
        # Ensure all 256 single bytes are available as fallback
        for i in range(256):
            b = bytes([i])
            if b not in byte_tokens_set:
                byte_tokens_set.add(b)
                hex_tokens.append(b.hex())
                
        # Overwrite the vocab file with hex encoding
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for hx in hex_tokens:
                f.write(hx + '\n')

        print(f"PathPiece tokenizer generated successfully. Saved to {output_dir}")
    except Exception as e:
        print(f"Error training PathPiece tokenizer: {e}")
        raise


def train_wordpiece_tokenizer(lang, vocab_size, input_file, output_dir, morph=False, sslm=False):
    """
    Trains a WordPiece tokenizer using the HuggingFace tokenizers library.

    The trained tokenizer is saved as a JSON file and a vocab file in output_dir.
    Output naming: {lang}_wordpiece_{vocab_size}.json and {lang}_wordpiece_{vocab_size}.vocab
    """
    if sslm == True:
        output_json = os.path.join(output_dir, f'{lang}_sslm-wp_{vocab_size}.json')
        output_vocab = os.path.join(output_dir, f'{lang}_sslm-wp_{vocab_size}.vocab')
    else:
        if morph == False:
            output_json = os.path.join(output_dir, f'{lang}_wordpiece_{vocab_size}.json')
            output_vocab = os.path.join(output_dir, f'{lang}_wordpiece_{vocab_size}.vocab')
        elif morph == True:
            output_json = os.path.join(output_dir, f'{lang}_morphwp_{vocab_size}.json')
            output_vocab = os.path.join(output_dir, f'{lang}_morphwp_{vocab_size}.vocab')

    print(f"Training WordPiece tokenizer for {lang}...")
    print(f"  Input: {input_file}")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Output: {output_json}")

    try:
        # Initialize a WordPiece tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        # Normalizer: NFKC unicode normalization + lowercase
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase(),
        ])

        # Pre-tokenizer: split on whitespace (like BERT)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Decoder: WordPiece decoder to reconstruct text from tokens
        tokenizer.decoder = decoders.WordPiece()

        # Trainer
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            continuing_subword_prefix="##",
        )

        # Train from the input file
        tokenizer.train([input_file], trainer)

        # Save the tokenizer as JSON
        tokenizer.save(output_json)

        # Also export a human-readable vocab file
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        with open(output_vocab, 'w', encoding='utf-8') as f:
            for token, idx in sorted_vocab:
                f.write(f"{token}\t{idx}\n")

        print(f"WordPiece tokenizer trained successfully. Saved to {output_dir}")
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    except Exception as e:
        print(f"Error training WordPiece tokenizer: {e}")
        raise


def train_morfessor_tokenizer(lang, vocab_size, input_file, output_dir):
    """
    Trains a Morfessor tokenizer using the morfessor python interface.
    The vocab_size parameter is included to maintain the interface, though Morfessor
    typically determines its own morph vocabulary based on the corpus.
    Output naming: {lang}_morfessor_{vocab_size}.bin
    """
    import morfessor
    output_bin = os.path.join(output_dir, f'{lang}_morfessor_{vocab_size}.bin')

    print(f"Training Morfessor tokenizer for {lang}...")
    print(f"  Input: {input_file}")
    print(f"  (Interface matching) Vocab Size: {vocab_size}")
    print(f"  Output: {output_bin}")

    try:
        io = morfessor.MorfessorIO()
        train_data = list(io.read_corpus_file(input_file))

        model = morfessor.BaselineModel()
        model.load_data(train_data)
        model.train_batch()

        io.write_binary_model_file(output_bin, model)
        print(f"Morfessor tokenizer trained successfully. Saved to {output_dir}")

    except Exception as e:
        print(f"Error training Morfessor tokenizer: {e}")
        raise

def train_superbpe_tokenizer(lang, vocab_size, input_file, output_dir, base_vocab_size, stage1_regex, stage2_regex):
    """
    Trains a SuperBPE tokenizer using local scripts in the superbpe folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    superbpe_dir = os.path.join(REPO_ROOT, "superbpe")
    if superbpe_dir not in sys.path:
        sys.path.insert(0, superbpe_dir)
        
    from utils import construct_hf_tokenizer
    
    # Create temp corpus dir with symlink
    temp_corpus_dir = os.path.join(output_dir, f"temp_corpus_{lang}_{vocab_size}")
    os.makedirs(temp_corpus_dir, exist_ok=True)
    
    input_file_abs = os.path.abspath(input_file)
    if not os.path.exists(input_file_abs):
        raise FileNotFoundError(f"Resolved input file does not exist: {input_file_abs}")
        
    temp_input_txt = os.path.join(temp_corpus_dir, f"train_{lang}.txt")
    if not os.path.exists(temp_input_txt):
        os.symlink(input_file_abs, temp_input_txt)
        
    stage1_output_dir = os.path.join(output_dir, f"{lang}_superbpe_stage1_{base_vocab_size}")
    os.makedirs(stage1_output_dir, exist_ok=True)
    
    print(f"--- SuperBPE Stage 1: Training base subword tokenizer (vocab size {base_vocab_size}) ---")
    superbpe_train_script = os.path.join(superbpe_dir, "train_tokenizer.py")
    
    stage1_cmd = [
        sys.executable, superbpe_train_script,
        "--output_dir", os.path.abspath(stage1_output_dir),
        "--corpus_dir", os.path.abspath(temp_corpus_dir),
        "--vocab_size", str(base_vocab_size),
        "--regex_string", stage1_regex
    ]
    print("Running:", " ".join(stage1_cmd))
    subprocess.run(stage1_cmd, check=True)
    
    stage2_output_dir = os.path.join(output_dir, f"{lang}_superbpe_stage2_{vocab_size}")
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    print(f"--- SuperBPE Stage 2: Extending tokenizer (vocab size {vocab_size}) ---")
    shutil.copy(os.path.join(stage1_output_dir, "meta.json"), os.path.join(stage2_output_dir, "meta.json"))
    
    with open(os.path.join(stage1_output_dir, "merges.txt"), "r", encoding="utf-8") as fin:
        merges_lines = fin.readlines()
        
    num_inherit_merges = base_vocab_size
    with open(os.path.join(stage2_output_dir, "merges.txt"), "w", encoding="utf-8") as fout:
        fout.writelines(merges_lines[:num_inherit_merges])
        
    stage2_cmd = [
        sys.executable, superbpe_train_script,
        "--output_dir", os.path.abspath(stage2_output_dir),
        "--vocab_size", str(vocab_size),
        "--regex_string", stage2_regex
    ]
    print("Running:", " ".join(stage2_cmd))
    subprocess.run(stage2_cmd, check=True)
    
    print("--- SuperBPE Constructing HF Tokenizer ---")
    # Change CWD for construct_hf_tokenizer context if needed, but the function takes an absolute path
    construct_hf_tokenizer(stage2_output_dir)
    
    final_output_json = os.path.join(output_dir, f"{lang}_superbpe_{base_vocab_size}_{vocab_size}.json")
    shutil.copy(os.path.join(stage2_output_dir, "tokenizer.json"), final_output_json)
    print(f"SuperBPE tokenizer successfully trained and saved to {final_output_json}")
    
    # cleanup temp
    if os.path.exists(temp_corpus_dir):
        shutil.rmtree(temp_corpus_dir)


def convert_to_jsonl(input_file, jsonl_file):
    import json
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(jsonl_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if line:
                json.dump({"text": line}, fout, ensure_ascii=False)
                fout.write('\n')

def train_boundlessbpe_tokenizer(lang, vocab_size, input_file, output_dir, tau, recalc, patname, blowup):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    boundlessbpe_dir = os.path.join(REPO_ROOT, "boundlessbpe")
    if boundlessbpe_dir not in sys.path:
        sys.path.insert(0, boundlessbpe_dir)
        
    from boundlessbpe import FasterHalfDirectRegexTokenizer
    import boundlessbpe.regexconstants as bperegex

    temp_jsonl_file = os.path.join(output_dir, f"temp_{lang}_{vocab_size}.jsonl")
    print(f"Creating temporary JSONL corpus for BoundlessBPE: {temp_jsonl_file}")
    convert_to_jsonl(input_file, temp_jsonl_file)
    
    num_lines = 0
    with open(temp_jsonl_file, 'r', encoding='utf-8') as f:
        for _ in f:
            num_lines += 1

    pattern = None
    if patname == "ultimate":
        pattern = bperegex.ULTIMATE_PATTERN_V1
    elif patname == "ultimate2":
        pattern = bperegex.ULTIMATE_PATTERN
    elif patname == "gpt2":
        pattern = bperegex.GPT2_SPLIT_PATTERN
    elif patname == "gpt4":
        pattern = bperegex.GPT4_SPLIT_PATTERN
    elif patname == "gpt4o":
        pattern = bperegex.GPT4O_SPLIT_PATTERN
    else:
        raise ValueError(f"Unknown Pattern Name for BoundlessBPE: {patname}")

    print(f"--- Training BoundlessBPE Tokenizer (vocab size {vocab_size}) ---")
    tokenizer = FasterHalfDirectRegexTokenizer(tau, pattern)
    
    outprefix = os.path.join(output_dir, f"{lang}_boundlessbpe_{vocab_size}")
    blowup_bool = bool(blowup)
    
    tokenizer.train(temp_jsonl_file, outprefix, num_lines, vocab_size, recalc, blowup_bool)
    tokenizer.register_special_tokens({"<|endoftext|>": vocab_size})
    
    print(f"Saving BoundlessBPE tokenizer to {outprefix}")
    tokenizer.save(outprefix)
    
    # Test load
    tokenizer2 = FasterHalfDirectRegexTokenizer(tau)
    tokenizer2.load(outprefix + ".model")
    
    if os.path.exists(temp_jsonl_file):
        os.remove(temp_jsonl_file)
    print("BoundlessBPE tokenizer successfully trained.")

def train_pickybpe_tokenizer(lang, vocab_size, input_file, output_dir, threshold):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickybpe_dir = os.path.join(REPO_ROOT, "picky_bpe")
    
    out_model = os.path.join(output_dir, f"{lang}_pickybpe_{vocab_size}.json")
    
    print(f"--- Training PickyBPE Tokenizer (vocab size {vocab_size}, threshold {threshold}) ---")
    trainer_script = os.path.join(pickybpe_dir, "bpe_trainer.py")
    
    cmd = [
        sys.executable, trainer_script,
        "--input_file", os.path.abspath(input_file),
        "--model_file", os.path.abspath(out_model),
        "--vocab_size", str(vocab_size),
        "--threshold", str(threshold)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("PickyBPE tokenizer successfully trained and saved to {out_model}.")

def train_myte_tokenizer(lang, vocab_size, input_file, output_dir):
    print(f"--- Training MYTE Tokenizer (vocab size {vocab_size}) ---")
    tokenizer = MyteTokenizer()
    tokenizer.train(input_file, vocab_size, output_dir, lang)

def train_tokenizer(lang, vocab_size, model_type, input_file, output_dir, character_coverage,
                    superbpe_base_vocab_size=10000, superbpe_stage1_regex="", superbpe_stage2_regex="",
                    boundlessbpe_tau=0.9, boundlessbpe_recalc=1000, boundlessbpe_patname="ultimate2", boundlessbpe_blowup=1,
                    pickybpe_threshold=0.9):
    """
    Trains a tokenizer (SentencePiece BPE/Unigram or HuggingFace WordPiece).

    Args:
        lang (str): Language code (ISO 639-3).
        vocab_size (int): Vocabulary size.
        model_type (str): Model type ('unigram', 'bpe', or 'wordpiece').
        input_file (str): Path to the training data.
        output_dir (str): Directory to save the tokenizer.
        character_coverage (float): Character coverage (used for SentencePiece only).
    """
    input_file = _resolve_input_file(lang, model_type, input_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_type == 'wordpiece':
        train_wordpiece_tokenizer(lang, vocab_size, input_file, output_dir)
    elif model_type == 'sslm-wp':
        train_wordpiece_tokenizer(lang, vocab_size, input_file, output_dir, morph=False, sslm=True)
    elif model_type in ['morphbpe', 'morphulm', 'morphwp']:
        if lang == 'eng':
            temp_corpus_path = input_file
        else:
            import sys
            import tempfile
            try:
                import morfessor
            except ImportError:
                print("Error: morfessor library not installed. Cannot train morph tokenizers.")
                sys.exit(1)
            
            morfessor_model_path = os.path.join(output_dir, f'{lang}_morfessor_10000.bin') # hardcoded 10000 vocab, since morfessor doesn't have a fixed predefined vocabulary as such
            if not os.path.exists(morfessor_model_path):
                print(f"Error: Morfessor model not found at {morfessor_model_path}. Train morfessor first.")
                return
                
            io = morfessor.MorfessorIO()
            morfessor_model = io.read_binary_model_file(morfessor_model_path)
            
            # temp_dir = tempfile.mkdtemp()
            temp_dir = "temp_dir"
            temp_corpus_path = os.path.join(temp_dir, f"temp_morph_corpus_{lang}.txt")
            print(f"Creating temporary morfessor-pretokenized corpus at {temp_corpus_path}...")
        
        try:
            if lang != "eng" and not os.path.exists(temp_corpus_path):
                with open(input_file, 'r', encoding='utf-8') as f_in, \
                    open(temp_corpus_path, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        words = line.strip().split()
                        segmented_words = []
                        for word in words:
                            morphs = morfessor_model.viterbi_segment(word)[0]
                            segmented_words.append(" ".join(morphs))
                        f_out.write(" ".join(segmented_words) + "\n")
                    
            if model_type == 'morphwp':
                train_wordpiece_tokenizer(lang, vocab_size, temp_corpus_path, output_dir, morph=True)
                # rename files # DO NOT DO THIS
                # os.rename(os.path.join(output_dir, f'{lang}_wordpiece_{vocab_size}.json'), os.path.join(output_dir, f'{lang}_morphwp_{vocab_size}.json'))
                # os.rename(os.path.join(output_dir, f'{lang}_wordpiece_{vocab_size}.vocab'), os.path.join(output_dir, f'{lang}_morphwp_{vocab_size}.vocab'))
            else:
                sp_model_type = 'bpe' if model_type == 'morphbpe' else 'unigram'
                train_sentencepiece_tokenizer(lang, vocab_size, model_type, sp_model_type, temp_corpus_path, output_dir, character_coverage)
                # rename files # DO NOT DO THIS
                # os.rename(os.path.join(output_dir, f'{lang}_{sp_model_type}_{vocab_size}.model'), os.path.join(output_dir, f'{lang}_{model_type}_{vocab_size}.model'))
                # os.rename(os.path.join(output_dir, f'{lang}_{sp_model_type}_{vocab_size}.vocab'), os.path.join(output_dir, f'{lang}_{model_type}_{vocab_size}.vocab'))
                
        finally:
            print(f"Cleaning skipped, to do later manually...")
            # print(f"Cleaning up temporary directory {temp_dir}...")
            # shutil.rmtree(temp_dir, ignore_errors=True)
            
    elif model_type == 'morfessor':
        train_morfessor_tokenizer(lang, vocab_size, input_file, output_dir)
    elif model_type == 'superbpe':
        train_superbpe_tokenizer(lang, vocab_size, input_file, output_dir, superbpe_base_vocab_size, superbpe_stage1_regex, superbpe_stage2_regex)
    elif model_type == 'boundlessbpe':
        train_boundlessbpe_tokenizer(lang, vocab_size, input_file, output_dir, boundlessbpe_tau, boundlessbpe_recalc, boundlessbpe_patname, boundlessbpe_blowup)
    elif model_type == 'pickybpe':
        train_pickybpe_tokenizer(lang, vocab_size, input_file, output_dir, pickybpe_threshold)
    elif model_type == 'pathpiece':
        train_pathpiece_tokenizer(lang, vocab_size, input_file, output_dir, character_coverage)
    elif model_type == 'myte':
        train_myte_tokenizer(lang, vocab_size, input_file, output_dir)
    elif model_type == 'sslm-bpe':
        sp_model_type = 'bpe'
        train_sentencepiece_tokenizer(lang, vocab_size, model_type, sp_model_type, input_file, output_dir, character_coverage)
    elif model_type == 'sslm-ulm':
        sp_model_type = 'unigram'
        train_sentencepiece_tokenizer(lang, vocab_size, model_type, sp_model_type, input_file, output_dir, character_coverage)
    else:
        train_sentencepiece_tokenizer(lang, vocab_size, model_type, model_type, input_file, output_dir, character_coverage)

def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument("--lang", type=str, required=True, help="Language code (ISO 639-3).")
    parser.add_argument("--vocab_size", type=int, default=25000, help="Vocabulary size (T: final vocab size).")
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "wordpiece", "morfessor", "superbpe", "boundlessbpe", "pickybpe", "pathpiece", "myte", "morphbpe", "morphulm", "morphwp", "sslm-bpe", "sslm-ulm", "sslm-wp"], help="Model type.")
    parser.add_argument("--input_file", type=str, help="Path to training data. Defaults to ../dataset/{lang}/train.{lang}")
    parser.add_argument("--output_dir", type=str, default="tokenizers-bin", help="Directory to save the tokenizer.")
    parser.add_argument("--character_coverage", type=float, default=1.0, help="Character coverage.")
    parser.add_argument("--superbpe_base_vocab_size", type=int, default=10000, help="Base vocab size for SuperBPE (t: vocab size of subword stage).")
    parser.add_argument("--superbpe_stage1_regex", type=str, default="")
    parser.add_argument("--superbpe_stage2_regex", type=str, default="")
    parser.add_argument("--boundlessbpe_tau", type=float, default=0.9, help="Tau parameter for BoundlessBPE.")
    parser.add_argument("--boundlessbpe_recalc", type=int, default=1000, help="Recalc parameter for BoundlessBPE.")
    parser.add_argument("--boundlessbpe_patname", type=str, default="ultimate2", help="Pattern name for BoundlessBPE.")
    parser.add_argument("--boundlessbpe_blowup", type=int, default=1, choices=[0, 1], help="Blowup parameter for BoundlessBPE.")
    parser.add_argument("--pickybpe_threshold", type=float, default=0.9, help="IoS threshold for PickyBPE.")

    args = parser.parse_args()

    train_tokenizer(
        args.lang,
        args.vocab_size,
        args.model_type,
        args.input_file,
        args.output_dir,
        args.character_coverage,
        args.superbpe_base_vocab_size,
        args.superbpe_stage1_regex,
        args.superbpe_stage2_regex,
        args.boundlessbpe_tau,
        args.boundlessbpe_recalc,
        args.boundlessbpe_patname,
        args.boundlessbpe_blowup,
        args.pickybpe_threshold
    )

if __name__ == "__main__":
    main()
