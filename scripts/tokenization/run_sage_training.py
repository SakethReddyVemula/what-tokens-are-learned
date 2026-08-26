import argparse
import os
import sys
import subprocess

# Special tokens lists
SPECIAL_TOKENS_WP = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
SPECIAL_TOKENS_SP = ["<unk>", "<s>", "</s>"]

# Repo root: these scripts live in scripts/tokenization/, while the external
# tokenizer clones (superbpe/, boundlessbpe/, picky_bpe/, SaGe/) and the
# sibling dataset/ directory are resolved relative to the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_special(token):
    return token in SPECIAL_TOKENS_WP or token in SPECIAL_TOKENS_SP

def to_sage_format(token, model_type):
    """
    Prepares a token for SaGe (raw text bytes).
    """
    if model_type == 'wordpiece':
        if token.startswith("##"):
            return token[2:] # "##ing" -> "ing"
        else:
            return " " + token # "apple" -> " apple" (space at start)
    else: # bpe/unigram (SentencePiece)
        # SentencePiece uses U+2581 (▁) for space
        return token.replace('\u2581', ' ')

def from_sage_format(token_str, model_type):
    """
    Converts a SaGe token (raw text) back to tokenizer format.
    """
    if model_type == 'wordpiece':
        if token_str.startswith(" "):
            return token_str[1:] # " apple" -> "apple"
        else:
            return "##" + token_str # "ing" -> "##ing"
    else: # bpe/unigram
        return token_str.replace(' ', '\u2581')

def convert_vocab_to_hex(input_vocab_path, output_vocab_path, model_type):
    """
    Converts a SentencePiece/Tokenizers vocab file to SaGe's hex format.
    Separates special tokens to keep them safe.
    """
    print(f"Converting vocab from {input_vocab_path} to {output_vocab_path}...")
    
    tokens = []
    special_tokens = []
    
    with open(input_vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            token = parts[0]
            
            if is_special(token):
                special_tokens.append(token)
                continue
            
            # Convert to SaGe format (raw bytes representation)
            sage_token = to_sage_format(token, model_type)
            tokens.append(sage_token)

    # Convert to bytes
    byte_tokens = []
    for t in tokens:
        byte_tokens.append(t.encode('utf-8'))

    # Ensure all 256 single bytes are present
    existing_bytes = set(byte_tokens)
    added_count = 0
    for i in range(256):
        b = bytes([i])
        if b not in existing_bytes:
            byte_tokens.append(b)
            added_count += 1
    
    print(f"Original vocab size (excluding special): {len(tokens)}")
    print(f"Added {added_count} missing single-byte tokens.")
    print(f"Final input vocab size: {len(byte_tokens)}")

    # Write to hex file
    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        for b in byte_tokens:
            f.write(b.hex() + '\n')
    
    print("Conversion complete.")
    return len(byte_tokens), special_tokens

def save_final_vocab(hex_vocab_path, output_path, special_tokens, model_type):
    """
    Reads SaGe hex output, converts to string, re-applies format, adds special tokens, and saves.
    """
    print(f"Processing final vocab from {hex_vocab_path}...")
    
    vocab_tokens = []
    
    # Read hex lines
    with open(hex_vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                # Decode hex to bytes, then to utf-8 string
                token_bytes = bytes.fromhex(line)
                token_str = token_bytes.decode('utf-8')
                
                # Convert back to model specific format
                formatted_token = from_sage_format(token_str, model_type)
                vocab_tokens.append(formatted_token)
            except UnicodeDecodeError:
                # Some bytes might not be valid utf-8 if SaGe composed them strangely, 
                # but usually it respects the corpus. 
                # Or if single bytes added were raw.
                # Use repr or replacement?
                # For tokenizer vocab, we usually need strings.
                print(f"Warning: Could not decode token bytes: {line}")
                continue

    # Combine: Special Characters First (standard convention) + SaGe Vocab
    final_vocab = special_tokens + vocab_tokens
    
    print(f"Saving {len(final_vocab)} tokens to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Format depends on model type somewhat, but list of tokens is common
        # BPE/Unigram usually expects scores, but we don't have SaGe scores here easily
        # (they are in a separate file if at all). 
        # For now, we write just tokens (WordPiece style) or token\t0 (BPE style)
        
        for idx, token in enumerate(final_vocab):
            if model_type == 'wordpiece':
                f.write(f"{token}\n")
            else:
                # SentencePiece usually wants score. We'll use dummy 0 or -idx
                f.write(f"{token}\t0\n")


def run_sage(lang, vocab_size, model_type, sage_dir, dataset_dir, tokenizers_bin_dir):
    # 1. Resolve paths
    input_vocab_filename = f"{lang}_{model_type}_{vocab_size}.vocab"
    input_vocab_path = os.path.join(tokenizers_bin_dir, input_vocab_filename)
    
    if not os.path.exists(input_vocab_path):
        print(f"Error: Input vocab file not found: {input_vocab_path}")
        return

    corpus_path = os.path.join(dataset_dir, lang, f"train.{lang}")
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found: {corpus_path}")
        return

    # Create temp directory
    temp_dir = os.path.join(os.path.dirname(sage_dir), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    experiment_name = f"sage_{lang}_{model_type}_{vocab_size}"
    
    input_work_dir = os.path.join(temp_dir, f"{experiment_name}_inputs")
    os.makedirs(input_work_dir, exist_ok=True)
    
    # 2. Convert vocab to hex
    hex_vocab_path = os.path.join(input_work_dir, "initial_vocab_hex.vocab")
    current_size, special_tokens = convert_vocab_to_hex(input_vocab_path, hex_vocab_path, model_type)
    
    # 3. Define schedule
    target_size = int(vocab_size)
    
    # Adjust target size: The target passed to SaGe is the number of tokens IT manages.
    # It does NOT verify special tokens.
    # So if we want total 10000, and we have 5 special tokens, we should ask SaGe for 9995.
    # But current_size includes added single bytes.
    sage_target_size = target_size - len(special_tokens)
    
    # Ensure schedule is valid (decreasing)
    if sage_target_size >= current_size:
         sage_target_size = current_size - 1 # Force at least one step if possible
    
    schedule = [current_size, sage_target_size]
    emb_schedule = [current_size]
    
    print(f"Running SaGe with schedule: {schedule}")
    
    # 4. Construct Command
    sage_script = os.path.join(sage_dir, "src", "main.py")
    
    env = os.environ.copy()
    sage_src_path = os.path.join(sage_dir, "src")
    env["PYTHONPATH"] = sage_src_path + os.pathsep + env.get("PYTHONPATH", "")

    abs_corpus_path = os.path.abspath(corpus_path)
    abs_hex_vocab_path = os.path.abspath(hex_vocab_path)
    partial_corpus_path = os.path.abspath(os.path.join(input_work_dir, "partial_corpus.txt"))

    cmd = [
        sys.executable, sage_script,
        experiment_name,
        "--corpus_filepath", abs_corpus_path,
        "--initial_vocabulary_filepath", abs_hex_vocab_path,
        "--vocabulary_schedule", *map(str, schedule),
        "--embeddings_schedule", *map(str, emb_schedule),
        "--workers", "4",
        "--partial_corpus_line_number", "50", 
        "--partial_corpus_filepath", partial_corpus_path,
        "--max_len", "16" 
    ]
    
    print("Executing command in temp dir:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True, cwd=temp_dir, env=env)
        print(f"\nSaGe training complete.")
        
        # 5. Process and Copy final files
        final_vocab_src = os.path.join(temp_dir, "results", experiment_name, "sage_vocabs", f"sage_vocab_{sage_target_size}.vocab")
        
        if os.path.exists(final_vocab_src):
            final_vocab_name = f"{lang}_{model_type}_sage_{vocab_size}.vocab"
            final_vocab_dest = os.path.join(tokenizers_bin_dir, final_vocab_name)
            
            save_final_vocab(final_vocab_src, final_vocab_dest, special_tokens, model_type)
            print("Successfully saved final tokenizer.")
        else:
            print(f"Warning: Final vocab file not found at {final_vocab_src}")
            
    except subprocess.CalledProcessError as e:
        print(f"SaGe training failed with error code {e.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SaGe tokenizer using existing tokenizers-bin vocab.")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., eng)")
    parser.add_argument("--vocab_size", type=str, default="10000", help="Vocab size of the input file (e.g. 10000)")
    parser.add_argument("--model_type", type=str, default="bpe", help="Model type of the input file (e.g. bpe)")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizers_bin_dir = os.path.join(REPO_ROOT, "tokenizers-bin")
    sage_dir = os.path.join(REPO_ROOT, "SaGe")
    dataset_dir = os.path.join(os.path.dirname(script_dir), "dataset")
    
    run_sage(args.lang, args.vocab_size, args.model_type, sage_dir, dataset_dir, tokenizers_bin_dir)
