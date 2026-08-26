import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract and convert final SaGe vocab")
    parser.add_argument("--langs", nargs="+", required=True, help="List of language codes")
    args = parser.parse_args()
    
    out_dir = "tokenizers-bin"
    os.makedirs(out_dir, exist_ok=True)
    
    for lang in args.langs:
        in_path = f"SaGe/results/sage_train_{lang}/sage_vocabs/sage_vocab_9900.vocab"
        out_path = os.path.join(out_dir, f"{lang}_sage_10000.vocab")
        
        if not os.path.exists(in_path):
            print(f"Warning: {in_path} does not exist. Skipping {lang}.")
            continue
            
        print(f"Converting {in_path} -> {out_path}")
        with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                hex_str = line.strip()
                if not hex_str:
                    continue
                try:
                    # Parse the hex into bytes
                    token_bytes = bytes.fromhex(hex_str)
                    # Decode with replace to avoid failing on invalid utf-8 sequences
                    token_str = token_bytes.decode('utf-8', errors='replace')
                    
                    import json
                    f_out.write(json.dumps(token_str) + "\n")
                except ValueError as e:
                    print(f"Warning: Could not parse hex string '{hex_str}': {e}")
                    
if __name__ == "__main__":
    main()
