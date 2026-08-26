import argparse
import os
import sentencepiece as spm
from tokenizers import Tokenizer

# Repo root: scripts live two levels below it, while data directories
# (tokenizers-bin/, evaluation_results/, ...) and external tokenizer
# clones live at the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def apply_tokenizer(lang, split, model_path, input_file, output_file, vocab_size, model_type):
    """
    Applies a trained SentencePiece tokenizer to a file.
    """

    # Determine Model Path
    if not model_path:
        if model_type == 'wordpiece':
            model_filename = f'{lang}_{model_type}_{vocab_size}.json'
        else:
            model_filename = f'{lang}_{model_type}_{vocab_size}.model'
        # Default assumption: models are in tokenizers-bin/ in the same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(REPO_ROOT, 'tokenizers-bin', model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Determine Input File
    if not input_file:
        # Default assumption: dataset/{lang}/{split}.{lang}
        # Assuming script is in TokTrainSuite/ and dataset/ is in valid relative location (../dataset)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(os.path.dirname(script_dir), 'dataset')
        input_file = os.path.join(dataset_dir, lang, f'{split}.{lang}')

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Determine Output File
    if not output_file:
        # Default assumption: tokenized_data/{lang}/{split}.{model_type}.{vocab_size}.txt
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(REPO_ROOT, 'tokenized_data', lang)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'{split}.{model_type}.{vocab_size}.txt')
    else:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print(f"Applying tokenizer...")
    print(f"  Language: {lang}")
    print(f"  Split: {split}")
    print(f"  Model: {model_path}")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")

    if model_type == 'wordpiece':
        wp_tokenizer = Tokenizer.from_file(model_path)
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    text = line.strip()
                    encoding = wp_tokenizer.encode(text)
                    tokens = encoding.tokens
                    tokenized_line = ' '.join(tokens)
                    f_out.write(tokenized_line + '\n')
            print(f"Tokenization complete. Output saved to {output_file}")
        except Exception as e:
            print(f"Error applying tokenizer: {e}")
            raise
    else:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)

        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    # remove newline to tokenize, then join tokens
                    text = line.strip()
                    tokens = sp.encode_as_pieces(text)
                    tokenized_line = ' '.join(tokens)
                    f_out.write(tokenized_line + '\n')
                    
            print(f"Tokenization complete. Output saved to {output_file}")

        except Exception as e:
            print(f"Error applying tokenizer: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Apply a SentencePiece tokenizer to a dataset.")
    parser.add_argument("--lang", type=str, required=True, help="Language code (ISO 639-3).")
    parser.add_argument("--split", type=str, required=True, help="Dataset split (train, valid, test).")
    
    # Arguments to infer model path if not provided
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size (used to find model).")
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "wordpiece"], help="Model type (used to find model).")
    
    parser.add_argument("--model_path", type=str, help="Path to the trained model file. Overrides automatic detection.")
    parser.add_argument("--input_file", type=str, help="Path to input file. Overrides automatic detection.")
    parser.add_argument("--output_file", type=str, help="Path to output file. Overrides automatic detection.")

    args = parser.parse_args()

    apply_tokenizer(
        args.lang,
        args.split,
        args.model_path,
        args.input_file,
        args.output_file,
        args.vocab_size,
        args.model_type
    )

if __name__ == "__main__":
    main()
