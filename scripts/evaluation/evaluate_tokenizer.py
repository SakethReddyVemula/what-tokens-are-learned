import argparse
import os
import json
import subprocess
import sys
import tempfile

# Repo root: scripts live two levels below it, while data directories
# (tokenizers-bin/, evaluation_results/, ...) and external tokenizer
# clones live at the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_tokenizer(lang, split, model_path, input_file, output_dir, vocab_size, model_type, analysis_suite_path, run_grouped, morphscore=False, morphscore_data_dir="morphscore_data"):
    """
    Evaluates a trained SentencePiece tokenizer using tokenizer-analysis-suite.
    """

    # 1. Resolve Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model Path
    if not model_path:
        if model_type == 'wordpiece':
            model_filename = f'{lang}_{model_type}_{vocab_size}.json'
        else:
            model_filename = f'{lang}_{model_type}_{vocab_size}.model'
        model_path = os.path.join(REPO_ROOT, 'tokenizers-bin', model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_abs_path = os.path.abspath(model_path)

    # Input File (Evaluation Data)
    if not input_file:
        # Default: ../dataset/{lang}/{split}.{lang}
        dataset_dir = os.path.join(os.path.dirname(script_dir), 'dataset')
        input_file = os.path.join(dataset_dir, lang, f'{split}.{lang}')

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    input_abs_path = os.path.abspath(input_file)

    # Analysis Suite Path
    if not analysis_suite_path:
        analysis_suite_path = os.path.join(os.path.dirname(script_dir), 'tokenizer-analysis-suite')
    
    analysis_script = os.path.join(analysis_suite_path, 'scripts', 'run_tokenizer_analysis.py')
    if not os.path.exists(analysis_script):
        raise FileNotFoundError(f"Analysis script not found at: {analysis_script}\nPlease check --analysis_suite_path.")

    # Output Directory
    if not output_dir:
        output_dir = os.path.join(REPO_ROOT, 'evaluation_results', f'{lang}_{model_type}_{vocab_size}_{split}')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_abs_path = os.path.abspath(output_dir)

    print(f"Preparing evaluation for {lang}...")
    print(f"  Model: {model_abs_path}")
    print(f"  Data: {input_abs_path}")
    print(f"  Output: {output_abs_path}")

    # 2. Generate Configuration Files
    # We will create temporary config files
    
    # Add suite to sys.path to import constants
    if analysis_suite_path:
        abs_suite_path = os.path.abspath(analysis_suite_path)
        if abs_suite_path not in sys.path:
            sys.path.append(abs_suite_path)

    # Determine language code for analysis suite
    # MorphScore requires specific codes (e.g. eng_Latn) matching filenames in FLORES_TO_MS_FILES
    # Try to import mapping from suite
    analysis_lang_code = lang
    try:
        from tokenizer_analysis.loaders.constants import FLORES_to_ISO639_2
        # Create reverse mapping: ISO 639-3 -> FLORES code (e.g., 'eng' -> 'eng_Latn')
        # Note: This simple inversion takes the last encountered mapping if duplicates exist.
        ISO639_3_to_FLORES = {v: k for k, v in FLORES_to_ISO639_2.items()}
        
        if lang in ISO639_3_to_FLORES:
            analysis_lang_code = ISO639_3_to_FLORES[lang]
            print(f"Mapped language '{lang}' to '{analysis_lang_code}' for analysis.")
    except ImportError:
        print("Warning: Could not import constants from tokenizer-analysis-suite. Using original language code.")
    except Exception as e:
        print(f"Warning: Error during language mapping: {e}")

    # Tokenizer Config
    tokenizer_name = f"{lang}_{model_type}_{vocab_size}"
    tokenizer_config = {
        tokenizer_name: {
            "class": "wordpiece" if model_type == "wordpiece" else "sentencepiece",
            "path": model_abs_path
        }
    }

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a symlink to input_file with .txt extension so tokenizer-analysis-suite handles it without warnings
            # Using absolute path for symlink target
            temp_input_path = os.path.join(temp_dir, f"{lang}.txt")
            os.symlink(input_abs_path, temp_input_path)

            tok_config_path = os.path.join(temp_dir, 'tokenizer_config.json')
            lang_config_path = os.path.join(temp_dir, 'language_config.json')

            # Language Config
            # Use the temp input path which has .txt extension
            language_config = {
                "languages": {
                    analysis_lang_code: {
                        "name": lang, # Use the original lang code as name for simplicity, or could map to full name
                        "iso_code": lang,
                        "data_path": temp_input_path 
                    }
                }
            }

            with open(tok_config_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
            
            with open(lang_config_path, 'w') as f:
                json.dump(language_config, f, indent=2)

            # 3. Construct Command
            cmd = [
                sys.executable, 
                analysis_script,
                '--tokenizer-config', tok_config_path,
                '--language-config', lang_config_path,
                '--output-dir', output_abs_path
            ]
            
            if morphscore:
                cmd.append("--morphscore")
                if morphscore_data_dir:
                    cmd.append(f"--morphscore-data-dir={os.path.abspath(morphscore_data_dir)}")
            
            # The suite might default to creating plots using Tkinter which fails on headless.
            # We should probably set MPLBACKEND=Agg env var if not already set.
            env = os.environ.copy()
            if 'MPLBACKEND' not in env:
                env['MPLBACKEND'] = 'Agg'
            
            # Since the suite is installed as a package (editable) or assumed to be in path?
            # The user prompt said: "The scripts are present in @[tokenizer-analysis-suite]"
            # It didn't explicitly say it is installed in the environment.
            # If it's not installed via pip, we might need to add it to PYTHONPATH.
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{os.path.abspath(analysis_suite_path)}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = os.path.abspath(analysis_suite_path)

            print(f"Running analysis script: {' '.join(cmd)}")
            
            # Execute
            subprocess.run(cmd, check=True, env=env)
            
            print(f"Evaluation complete. Results saved to {output_dir}")

    except subprocess.CalledProcessError as e:
        print(f"Error running analysis script: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a tokenizer using tokenizer-analysis-suite.")
    parser.add_argument("--lang", type=str, required=True, help="Language code (ISO 639-3).")
    parser.add_argument("--split", type=str, default="valid", help="Dataset split to evaluate on (valid, test).")
    
    # Inferred arguments
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size (used to find model).")
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "wordpiece"], help="Model type (used to find model).")
    
    parser.add_argument("--morphscore", action="store_true", help="Enable MorphScore evaluation.")
    parser.add_argument("--morphscore_data_dir", type=str, default="morphscore_data", help="Directory containing MorphScore data.")

    # Overrides
    parser.add_argument("--model_path", type=str, help="Path to the trained model file.")
    parser.add_argument("--input_file", type=str, help="Path to evaluation data file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save results.")
    
    parser.add_argument("--analysis_suite_path", type=str, help="Path to tokenizer-analysis-suite repository.")
    parser.add_argument("--run_grouped", action='store_true', help="Run grouped analysis (not relevant for single language usually).")

    args = parser.parse_args()

    evaluate_tokenizer(
        args.lang,
        args.split,
        args.model_path,
        args.input_file,
        args.output_dir,
        args.vocab_size,
        args.model_type,
        args.analysis_suite_path,
        args.run_grouped,
        args.morphscore,
        args.morphscore_data_dir
    )

if __name__ == "__main__":
    main()
