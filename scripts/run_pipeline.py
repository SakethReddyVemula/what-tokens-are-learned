import argparse
import subprocess
import sys
import os
import shlex

def run_command(cmd):
    """Executes a shell command and prints it."""
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Exit code: {e.returncode}")
        # We might want to continue pipeline even if one fails? 
        # For now, let's raise to stop on error, or maybe just return False?
        # User requested "easier works", so maybe stopping on error is better to debug, 
        # or maybe logging error and continuing. Let's raise for now to be safe.
        raise

def run_pipeline(langs, vocab_sizes, model_types, dataset_split, analysis_suite_path, morphscore=False, morphscore_data_dir="morphscore_data",
                 superbpe_base_vocab_size=10000, superbpe_stage1_regex="", superbpe_stage2_regex="",
                 boundlessbpe_tau=0.9, boundlessbpe_recalc=1000, boundlessbpe_patname="ultimate2", boundlessbpe_blowup=1,
                 pickybpe_threshold=0.9):
    """
    Runs the training and evaluation pipeline.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "tokenization", "train_tokenizer.py")
    eval_script = os.path.join(script_dir, "evaluation", "evaluate_tokenizer.py")

    for lang in langs:
        print(f"\n{'='*40}")
        print(f"Processing Language: {lang}")
        print(f"{'='*40}\n")
        
        for model_type in model_types:
            for vocab_size in vocab_sizes:
                print(f"--- Configuration: Lang={lang}, Type={model_type}, Vocab={vocab_size} ---")
                
                # 1. Train Tokenizer
                print(f"Step 1: Training Tokenizer...")
                # Use list for subprocess instead of string to avoid shell quoting issues
                # But since run_command uses shell=True and joins with space, we need manual quoting if we keep that design.
                # Interactive shell might be tricky with spaces.
                # Better approach: modify run_command to accept list and shell=False (if possible) or quote properly.
                
                # Let's quote the paths
                train_cmd_str = f'"{sys.executable}" "{train_script}" --lang {lang} --vocab_size {vocab_size} --model_type {model_type}'
                if model_type == "superbpe":
                    train_cmd_str += f' --superbpe_base_vocab_size {superbpe_base_vocab_size}'
                    train_cmd_str += f' --superbpe_stage1_regex {shlex.quote(superbpe_stage1_regex)}'
                    train_cmd_str += f' --superbpe_stage2_regex {shlex.quote(superbpe_stage2_regex)}'
                
                if model_type == "boundlessbpe":
                    train_cmd_str += f' --boundlessbpe_tau {boundlessbpe_tau}'
                    train_cmd_str += f' --boundlessbpe_recalc {boundlessbpe_recalc}'
                    train_cmd_str += f' --boundlessbpe_patname {shlex.quote(boundlessbpe_patname)}'
                    train_cmd_str += f' --boundlessbpe_blowup {boundlessbpe_blowup}'
                
                if model_type == "pickybpe":
                    train_cmd_str += f' --pickybpe_threshold {pickybpe_threshold}'
                
                try:
                    run_command(train_cmd_str)
                except subprocess.CalledProcessError:
                    print("Skipping evaluation due to training failure.")
                    continue

                # # 2. Evaluate Tokenizer
                # print(f"Step 2: Evaluating Tokenizer...")
                # eval_cmd_str = f'"{sys.executable}" "{eval_script}" --lang {lang} --split {dataset_split} --vocab_size {vocab_size} --model_type {model_type}'
                
                # if analysis_suite_path:
                #     eval_cmd_str += f' --analysis_suite_path "{analysis_suite_path}"'

                # if morphscore:
                #     eval_cmd_str += ' --morphscore'
                #     if morphscore_data_dir:
                #         eval_cmd_str += f' --morphscore_data_dir "{morphscore_data_dir}"'

                # try:
                #     run_command(eval_cmd_str)
                # except subprocess.CalledProcessError:
                #     print("Evaluation failed.")
                #     continue
                
                print(f"Completed configuration: {lang} {model_type} {vocab_size}\n")

def main():
    parser = argparse.ArgumentParser(description="Run tokenizer training and evaluation pipeline.")
    
    parser.add_argument("--langs", nargs="+", required=True, help="List of language codes to process.")
    parser.add_argument("--vocab_sizes", nargs="+", type=int, default=[25000], help="List of vocabulary sizes (T: final vocab size).")
    parser.add_argument("--model_types", nargs="+", default=["unigram", "bpe"], choices=["unigram", "bpe", "wordpiece", "morfessor", "superbpe", "boundlessbpe", "pickybpe", "pathpiece", "myte", "morphbpe", "morphulm", "morphwp", "sslm-bpe", "sslm-ulm", "sslm-wp"], help="List of model types.")
    
    parser.add_argument("--superbpe_base_vocab_size", type=int, default=10000, help="Base vocabulary size for SuperBPE stage 1 (t: vocab size of subword stage).")
    parser.add_argument("--superbpe_stage1_regex", type=str, default=r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+", help="Regex for Stage 1 of SuperBPE.")
    parser.add_argument("--superbpe_stage2_regex", type=str, default=r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)", help="Regex for Stage 2 of SuperBPE.")
    
    parser.add_argument("--boundlessbpe_tau", type=float, default=0.9, help="Tau parameter for BoundlessBPE.")
    parser.add_argument("--boundlessbpe_recalc", type=int, default=1000, help="Recalc parameter for BoundlessBPE.")
    parser.add_argument("--boundlessbpe_patname", type=str, default="ultimate2", help="Pattern name for BoundlessBPE.")
    parser.add_argument("--boundlessbpe_blowup", type=int, default=1, choices=[0, 1], help="Blowup parameter for BoundlessBPE.")

    parser.add_argument("--pickybpe_threshold", type=float, default=0.9, help="IoS threshold for PickyBPE.")
    
    parser.add_argument("--dataset_split", type=str, default="valid", help="Dataset split for evaluation (default: valid).")
    parser.add_argument("--analysis_suite_path", type=str, help="Path to tokenizer-analysis-suite.")
    parser.add_argument("--morphscore", action="store_true", help="Enable MorphScore evaluation.")
    parser.add_argument("--morphscore_data_dir", type=str, default="morphscore_data", help="Directory containing MorphScore data.")
    
    args = parser.parse_args()

    run_pipeline(
        args.langs,
        args.vocab_sizes,
        args.model_types,
        args.dataset_split,
        args.analysis_suite_path,
        args.morphscore,
        args.morphscore_data_dir,
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
