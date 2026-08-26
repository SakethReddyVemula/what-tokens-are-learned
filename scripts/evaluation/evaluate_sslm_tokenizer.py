import argparse
import os
import json
import shutil
import numpy as np
from collections import defaultdict
import re
from huggingface_hub import hf_hub_download, HfApi

def compute_metrics(file_path):
    total_words = 0
    total_subwords = 0
    token_freq = defaultdict(int)
    token_neighbors = defaultdict(set)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                words = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            sentence_subwords = []
            for word_subwords in words:
                total_words += 1
                total_subwords += len(word_subwords)
                for sw in word_subwords:
                    token_freq[sw] += 1
                    sentence_subwords.append(sw)
                    
            n = len(sentence_subwords)
            for i, token in enumerate(sentence_subwords):
                start = max(0, i - 2)
                end = min(n, i + 3) # Window of 5 (+/- 2 on each side)
                for j in range(start, end):
                    if i != j:
                        token_neighbors[token].add(sentence_subwords[j])
                        
    return total_words, total_subwords, token_freq, token_neighbors

def main():
    parser = argparse.ArgumentParser(description="Evaluate SSLM tokenizers from segmented data on Hugging Face")
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--step", type=str, default=None, help="Specific step/epoch to evaluate (e.g., 24 or 1_50). If not provided, evaluate all available.")
    parser.add_argument("--eval_all_steps", action="store_true", help="If no step is specified, evaluate all steps. Default is to evaluate only the highest step.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--hf_repo", type=str, default="SakethVemula/sslm-corpus-segments")
    parser.add_argument("--metrics", nargs="+", required=True, help="List of metrics to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    
    args = parser.parse_args()

    # Create output directories for each metric
    for metric in args.metrics:
        os.makedirs(os.path.join(args.output_dir, metric), exist_ok=True)
        
    api = HfApi()
    print(f"Fetching file list from repository: {args.hf_repo}...")
    try:
        all_repo_files = api.list_repo_files(repo_id=args.hf_repo, repo_type="dataset")
    except Exception as e:
        print(f"Failed to fetch file list from {args.hf_repo}: {e}")
        return

    for lang in args.langs:
        # Filter files for the current language
        lang_files = [f for f in all_repo_files if f.startswith(f"{lang}/") and f.endswith(".jsonl")]
        
        # If split is test, it might or might not be in the name based on the user's setup, but let's assume
        # the filenames are like: seg_corpus_{lang}_checkpoint{step}.jsonl
        
        target_files = []
        for file in lang_files:
            filename = os.path.basename(file)
            # Match seg_corpus_{lang}_checkpoint{step}.jsonl or similar
            match = re.search(r'checkpoint_?([\d_]+)', filename)
            if match:
                file_step = match.group(1)
                target_files.append((file, file_step))
        
        if not target_files:
            print(f"No matching SSLM segmentation files found for language {lang}" + (f" and step {args.step}." if args.step else "."))
            continue
            
        if args.step:
            target_files = [(f, s) for f, s in target_files if s == args.step]
            if not target_files:
                print(f"Skipping language {lang}. No file found for specified step {args.step}.")
                continue
        elif args.eval_all_steps:
            print(f"No specific step provided for language {lang}. Evaluating ALL available steps.")
        else:
            # If no step is provided, find the highest step
            def parse_step(step_str):
                # Handle cases like "24" and "1_50"
                parts = step_str.split('_')
                if len(parts) == 1:
                    return (int(parts[0]), 0) # Just epoch
                elif len(parts) == 2:
                    return (int(parts[0]), int(parts[1])) # Epoch and step
                else:
                    return (0, 0)
                    
            highest_file, highest_step = max(target_files, key=lambda x: parse_step(x[1]))
            target_files = [(highest_file, highest_step)]
            print(f"No step provided for language {lang}. Selected highest step: {highest_step}")
            
            
        print(f"\nFound {len(target_files)} file(s) for language {lang}:")
        for f, step in target_files:
            print(f" - {f} (Step: {step})")
            
        for path_in_repo, file_step in target_files:
            output_model_type = "sslm"
            print(f"\nEvaluating {lang} - SSLM Step/Epoch {file_step}")
            
            # Download from HF
            try:
                local_file = hf_hub_download(
                    repo_id=args.hf_repo,
                    filename=path_in_repo,
                    repo_type="dataset",
                    cache_dir="temp_cache"
                )
            except Exception as e:
                print(f"Could not download {path_in_repo} from {args.hf_repo}: {e}")
                continue
                
            print(f"Downloaded to {local_file}. Computing metrics...")
            
            # Compute stats
            total_words, total_subwords, token_freq, token_neighbors = compute_metrics(local_file)
            
            # Process metrics
            for metric in args.metrics:
                metric_dir = os.path.join(args.output_dir, metric, lang)
                os.makedirs(metric_dir, exist_ok=True)
                # Save out_file with checkpoint step and split
                out_file = os.path.join(metric_dir, f"{output_model_type}_checkpoint_{file_step}_{args.split}.json")
                
                if metric == "fertility":
                    fert = total_subwords / total_words if total_words > 0 else 0
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump({"fertility": fert, "total_subwords": total_subwords, "total_words": total_words}, f, indent=4)
                
                elif metric == "compression":
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump({"compression_rate": total_subwords, "total_subwords": total_subwords}, f, indent=4)
                        
                elif metric == "freq_dist":
                    # Sort token freq by frequency for readability
                    sorted_freq = dict(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump(sorted_freq, f, indent=4, ensure_ascii=False)
                        
                elif metric == "length_dist":
                    length_freq = defaultdict(int)
                    for t, c in token_freq.items():
                        length_freq[len(t)] += c
                    sorted_length = dict(sorted(length_freq.items()))
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump(sorted_length, f, indent=4)
                        
                elif metric == "exponence":
                    avg_exponence_per_type = {t: len(neighbors) for t, neighbors in token_neighbors.items()}
                    overall_avg = sum(avg_exponence_per_type.values()) / len(avg_exponence_per_type) if avg_exponence_per_type else 0
                    # sorting for better readability
                    sorted_expo = dict(sorted(avg_exponence_per_type.items(), key=lambda item: item[1], reverse=True))
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "overall_average_exponence": overall_avg,
                            "per_token_exponence": sorted_expo
                        }, f, indent=4, ensure_ascii=False)
                        
                elif metric == "vocab_size":
                    eff_size = len(token_freq)
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump({"effective_vocab_size": eff_size}, f, indent=4)

                elif metric == "ttr":
                    ttr_val = len(token_freq) / total_subwords if total_subwords > 0 else 0
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "ttr": ttr_val,
                            "unique_types": len(token_freq),
                            "total_tokens": total_subwords
                        }, f, indent=4)
                        
                elif metric in ["renyi_entropy", "renyi_efficiency", "shannon_entropy", "shannon_efficiency"]:
                    # Compute probabilities
                    freqs = list(token_freq.values())
                    probs = np.array(freqs) / total_subwords if total_subwords > 0 else np.array([])
                    vocab_size = len(token_freq)
                    
                    val = 0.0
                    if len(probs) > 0:
                        if metric == "renyi_entropy":
                            power = 3.0
                            scale = 1 / (1 - power)
                            val = scale * np.log2(np.sum(probs ** power))
                        elif metric == "renyi_efficiency":
                            power = 3.0
                            scale = 1 / (1 - power)
                            val = scale * np.log2(np.sum(probs ** power)) / np.log2(vocab_size) if vocab_size > 1 else 0
                        elif metric == "shannon_entropy":
                            val = -np.sum(probs * np.log2(probs))
                        elif metric == "shannon_efficiency":
                            val = -np.sum(probs * np.log2(probs)) / np.log2(vocab_size) if vocab_size > 1 else 0

                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            metric: val,
                            "vocab_size": vocab_size,
                            "total_subwords": total_subwords
                        }, f, indent=4)
            
            # Delete cache immediately to save memory (delete the entire temp_cache dir)
            shutil.rmtree("temp_cache", ignore_errors=True)
            print(f"Finished evaluating and cleared local segmentations for {lang} - SSLM Step {file_step}")

if __name__ == "__main__":
    main()
