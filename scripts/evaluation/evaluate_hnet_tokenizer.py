import argparse
import os
import json
import shutil
import numpy as np
from collections import defaultdict
from huggingface_hub import hf_hub_download

def compute_metrics(file_path):
    total_words = 0
    total_subwords = 0
    token_freq = defaultdict(int)
    token_neighbors = defaultdict(set)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {file_path}: {e}")
            return 0, 0, defaultdict(int), defaultdict(set)
            
        for sentence_group in data:
            if not sentence_group:
                continue
            # sentence_group is e.g. [ [ "\\xfe", "It", "'s " ... ] ]
            sentence_subwords = []
            
            for word_subwords in sentence_group:
                # Based on the H-Net JSON format, words aren't strictly separated out as lists.
                # Instead, word_subwords seems to be a single sequence of tokens for an entire sequence or chunk.
                # We need to filter out the `\xfe` start token.
                filtered_subwords = [sw for sw in word_subwords if sw != "\\xfe"]
                
                for sw in filtered_subwords:
                    token_freq[sw] += 1
                    sentence_subwords.append(sw)
                
                # To calculate words, since space is inside the token (e.g., "easy ", "eno"), 
                # we must reconstruct the original text and count spaces to estimate total words.
                # Or count tokens ending in space.
                reconstructed_text = "".join(filtered_subwords)
                # Count total words: roughly number of spaces + 1 (last word doesn't have a space usually)
                # If text is empty, it's 0.
                words_in_seq = len(reconstructed_text.split()) if reconstructed_text.strip() else 0
                total_words += words_in_seq
                total_subwords += len(filtered_subwords)
                
            n = len(sentence_subwords)
            for i, token in enumerate(sentence_subwords):
                start = max(0, i - 2)
                end = min(n, i + 3) # Window of 5 (+/- 2 on each side)
                for j in range(start, end):
                    if i != j:
                        token_neighbors[token].add(sentence_subwords[j])
                        
    return total_words, total_subwords, token_freq, token_neighbors

def main():
    parser = argparse.ArgumentParser(description="Evaluate H-Net tokenizers from segmented data on Hugging Face")
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--hf_repo", type=str, default="SakethVemula/hnet-segments")
    parser.add_argument("--metrics", nargs="+", required=True, help="List of metrics to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    
    args = parser.parse_args()

    # Create output directories for each metric
    for metric in args.metrics:
        os.makedirs(os.path.join(args.output_dir, metric), exist_ok=True)

    for lang in args.langs:
        output_model_type = "hnet"
        # H-Net relies on checking specifically this file: seg_{lang}_best_model.pt.json
        filename = f"seg_{lang}_best_model.pt.json"
        path_in_repo = f"{lang}/{filename}"
        
        print(f"\nEvaluating {lang} - H-Net Best Model")
        
        # Download from HF
        try:
            local_file = hf_hub_download(
                repo_id=args.hf_repo,
                filename=path_in_repo,
                repo_type="dataset",
                cache_dir="temp_cache_hnet"
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
            out_file = os.path.join(metric_dir, f"{output_model_type}_best_model_{args.split}.json")
            
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
        shutil.rmtree("temp_cache_hnet", ignore_errors=True)
        print(f"Finished evaluating and cleared local segmentations for {lang} - H-Net Best Model")

if __name__ == "__main__":
    main()
