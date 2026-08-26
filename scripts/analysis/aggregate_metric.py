import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Collect metric outputs from local evaluation_results directory for a specific metric and language.")
    parser.add_argument("--metric", required=True, help="Metric to aggregate (e.g. fertility, compression)")
    parser.add_argument("--lang", required=True, help="Language code (e.g. tel, fin)")
    parser.add_argument("--input_dir", default="evaluation_results", help="Base directory where evaluations are stored")
    parser.add_argument("--output_file", default="temp_out.txt", help="File to write the output")
    
    args = parser.parse_args()
    
    target_dir = os.path.join(args.input_dir, args.metric, args.lang)
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} does not exist.")
        return

    results = []
    
    for file in os.listdir(target_dir):
        if file.endswith(".json"):
            file_path = os.path.join(target_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    
                    # Instead of parsing everything arbitrarily, since schemas slightly vary,
                    # just extract the metric value if it matches the key name:
                    # Or print the whole dictionary for brevity.
                    if args.metric in data:
                        val = data[args.metric]
                    # Specific fallbacks if the key is slightly different:
                    elif args.metric == "compression" and "compression_rate" in data:
                        val = data["compression_rate"]
                    elif args.metric == "vocab_size" and "effective_vocab_size" in data:
                        val = data["effective_vocab_size"]
                    elif args.metric == "exponence" and "overall_average_exponence" in data:
                        val = data["overall_average_exponence"]
                    elif getattr(type(data), "__name__") == "dict" and len(data) == 1:
                         # e.g for length_dist and freq_dist
                         val = "JSON Dump Follows..."
                    else:
                         val = data
                    
                except json.JSONDecodeError:
                    val = "Error parsing JSON"

                tokenizer_name = file.replace(".json", "")
                results.append((tokenizer_name, val))
                
    results.sort(key=lambda x: str(x[0]))
    
    with open(args.output_file, "w", encoding="utf-8") as out:
        out.write(f"Results for Metric: {args.metric} | Language: {args.lang}\n")
        out.write("=" * 50 + "\n")
        for tokenizer_name, value in results:
            out.write(f"{tokenizer_name:<30} | {value}\n")
            
    print(f"Results written to {args.output_file}")

if __name__ == "__main__":
    main()
