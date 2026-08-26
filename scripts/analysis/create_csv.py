import os
import json
import csv
import argparse
import re

LANG_INFO = {
    "fin": ("Finnish", "latn"),
    "hun": ("Hungarian", "latn"),
    "mal": ("Malayalam", "mlym"),
    "tam": ("Tamil", "taml"),
    "tel": ("Telugu", "telu"),
    "kir": ("Kyrgyz", "cyrl"),
    "tur": ("Turkish", "latn"),
    "mon": ("Mongolian", "cyrl"),
    "ind": ("Indonesian", "latn"),
    "san": ("Sanskrit", "deva"),
    "hin": ("Hindi", "deva"),
    "snd": ("Sindhi", "arab"),
    "hrv": ("Croatian", "latn"),
    "rus": ("Russian", "cyrl"),
    "fas": ("Persian", "arab"),
    "eng": ("English", "latn"),
    "swe": ("Swedish", "latn"),
    "heb": ("Hebrew", "hebr")
}

TOKENIZER_MAP = {
    "bpe-dropout": "BPE-dropout",
    "bpe": "BPE",
    "hnet": "H-Nets",
    "morphbpe": "MorphBPE",
    "morphbpe-dropout": "MorphBPE-dropout",
    "morphulm": "MorphULM",
    "morphwp": "MorphWP",
    "myte": "MYTE",
    "boundlessbpe": "BoundlessBPE",
    "superbpe": "SuperBPE",
    "morfessor": "Morfessor",
    "unigram": "Unigram",
    "wordpiece": "WordPiece",
    "pathpiece": "Pathpiece",
    "pickybpe": "PickyBPE",
    "sage": "SaGe",
    "sslm": "SSLM",
}

def format_tokenizer_name(name):
    return TOKENIZER_MAP.get(name.lower(), name)

def parse_step(step_str):
    parts = step_str.split('_')
    if len(parts) == 1:
        return (int(parts[0]), 0)
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    return (0, 0)

def main():
    parser = argparse.ArgumentParser(description="Generate a hierarchical CSV table from evaluation metrics")
    parser.add_argument("--metric", required=True, help="Metric to aggregate (e.g. fertility, compression)")
    parser.add_argument("--input_dir", default="evaluation_results", help="Base directory where evaluations are stored")
    parser.add_argument("--output_base_dir", default="evaluation_csv_results", help="Base directory to save output CSVs")
    args = parser.parse_args()

    # For morphscore metrics, they use the same base directory "morphscore"
    base_metric = "morphscore" if args.metric.startswith("morphscore_") else args.metric
    metric_dir = os.path.join(args.input_dir, base_metric)
    if not os.path.exists(metric_dir):
        print(f"Error: Directory {metric_dir} does not exist.")
        return

    # Create output directory
    output_metric_dir = os.path.join(args.output_base_dir, args.metric)
    os.makedirs(output_metric_dir, exist_ok=True)
    output_csv = os.path.join(output_metric_dir, f"{args.metric}_results.csv")

    data = {}
    all_columns = set()

    for lang in os.listdir(metric_dir):
        lang_dir = os.path.join(metric_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
            
        if lang not in data:
            data[lang] = {}
            
        sslm_files = []
        
        for file in os.listdir(lang_dir):
            if not file.endswith(".json"):
                continue
                
            file_path = os.path.join(lang_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
                    
                val = None
                if args.metric in jdata:
                    val = jdata[args.metric]
                elif args.metric == "compression" and "compression_rate" in jdata:
                    val = jdata["compression_rate"]
                elif args.metric == "vocab_size" and "effective_vocab_size" in jdata:
                    val = jdata["effective_vocab_size"]
                elif args.metric == "exponence" and "overall_average_exponence" in jdata:
                    val = jdata["overall_average_exponence"]
                elif args.metric in ["renyi_efficiency", "renyi_entropy", "shannon_efficiency", "shannon_entropy"] and args.metric in jdata:
                    val = jdata[args.metric]
                elif args.metric == "ttr" and "ttr" in jdata:
                    val = jdata["ttr"]
                elif args.metric == "morphscore_f1" and "f1" in jdata:
                    val = jdata["f1"]
                elif args.metric == "morphscore_precision" and "precision" in jdata:
                    val = jdata["precision"]
                elif args.metric == "morphscore_recall" and "recall" in jdata:
                    val = jdata["recall"]
                elif args.metric == "mean_token_length" or args.metric == "length_dist":
                    # For length_dist, we want to calculate the mean token length
                    # Format is {"length": frequency, ...} like {"1": 500, "2": 300}
                    total_tokens = sum(jdata.values())
                    if total_tokens > 0:
                        total_length = sum(int(length) * count for length, count in jdata.items())
                        val = total_length / total_tokens
                    else:
                        val = 0.0
                
                if val is None:
                    continue
                    
                name_without_ext = file.replace(".json", "")
                
                if name_without_ext.startswith("sslm_checkpoint"):
                    match = re.search(r'sslm_checkpoint_([\d_]+)_', name_without_ext)
                    if match:
                        step_str = match.group(1)
                        sslm_files.append((step_str, val))
                else:
                    data[lang][name_without_ext] = val
                    all_columns.add(name_without_ext)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                
        if sslm_files:
            highest_sslm = max(sslm_files, key=lambda x: parse_step(x[0]))
            col_name = "sslm_highest_step"
            data[lang][col_name] = highest_sslm[1]
            all_columns.add(col_name)

    def get_tok_config(col):
        if col.startswith('sslm_'):
            return 'sslm', col[5:]
        if col.startswith('hnet_'):
            parts = col.split('_', 1)
            conf = parts[1] if len(parts) > 1 else ""
            if conf.endswith('_test'):
                conf = conf[:-5]
            return 'hnet', conf
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return col, ""

    def sort_key(col_name):
        tok, conf = get_tok_config(col_name)
        fmt_tok = format_tokenizer_name(tok)
        
        if conf.isdigit():
            return (fmt_tok.lower(), 0, int(conf))
        return (fmt_tok.lower(), 1, conf)

    sorted_cols = sorted(list(all_columns), key=sort_key)
    
    header1 = ["Language", "lang code", "script"]
    header2 = ["", "", ""]
    
    last_tok = None
    for col in sorted_cols:
        tok, conf = get_tok_config(col)
        fmt_tok = format_tokenizer_name(tok)
        
        if fmt_tok != last_tok:
            header1.append(fmt_tok)
            last_tok = fmt_tok
        else:
            header1.append("")
            
        if conf.isdigit():
            header2.append(f"v{conf}")
        else:
            header2.append(conf)

    # Standard order of languages matching the screenshot provided
    standard_langs = ["fin", "hun", "mal", "tam", "tel", "kir", "tur", "mon", "ind", "san", "hin", "snd", "hrv", "rus", "fas", "eng", "swe", "heb"]
    all_langs = standard_langs + [l for l in data.keys() if l not in standard_langs]
    seen = set()
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)
        
        for lang in all_langs:
            if lang in seen or lang not in data:
                continue
            seen.add(lang)
            
            row = []
            info = LANG_INFO.get(lang, (lang, ""))
            row.extend([info[0], lang, info[1]])
            
            for col in sorted_cols:
                row.append(data[lang].get(col, ""))
                
            writer.writerow(row)
            
    print(f"CSV generated successfully at {output_csv}")

if __name__ == "__main__":
    main()
