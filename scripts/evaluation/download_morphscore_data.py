import argparse
import os
import requests
from pathlib import Path

# Mapping from ISO 639-3 to MorphScore filename prefix (Language Name)
# Based on Hugging Face dataset file naming convention (e.g., english_data.csv)
ISO_TO_FILENAME_PREFIX = {
    'eng': 'english',
    'spa': 'spanish',
    'deu': 'german',
    'fra': 'french',
    'rus': 'russian',
    'ita': 'italian',
    'por': 'portuguese',
    'pol': 'polish',
    'jpn': 'japanese',
    'vie': 'vietnamese',
    'tur': 'turkish',
    'nld': 'dutch',
    'ind': 'indonesian',
    'ara': 'arabic',
    'arb': 'arabic',
    'ces': 'czech',
    'fas': 'persian',
    'ell': 'greek',
    'zho': 'chinese', # might be mandarin or chinese?
    'cmn': 'chinese',
    'hin': 'hindi',
    'kor': 'korean',
    'tha': 'thai',
    'heb': 'hebrew',
    'ben': 'bengali',
    'tam': 'tamil',
    'kat': 'georgian',
    'mar': 'marathi',
    'tel': 'telugu',
    'nob': 'norwegian',
    'swe': 'swedish',
    'ron': 'romanian',
    'ukr': 'ukrainian',
    'hun': 'hungarian',
    'dan': 'danish',
    'fin': 'finnish',
    'bul': 'bulgarian',
    'slk': 'slovak',
    'cat': 'catalan',
    'urd': 'urdu',
    'bel': 'belarusian',
    'tgk': 'tajik',
    'guj': 'gujarati',
    'kan': 'kannada',
    'mal': 'malayalam',
    'pan': 'punjabi',
    'est': 'estonian',
    'lit': 'lithuanian',
    'lav': 'latvian',
    'slv': 'slovenian',
    'hrv': 'croatian',
    'srp': 'serbian',
    'mkd': 'macedonian',
    'sqi': 'albanian',
    'hye': 'armenian',
    # Add more as needed
}

HF_REPO_URL = "https://huggingface.co/datasets/catherinearnett/morphscore/resolve/main"

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"File not found: {url}")
        else:
            print(f"HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download MorphScore datasets.")
    parser.add_argument("--langs", nargs="+", required=True, help="List of language codes to download (e.g., 'eng', 'spa').")
    parser.add_argument("--output_dir", default="morphscore_data", help="Directory to save datasets.")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in args.langs:
        # Resolve to MorphScore filename prefix (Language Name)
        filename_prefix = ISO_TO_FILENAME_PREFIX.get(lang)
        
        if not filename_prefix:
            print(f"Warning: No mapping found for language '{lang}'. Trying to use '{lang}' as prefix.")
            filename_prefix = lang
        
        # Construct filename
        filename = f"{filename_prefix}_data.csv"
        url = f"{HF_REPO_URL}/{filename}"
        output_path = output_dir / filename
        
        success = download_file(url, output_path)
        
        if not success:
            print(f"Failed to download data for {lang} ({filename}).")
            # Try alternate extension if csv fails, though csv seems to be the standard based on findings
            filename_parquet = f"{filename_prefix}_data.parquet"
            url_parquet = f"{HF_REPO_URL}/{filename_parquet}"
            output_path_parquet = output_dir / filename_parquet
            
            print(f"Trying parquet: {filename_parquet}")
            download_file(url_parquet, output_path_parquet)

if __name__ == "__main__":
    main()
