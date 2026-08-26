from huggingface_hub import HfApi

api = HfApi()

files = [
    "fin/bpe_v5000.csv",
    "hun/bpe_v5000.csv",
    "mal/bpe_v5000.csv",
    "tam/bpe_v5000.csv",
    "tel/bpe_v5000.csv",
    "kir/bpe_v5000.csv",
    "tur/bpe_v5000.csv",
    # "mon/bpe_v5000.csv",
    "ind/bpe_v5000.csv",
    # "san/bpe_v5000.csv",
    "hin/bpe_v5000.csv",
    "snd/bpe_v5000.csv",
    "hrv/bpe_v5000.csv",
    "rus/bpe_v5000.csv",
    "fas/bpe_v5000.csv",
    "eng/bpe_v5000.csv",
    "swe/bpe_v5000.csv",
    "heb/bpe_v5000.csv",
    # "eng/sage_v10000_test.jsonl",
    # "snd/sage_v10000_test.jsonl"
]

for f in files:
    api.delete_file(
        path_in_repo=f,
        repo_id="SakethVemula/fixed-tokenizer-morphscore-segments",
        repo_type="dataset"
    )