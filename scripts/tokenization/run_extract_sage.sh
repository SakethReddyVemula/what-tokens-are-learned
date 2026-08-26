#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Define the languages
# LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
LANGS="snd"

echo "Extracting and converting SaGe vocabularies for languages: $LANGS"

# Run the python script
python3 "$SCRIPT_DIR/extract_and_covert_final_sage_vocab.py" --langs $LANGS

echo "Done!"
