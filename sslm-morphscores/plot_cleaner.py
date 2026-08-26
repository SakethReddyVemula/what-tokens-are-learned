import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_morphscore_by_steps(lang, csv_path, output_path):
    # Load the data
    df = pd.read_csv(csv_path)

    # Parsing function to extract epoch and step safely
    def parse_checkpoint(name):
        # Specifically matches 'checkpoint_{epoch}_{step}.pt'
        match = re.search(r'checkpoint_(\d+)_(\d+)\.pt', str(name))
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    # Apply parsing
    df[['epoch', 'step']] = df['checkpoint_name'].apply(lambda x: pd.Series(parse_checkpoint(x)))

    # Filter out inconsistent names and sort numerically
    df_clean = df.dropna(subset=['epoch', 'step']).copy()
    df_clean['step'] = df_clean['step'].astype(int)
    df_clean['epoch'] = df_clean['epoch'].astype(int)
    df_clean = df_clean.sort_values(['epoch', 'step'])

    # Prepare labels (only the step number)
    df_clean['label'] = df_clean['step'].astype(str)

    # Setup Plot
    plt.figure(figsize=(8, 12)) # Initial size
    
    # Plotting lines
    plt.plot(df_clean['label'], df_clean['precision'], marker='o', label='Precision', markersize=2)
    plt.plot(df_clean['label'], df_clean['recall'], marker='s', label='Recall', markersize=2)
    plt.plot(df_clean['label'], df_clean['f1'], marker='^', label='F1 Score', markersize=2)

    # 1. Y-axis fixed 0 to 1 scale
    plt.ylim(0, 1.0) 
    
    # 2. Square shape geometry
    plt.gca().set_box_aspect(1)

    # Labels and Titles
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title("Morphological Alignment - " + lang, fontsize=14, pad=15)
    
    # Handle X-axis tick frequency to keep it readable
    n = len(df_clean)
    if n > 20:
        indices = list(range(0, n, max(1, n // 10)))
        if (n-1) not in indices: indices.append(n-1)
        plt.xticks(indices, df_clean['label'].iloc[indices], rotation=45)
    else:
        plt.xticks(rotation=45)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the result
    plt.savefig(output_path, format='pdf', bbox_inches='tight')

    print(f"Graph successfully saved as {output_path}")

if __name__ == "__main__":
    LANG_CODE = "heb_hebr"
    LANG = "Hebrew (SSLM)"
    input_path = LANG_CODE + ".csv"
    output_path = LANG_CODE + ".pdf"
    plot_morphscore_by_steps(LANG, input_path, output_path)