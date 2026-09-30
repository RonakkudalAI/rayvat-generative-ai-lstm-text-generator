import os
import re
import ssl
import urllib.request

# Bypass SSL certificate verification issues on Anaconda Windows setups
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_dataset(data_dir="data", filename="shakespeare.txt"):
    """
    Downloads Shakespeare text dataset and cleans Gutenberg headers/footers if present.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading Shakespeare dataset from {SHAKESPEARE_URL}...")
        context = ssl._create_unverified_context()
        req = urllib.request.urlopen(SHAKESPEARE_URL, context=context)
        content = req.read().decode("utf-8")
        
        # Clean Gutenberg metadata headers/footers if present
        clean_content = clean_gutenberg_metadata(content)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_content)
        print(f"Cleaned dataset saved to {file_path}")
    else:
        print(f"Dataset already exists at {file_path}")
    return file_path


def clean_gutenberg_metadata(text):
    """
    Removes Project Gutenberg headers, footers, and license notices if present.
    """
    start_match = re.search(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .*\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .*\*\*\*", text, re.IGNORECASE)

    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]

    return text.strip()


def load_and_clean_text(file_path):
    """
    Preprocessing pipeline:
    1. Lowercase text
    2. Remove punctuation via regex
    3. Normalize whitespace (tabs/multiple spaces/newlines -> single space)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation (keep letters, numbers, and space)
    text = re.sub(r"[^\w\s]", "", text)

    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_vocabulary(tokens, max_vocab_size=15000):
    """
    Builds word -> ID and ID -> word dictionaries with special reserved tokens:
    <PAD> -> 0
    <UNK> -> 1
    
    Caps maximum vocabulary size to max_vocab_size. Rare words map to <UNK>.
    """
    # Count token frequencies
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    # Sort tokens by frequency (descending)
    sorted_words = sorted(freq.keys(), key=lambda w: freq[w], reverse=True)

    # Cap vocabulary size (excluding <PAD> and <UNK>)
    effective_max = max_vocab_size - 2
    vocabulary_words = sorted_words[:effective_max]

    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    idx_to_word = {0: "<PAD>", 1: "<UNK>"}

    for idx, word in enumerate(vocabulary_words, start=2):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, len(word_to_idx)


if __name__ == "__main__":
    path = download_dataset()
    cleaned = load_and_clean_text(path)
    tokens = cleaned.split()
    w2i, i2w, size = build_vocabulary(tokens, max_vocab_size=15000)
    print("Preprocessing & Vocabulary Summary:")
    print(f"Total Word Tokens: {len(tokens):,}")
    print(f"Vocabulary Size (with <PAD> & <UNK>): {size:,}")
    print(f"First 10 words in sample: {tokens[:10]}")
