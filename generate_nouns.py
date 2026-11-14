from pathlib import Path

import nltk
import torch
from nltk.corpus import wordnet as wn
from nltk.downloader import Downloader


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "nouns.pt"


def wordnet_is_available():
    return bool(Downloader().is_installed("wordnet"))


def load_wordnet_nouns():
    if not wordnet_is_available():
        try:
            nltk.download("wordnet", quiet=False)
        except Exception as error:
            raise RuntimeError(
                "WordNet is unavailable. Install it with "
                "`python -m nltk.downloader wordnet` and retry."
            ) from error

    try:
        return {
            lemma.replace("_", " ").strip().lower()
            for lemma in wn.all_lemma_names(pos=wn.NOUN)
            if lemma and any(character.isalpha() for character in lemma)
        }
    except LookupError as error:
        raise RuntimeError(
            "WordNet is unavailable. Install it with "
            "`python -m nltk.downloader wordnet` and retry."
        ) from error


def main():
    nouns = sorted(load_wordnet_nouns())
    if not nouns:
        raise RuntimeError("WordNet returned no noun candidates.")

    torch.save(nouns, OUTPUT)
    print(f"Saved {len(nouns)} candidate nouns to {OUTPUT}")
    print("First 20 candidates:")
    print(nouns[:20])


if __name__ == "__main__":
    main()