import nltk
import torch
from nltk.corpus import wordnet as wn

nltk.download("wordnet")

nouns = set()

for word in wn.all_lemma_names(pos=wn.NOUN):
    word = word.replace("_", " ").strip().lower()

    if word and any(ch.isalpha() for ch in word):
        nouns.add(word)

nouns = sorted(nouns)

torch.save(nouns, "nouns.pt")

print(f"Saved {len(nouns)} candidate nouns to nouns.pt")
print("First 20 candidates:")
print(nouns[:20])