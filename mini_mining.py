"""Mini version of mining.py (same procedure, only the ID label set changes).

Original: 50th-largest similarity to the 1000 ImageNet classes (eta = 5%),
prompt "the <noun>", sort ascending (first = farthest, last = nearest).
Mini:     percentile = ceil(eta * N) for N ID classes (N=100 -> 5).
Changes vs original: noun encoding is batched (same maths); ID class names are
NOT removed from the noun list (like the original; --drop-id-names to remove them);
the mining prompt is configurable (paper's Alg. 1 says "a photo of a {}", the
released code uses "the {}"; default here = released code).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mini_common import load_clip, encode_texts, class_names

ROOT = Path(__file__).resolve().parent


def load_nouns():
    path = ROOT / "nouns.pt"
    if path.exists():
        nouns = torch.load(path, weights_only=False)
        print(f"Loaded {len(nouns)} nouns from {path}")
    else:
        import nltk
        from nltk.corpus import wordnet as wn
        nltk.download("wordnet", quiet=True)
        nouns = set()
        for syn in wn.all_synsets("n"):
            for lemma in syn.lemma_names():
                if lemma.isalpha() and lemma.islower():   # single lowercase words only
                    nouns.add(lemma)
        nouns = sorted(nouns)
        torch.save(nouns, path)
        print(f"Built {len(nouns)} WordNet nouns -> {path}")
    return [str(n).replace("_", " ") for n in nouns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="cifar100")
    ap.add_argument("--model", default="ViT-B/16")
    ap.add_argument("--eta", type=float, default=0.05, help="percentile (paper: 0.05)")
    ap.add_argument("--template", default="the {}",
                    help='mining prompt; released code "the {}", paper Alg.1 "a photo of a {}"')
    ap.add_argument("--drop-id-names", action="store_true")
    args = ap.parse_args()

    id_classes = class_names(args.id)
    nouns = np.array(load_nouns())
    if args.drop_id_names:
        nouns = np.array([n for n in nouns if n not in set(id_classes)])

    model, preprocess = load_clip(args.model)
    text_features = encode_texts(model, [f"a photo of a {c}" for c in id_classes])

    percentile = max(1, math.ceil(len(id_classes) * args.eta))   # original: 50 of 1000
    print(f"ID classes: {len(id_classes)}  ->  percentile index = {percentile}")

    scores = []
    for s in range(0, len(nouns), 1024):
        sample_features = encode_texts(model, [args.template.format(w) for w in nouns[s:s + 1024]])
        similarity = -(sample_features @ text_features.T)
        value, index = torch.kthvalue(similarity, percentile, dim=1)
        scores.append(-value.numpy())
        print(f"  {min(s + 1024, len(nouns))}/{len(nouns)}", end="\r")
    scores = np.concatenate(scores).astype(np.float32)
    sorted_indices = np.argsort(scores)

    sorted_nouns = [str(nouns[i]) for i in sorted_indices]
    torch.save(sorted_nouns, ROOT / f"sorted_nouns_{args.id}.pt")
    (ROOT / f"sorted_nouns_{args.id}.json").write_text(json.dumps(vars(args)))

    print("\nTotal nouns:", len(sorted_nouns))
    print("Lowest-ranked nouns:", sorted_nouns[:10])
    print("Highest-ranked nouns:", sorted_nouns[-10:])


if __name__ == "__main__":
    main()