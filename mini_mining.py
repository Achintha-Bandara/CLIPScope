import argparse
import math
from pathlib import Path

import numpy as np
import torch

from mini_common import load_clip, encode_texts, get_dataset, class_names

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
                # single lowercase words only (drops proper nouns / multiword)
                if lemma.isalpha() and lemma.islower():
                    nouns.add(lemma)
        nouns = sorted(nouns)
        torch.save(nouns, path)
        print(f"Built {len(nouns)} WordNet nouns -> {path}")
    return [str(n).replace("_", " ") for n in nouns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="cifar100")
    ap.add_argument("--model", default="ViT-B/16")
    ap.add_argument("--percentile", type=float, default=0.05)
    ap.add_argument("--max-nouns", type=int, default=None,
                    help="random subsample of the noun list (speed on CPU)")
    ap.add_argument("--keep-id-names", action="store_true",
                    help="do not remove nouns that equal an ID class name")
    args = ap.parse_args()

    id_classes = class_names(get_dataset(args.id))
    nouns = load_nouns()
    if not args.keep_id_names:
        id_set = set(id_classes)
        nouns = [n for n in nouns if n not in id_set]
    if args.max_nouns and args.max_nouns < len(nouns):
        rng = np.random.RandomState(0)
        nouns = [nouns[i] for i in rng.choice(len(nouns), args.max_nouns, replace=False)]

    model, _ = load_clip(args.model)
    id_feat = encode_texts(model, [f"a photo of a {c}" for c in id_classes])

    k = max(1, math.ceil(len(id_classes) * args.percentile))
    print(f"ID classes: {len(id_classes)}  ->  k = {k} ({args.percentile:.0%})")

    noun_feat = encode_texts(model, ["the " + w for w in nouns])   # same prompt as original
    sim = noun_feat @ id_feat.T                                   # N x C
    scores = torch.topk(sim, k, dim=1).values[:, -1].numpy()      # k-th largest similarity

    order = np.argsort(scores)                                    # ascending
    sorted_nouns = [nouns[i] for i in order]
    out = ROOT / f"sorted_nouns_{args.id}.pt"
    torch.save(sorted_nouns, out)

    print("Total nouns:", len(sorted_nouns))
    print("Farthest from ID:", sorted_nouns[:10])
    print("Nearest to ID:", sorted_nouns[-10:])
    print("Saved", out)


if __name__ == "__main__":
    main()