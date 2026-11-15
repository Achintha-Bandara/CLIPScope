"""Mini version of CLIPScope.py, following the paper's protocol.

Kept from the original run():
  * text bank = ID prompts + nouns[:M] + nouns[-M:]; first half of the OOD labels
    "a photo of a {c}", second half "a photo of {c}"; tau = 0.01 (x100)
  * histogram prior with counts initialised to 1 and updated after scoring
  * confidence = p1 * p2 / p0; AUROC and FPR at the first TPR > 0.95 (ID positive)
Changes (speed / reporting only):
  * images are encoded once (cached); the stream is replayed on features
  * p1, p2, p0 are logged, so all six g variants of the paper's Table 2 come from one pass
  * M defaults to 5 * N (paper: M = 5000 for N = 1000 classes)
  * stream orders random (mean of --trials, paper: 5), forward (ID first), reverse (OOD first)
  * --labels both | nearest | farthest reproduces the label-space ablation of Table 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Subset

from mini_common import DEVICE, load_clip, encode_texts, get_dataset, class_names

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
G_NAMES = ["p1", "p1/p0", "p2", "p2/p0", "p1*p2", "p1*p2/p0"]


@torch.no_grad()
def image_features(model, preprocess, name, model_name, L, data_seed=0, batch_size=128):
    cache = CACHE / f"{name}_{model_name.replace('/', '')}_{L}_{data_seed}.pt"
    if cache.exists():
        return torch.load(cache)
    ds = get_dataset(name)
    ds.transform = preprocess
    rng = np.random.RandomState(data_seed)
    idx = rng.permutation(len(ds))[:min(L, len(ds))].tolist()
    feats = []
    for n, (x, _) in enumerate(DataLoader(Subset(ds, idx), batch_size=batch_size)):
        f = model.encode_image(x.to(DEVICE)).float()
        f /= f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        print(f"  [{name}] {min((n + 1) * batch_size, len(idx))}/{len(idx)}", end="\r")
    print()
    feats = torch.cat(feats)
    torch.save(feats, cache)
    return feats


def build_text_features(model, id_classes, ood_labels):
    id_f = encode_texts(model, [f"a photo of a {c}" for c in id_classes])
    h = len(ood_labels) // 2
    o1 = encode_texts(model, [f"a photo of a {c}" for c in ood_labels[:h]])
    o2 = encode_texts(model, [f"a photo of {c}" for c in ood_labels[h:]])
    return torch.cat([id_f, o1, o2], 0)


def pick_labels(nouns, M, mode):
    if mode == "both":
        return nouns[:M] + nouns[-M:]
    return nouns[:M] if mode == "farthest" else nouns[-M:]


def make_order(kind, n_id, n_ood, rng):
    n = n_id + n_ood
    if kind == "random":
        return rng.permutation(n)
    if kind == "forward":                          # all ID first, then OOD
        return np.arange(n)
    return np.r_[np.arange(n_id, n), np.arange(n_id)]   # reverse: OOD first, then ID


def stream(id_f, ood_f, text_f, n_id, order):
    """Original run() loop on precomputed features; logs p1, p2, p0 per sample."""
    x = torch.cat([id_f, ood_f], 0)
    y = np.concatenate([np.zeros(len(id_f)), np.ones(len(ood_f))])
    sim = 100 * (x @ text_f.T)
    probs_id = sim[:, :n_id].softmax(dim=-1)
    p2 = sim.softmax(dim=-1)[:, :n_id].sum(dim=-1).numpy()
    count = torch.ones(n_id)
    p1, p0 = np.zeros(len(y)), np.zeros(len(y))
    for i in order:
        pred = probs_id[i].argmax()
        p1[i] = probs_id[i, pred].item()
        p0[i] = (count[pred] / count.sum()).item()
        count[pred] += 1                           # update after scoring
    return p1, p2, p0, y


def g_variants(p1, p2, p0):
    return {"p1": p1, "p1/p0": p1 / p0, "p2": p2, "p2/p0": p2 / p0,
            "p1*p2": p1 * p2, "p1*p2/p0": p1 * p2 / p0}


def evaluate(scores, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=0)
    auroc = metrics.auc(fpr, tpr)
    valid = np.flatnonzero(tpr > 0.95)
    fpr95 = fpr[valid[0]] if len(valid) else np.nan
    return auroc, fpr95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="cifar100")
    ap.add_argument("--ood", nargs="+", default=["svhn", "fashionmnist", "kmnist"])
    ap.add_argument("--L", type=int, default=2000, help="max samples per ID / OOD set")
    ap.add_argument("--M", type=int, default=None, help="labels per end (default 5 * N)")
    ap.add_argument("--labels", choices=["both", "nearest", "farthest"], default="both")
    ap.add_argument("--orders", nargs="+", default=["random"],
                    choices=["random", "forward", "reverse"])
    ap.add_argument("--trials", type=int, default=5, help="random-order trials (paper: 5)")
    ap.add_argument("--model", default="ViT-B/16")
    args = ap.parse_args()

    path = ROOT / f"sorted_nouns_{args.id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run mini_mining.py --id {args.id} first")
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("model") != args.model:
            raise SystemExit(f"Bank was mined with {meta.get('model')}, evaluating with {args.model}")
        print("Bank mined with:", meta)

    nouns = torch.load(path, weights_only=False)
    id_classes = class_names(args.id)
    n_id = len(id_classes)
    M = args.M or 5 * n_id
    ood_labels = pick_labels(nouns, M, args.labels)

    model, preprocess = load_clip(args.model)
    text_f = build_text_features(model, id_classes, ood_labels)
    print(f"device={DEVICE}  ID classes={n_id}  OOD labels={len(ood_labels)} "
          f"(M={M}, {args.labels})")

    id_f = image_features(model, preprocess, args.id, args.model, args.L)

    for kind in args.orders:
        trials = args.trials if kind == "random" else 1
        summary = {g: [] for g in G_NAMES}
        for ood in args.ood:
            ood_f = image_features(model, preprocess, ood, args.model, args.L)
            res = {g: [] for g in G_NAMES}
            for t in range(trials):
                order = make_order(kind, len(id_f), len(ood_f), np.random.RandomState(t))
                p1, p2, p0, y = stream(id_f, ood_f, text_f, n_id, order)
                for g, s in g_variants(p1, p2, p0).items():
                    res[g].append(evaluate(s, y))
            print(f"\n== {kind} order | ID={args.id} OOD={ood} "
                  f"(ID n={len(id_f)}, OOD n={len(ood_f)}, trials={trials}) ==")
            for g in G_NAMES:
                a = np.array(res[g]) * 100
                print(f"g = {g:9s} AUROC {a[:, 0].mean():6.2f}  FPR95 {a[:, 1].mean():6.2f}"
                      f"   (+- {a[:, 0].std():.2f} / {a[:, 1].std():.2f})")
                summary[g].append(a.mean(0))
        print(f"\n== {kind} order | average over OOD sets ==")
        for g in G_NAMES:
            a = np.array(summary[g]).mean(0)
            print(f"g = {g:9s} AUROC {a[0]:6.2f}  FPR95 {a[1]:6.2f}")


if __name__ == "__main__":
    main()