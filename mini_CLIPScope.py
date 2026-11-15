import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Subset

from mini_common import DEVICE, load_clip, encode_texts, get_dataset, class_names

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)


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


def stream_scores(id_f, ood_f, text_f, n_id, seed):
    """Replays the original online loop. Returns dict of score arrays, labels."""
    x = torch.cat([id_f, ood_f], 0)
    y = np.concatenate([np.zeros(len(id_f)), np.ones(len(ood_f))])
    sim = 100 * (x @ text_f.T)
    probs_id = sim[:, :n_id].softmax(-1)
    p2 = sim.softmax(-1)[:, :n_id].sum(-1)
    p1, pred = probs_id.max(-1)

    counts = torch.ones(n_id)
    full = np.zeros(len(y))
    for i in np.random.RandomState(seed).permutation(len(y)):
        c = pred[i]
        p0 = counts[c] / counts.sum()
        full[i] = (p1[i] * p2[i] / p0).item()
        counts[c] += 1
    return {"p1 (MSP)": p1.numpy(), "p1*p2": (p1 * p2).numpy(), "p1*p2/p0": full}, y


def evaluate(scores, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=0)
    auroc = metrics.auc(fpr, tpr)
    idx = np.argmax(tpr > 0.95)
    return auroc, fpr[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="cifar100")
    ap.add_argument("--ood", nargs="+", default=["svhn", "eurosat", "fashionmnist"])
    ap.add_argument("--L", type=int, default=2000, help="max samples per ID / OOD set")
    ap.add_argument("--k", type=int, default=1000, help="nouns taken from each end")
    ap.add_argument("--model", default="ViT-B/16")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    path = ROOT / f"sorted_nouns_{args.id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run mini_mining.py --id {args.id} first")
    nouns = torch.load(path, weights_only=False)
    ood_labels = nouns[:args.k] + nouns[-args.k:]

    id_classes = class_names(get_dataset(args.id))
    model, preprocess = load_clip(args.model)
    text_f = build_text_features(model, id_classes, ood_labels)
    print(f"device={DEVICE}  ID classes={len(id_classes)}  OOD labels={len(ood_labels)}")

    id_f = image_features(model, preprocess, args.id, args.model, args.L)

    summary = {}
    for ood in args.ood:
        ood_f = image_features(model, preprocess, ood, args.model, args.L)
        res = {}
        for seed in args.seeds:
            sc, y = stream_scores(id_f, ood_f, text_f, len(id_classes), seed)
            for name, s in sc.items():
                res.setdefault(name, []).append(evaluate(s, y))
        print(f"\n== ID={args.id}  OOD={ood}  (ID n={len(id_f)}, OOD n={len(ood_f)}) ==")
        for name, vals in res.items():
            a = np.array(vals)
            print(f"{name:10s} AUROC {a[:, 0].mean()*100:6.2f} +- {a[:, 0].std()*100:4.2f}   "
                  f"FPR95 {a[:, 1].mean()*100:6.2f} +- {a[:, 1].std()*100:4.2f}")
            summary.setdefault(name, []).append(a.mean(0))

    print("\n== Average over OOD sets ==")
    for name, vals in summary.items():
        a = np.array(vals).mean(0)
        print(f"{name:10s} AUROC {a[0]*100:6.2f}   FPR95 {a[1]*100:6.2f}")


if __name__ == "__main__":
    main()