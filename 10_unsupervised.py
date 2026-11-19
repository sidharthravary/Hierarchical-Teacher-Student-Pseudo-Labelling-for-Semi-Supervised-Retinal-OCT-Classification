"""10_unsupervised.py - unsupervised reference: clustering without any training labels.

ImageNet EfficientNet-B0 features (no OCT labels involved) -> PCA (50 dims) -> k-means with k = 3,
fitted on all training scans with their labels ignored. Labels are used only afterwards, on the
validation set, to name each cluster (Hungarian matching); the test set is then scored like every
other model. Repeated for k-means seeds 0, 1, 2.

Usage: python 10_unsupervised.py
Output: outputs/unsupervised_results.csv, outputs/metrics/unsupervised_kmeans_s<seed>.json,
        outputs/10_unsupervised.txt
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import load_config
from ssl_core import DEVICE, Data, Log, metrics, normalise, pretrained_backbone

SEEDS = [0, 1, 2]


@torch.no_grad()
def features(data, net, idx, batch=256):
    out = []
    for i in range(0, len(idx), batch):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            f = net(normalise(data.batch(idx[i:i + batch])))
        out.append(F.adaptive_avg_pool2d(f, 1).flatten(1).float().cpu())
    return torch.cat(out).numpy()


def main():
    cfg = load_config()
    out = cfg["output_dir"]
    log = Log(out / "10_unsupervised.txt")
    data = Data(cfg, 0, 0.10)
    train = np.concatenate([data.lab, data.pool])
    net = pretrained_backbone(cfg).to(DEVICE).eval()
    log(f"extracting ImageNet EfficientNet-B0 features: train {len(train)}, val {len(data.val)}, "
        f"test {len(data.test)} (no OCT training, no labels)")
    f_tr, f_va, f_te = (features(data, net, i) for i in (train, data.val, data.test))

    rows = []
    for seed in SEEDS:
        pca = PCA(50, random_state=seed).fit(f_tr)
        km = KMeans(3, n_init=10, random_state=seed).fit(pca.transform(f_tr))
        # name the clusters with the validation labels (Hungarian matching on the contingency table)
        cv = km.predict(pca.transform(f_va))
        table = np.array([[np.sum((cv == k) & (data.y[data.val] == c)) for c in range(3)] for k in range(3)])
        k_idx, c_idx = linear_sum_assignment(-table)
        name = dict(zip(k_idx, c_idx))
        # soft class scores from distances to the cluster centres (for AUC)
        d = km.transform(pca.transform(f_te))
        logits = -d / d.std()
        p_cluster = np.exp(logits - logits.max(1, keepdims=True))
        p_cluster /= p_cluster.sum(1, keepdims=True)
        probs = np.zeros_like(p_cluster)
        for k, c in name.items():
            probs[:, c] = p_cluster[:, k]
        m = metrics(data.y[data.test], probs)
        (out / "metrics" / f"unsupervised_kmeans_s{seed}.json").write_text(json.dumps(m, indent=1))
        log(f"seed {seed}: cluster -> class {[data.classes[name[k]] for k in range(3)]}, "
            f"test accuracy {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}, AUC {m['macro_auc']:.4f}, "
            f"under-grading {m['under_grading_rate']:.4f}")
        rows.append({"seed": seed, **{k: m[k] for k in ("accuracy", "macro_f1", "macro_auc", "under_grading_rate")},
                     **{f"recall_{c}": m["recall"][i] for i, c in enumerate(data.classes)}})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "unsupervised_results.csv", index=False)
    log("\nmean +/- SD over k-means seeds: " + ", ".join(
        f"{c} {df[c].mean() * 100:.2f} +/- {df[c].std() * 100:.2f}"
        for c in ("accuracy", "macro_f1", "macro_auc", "under_grading_rate")))


if __name__ == "__main__":
    main()
