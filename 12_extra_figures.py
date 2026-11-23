"""12_extra_figures.py - additional result figures for the report and presentation.

  12_training_curves.png   training loss and validation macro-F1 per epoch, every teacher-student round
                           (ITS+ and ITS, seed 0), parsed from the training logs
  12_roc_curves.png        ROC curves on the test set: one-vs-rest per class for ITS+, macro average per model
  12_per_class.png         precision / recall / F1 per class (mean over seeds) for supervised, ITS, ITS+
  12_clusters.png          unsupervised reference: PCA projection of ImageNet features coloured by true class
                           and by k-means cluster, plus the PCA variance retained
  12_method_diagram.png    diagram of the hierarchical teacher-student method

Usage: python 12_extra_figures.py
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve

from common import load_config
from ssl_core import DEVICE, Data, load_model, normalise, predict, pretrained_backbone, run_dir, run_name

COLORS = {"supervised": "#8a8f98", "its": "#3b78c2", "its_plus": "#1b7f55"}
NAMES = {"supervised": "Supervised", "its": "ITS (Paper 1)", "its_plus": "ITS+ (ours)"}
CLASS_COLORS = ["#4c9f70", "#e0a030", "#c8453a"]
EPOCH = re.compile(r"epoch\s+(\d+)\s+loss ([\d.]+)\s+val acc ([\d.]+)\s+val macro-F1 ([\d.]+)")


def parse_log(path):
    """Split a training log into segments (round 0, round 1 [tau], ..., fine-tune) of per-epoch values."""
    segs, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("round 0 teacher: training"):
            cur = {"name": "round 0 (teacher)", "rows": []}
            segs.append(cur)
        elif re.match(r"round \d+\s+threshold", s):
            r = s.split()[1]
            tau = s.split("threshold")[1].split(":")[0].strip()
            cur = {"name": f"round {r}" + ("" if "class-adaptive" in tau else f" (tau {tau})"), "rows": []}
            segs.append(cur)
        elif s.startswith("final fine-tune"):
            cur = {"name": "fine-tune", "rows": []}
            segs.append(cur)
        else:
            m = EPOCH.search(line)
            if m and cur is not None:
                cur["rows"].append([float(v) for v in m.groups()])
    return [s for s in segs if s["rows"]]


def training_curves(cfg, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    cmap = plt.get_cmap("viridis")
    for col, method in enumerate(["its_plus", "its"]):
        segs = parse_log(run_dir(cfg, run_name(method, 0.10, 0)) / "log.txt")
        for k, seg in enumerate(segs):
            r = np.array(seg["rows"])
            c = cmap(k / max(len(segs) - 1, 1))
            axes[0, col].plot(r[:, 0], r[:, 1], marker=".", color=c, label=seg["name"])
            axes[1, col].plot(r[:, 0], r[:, 3] * 100, marker=".", color=c, label=seg["name"])
            best = r[:, 3].argmax()
            axes[1, col].scatter(r[best, 0], r[best, 3] * 100, s=60, facecolor="none", edgecolor=c, linewidth=1.5)
        axes[0, col].set_title(f"{NAMES[method]}, seed 0: training loss", fontsize=10)
        axes[1, col].set_title(f"{NAMES[method]}, seed 0: validation macro-F1 (%) (circle = best epoch)",
                               fontsize=10)
        for ax in axes[:, col]:
            ax.set_xlabel("epoch within round")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "12_training_curves.png", dpi=130)
    plt.close(fig)


def roc_curves(cfg, data, out):
    y = data.y[data.test]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    grid = np.linspace(0, 1, 501)
    for method in ["supervised", "its", "its_plus"]:
        model, _ = load_model(cfg, run_dir(cfg, run_name(method, 0.10, 0)) / "best.pt")
        p = predict(data, model, data.test)
        del model
        tprs, aucs = [], []
        for c in range(3):
            fpr, tpr, _ = roc_curve(y == c, p[:, c])
            aucs.append(auc(fpr, tpr))
            tprs.append(np.interp(grid, fpr, tpr))
            if method == "its_plus":
                axes[0].plot(fpr, tpr, color=CLASS_COLORS[c], label=f"{data.classes[c]} (AUC {aucs[-1]:.3f})")
        axes[1].plot(grid, np.mean(tprs, 0), color=COLORS[method],
                     label=f"{NAMES[method]} (macro AUC {np.mean(aucs):.3f})")
    for ax, t in zip(axes, ["ITS+ (ours), seed 0: one-vs-rest ROC per class", "Macro-average ROC, seed 0"]):
        ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "12_roc_curves.png", dpi=130)
    plt.close(fig)


def per_class(cfg, out):
    classes = cfg["classes"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    width = 0.26
    for k, (metric, title) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]):
        for j, method in enumerate(["supervised", "its", "its_plus"]):
            vals = []
            for s in (0, 1, 2):
                m = json.loads((out / "metrics" / f"{run_name(method, 0.10, s)}.json").read_text())
                p, r = np.array(m["precision"]), np.array(m["recall"])
                vals.append({"precision": p, "recall": r, "f1": 2 * p * r / np.maximum(p + r, 1e-9)}[metric])
            v = np.mean(vals, 0) * 100
            bars = axes[k].bar(np.arange(3) + (j - 1) * width, v, width, color=COLORS[method], label=NAMES[method])
            for b, x in zip(bars, v):
                axes[k].text(b.get_x() + b.get_width() / 2, x + 1, f"{x:.0f}", ha="center", fontsize=7)
        axes[k].set_xticks(range(3))
        axes[k].set_xticklabels(classes)
        axes[k].set_title(f"{title} per class (%), mean of 3 seeds", fontsize=10)
        axes[k].set_ylim(0, 105)
        axes[k].spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "12_per_class.png", dpi=130)
    plt.close(fig)


@torch.no_grad()
def features(data, net, idx):
    out = []
    for i in range(0, len(idx), 256):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            f = net(normalise(data.batch(idx[i:i + 256])))
        out.append(F.adaptive_avg_pool2d(f, 1).flatten(1).float().cpu())
    return torch.cat(out).numpy()


def clusters(cfg, data, out):
    net = pretrained_backbone(cfg).to(DEVICE).eval()
    train = np.concatenate([data.lab, data.pool])
    f_tr, f_te = features(data, net, train), features(data, net, data.test)
    pca = PCA(50, random_state=0).fit(f_tr)
    km = KMeans(3, n_init=10, random_state=0).fit(pca.transform(f_tr))
    z = pca.transform(f_te)
    cl = km.predict(z)
    full = PCA(random_state=0).fit(f_tr)
    cum = np.cumsum(full.explained_variance_ratio_) * 100
    y = data.y[data.test]
    rng = np.random.default_rng(0)
    show = rng.choice(len(y), 2000, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for c in range(3):
        s = show[y[show] == c]
        axes[0].scatter(z[s, 0], z[s, 1], s=5, alpha=0.6, color=CLASS_COLORS[c], label=data.classes[c])
    for k in range(3):
        s = show[cl[show] == k]
        axes[1].scatter(z[s, 0], z[s, 1], s=5, alpha=0.6, color=["#6a4c93", "#1982c4", "#8ac926"][k],
                        label=f"cluster {k}")
    axes[0].set_title("Test scans, PCA of ImageNet features: true class", fontsize=10)
    axes[1].set_title("Same projection: k-means cluster (no labels)", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8, markerscale=3)
    axes[2].plot(np.arange(1, len(cum) + 1), cum, color="#333333")
    axes[2].axvline(50, color="#c8453a", linestyle="--", linewidth=1)
    axes[2].text(55, cum[49] - 8, f"50 components keep {cum[49]:.1f}% of the variance", fontsize=8,
                 color="#c8453a")
    axes[2].set_xlim(0, 300)
    axes[2].set_xlabel("number of principal components")
    axes[2].set_ylabel("cumulative variance retained (%)")
    axes[2].set_title("PCA variance retention", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "12_clusters.png", dpi=130)
    plt.close(fig)
    print(f"PCA: 50 components retain {cum[49]:.1f}% of the feature variance")


def method_diagram(out):
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, text="", color="#333333", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4,
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, text, ha="center", fontsize=8, color=color)

    box(0.1, 3.6, 2.1, 1.2, "Labelled scans\n10% of training cases\n(940 B-scans)", "#dfeee5")
    box(0.1, 0.5, 2.1, 1.2, "Unlabelled pool\n90% of training cases\n(8,933 B-scans)", "#eeeeee")
    box(2.9, 3.6, 2.4, 1.2, "TEACHER\nhierarchical EfficientNet-B0\n(round 0: labelled only)", "#d6e6f5", bold=False)
    box(2.9, 0.5, 2.4, 1.2, "Teacher predicts\nP(NORMAL | x), P(DRUSEN | x),\nP(CNV | x) for every\nunlabelled scan x", "#f4f4f4")
    box(6.0, 0.25, 2.6, 1.7, "Pseudo-label admission\n1. class-adaptive thresholds\n2. full / partial / reject\n"
        "3. agreement with neighbouring\n    B-scans (window of 5)", "#fbe9d0")
    box(6.0, 3.4, 2.6, 1.6, "STUDENT training\nbatch = 20 labelled (weak aug.)\n+ 44 pseudo-labelled (strong aug.)\n"
        "loss = L_lab + λ(t) · L_pseudo", "#d6e6f5")
    box(9.3, 3.6, 1.7, 1.2, "Best epoch\n(validation\nmacro-F1)", "#eeeeee")
    box(11.3, 3.6, 1.6, 1.2, "Best round\n(+ fine-tune\nif better)\n→ checkpoint", "#dfeee5", bold=False)
    box(11.3, 0.5, 1.6, 1.2, "Test set\n109 unseen\npatients", "#f4dede")
    arrow(2.2, 4.2, 2.9, 4.2, "train")
    arrow(2.2, 1.1, 2.9, 1.1)
    arrow(4.1, 3.6, 4.1, 1.7, "")
    arrow(5.3, 1.1, 6.0, 1.1)
    arrow(7.3, 1.95, 7.3, 3.4, "pseudo-labels")
    arrow(1.15, 4.8, 7.3, 5.0, "", color="#1b7f55", rad=-0.18)
    ax.text(4.2, 5.55, "labelled scans also go into every student batch", ha="center", fontsize=8,
            color="#1b7f55")
    arrow(8.6, 4.2, 9.3, 4.2)
    arrow(11.0, 4.2, 11.3, 4.2)
    arrow(12.1, 3.6, 12.1, 1.7, "benchmark")
    arrow(6.0, 3.75, 5.3, 3.75, "", color="#b5532a")
    ax.text(5.65, 2.85, "student becomes\nnext teacher\n(4 rounds)", ha="center", fontsize=8,
            color="#b5532a")
    ax.text(6.5, 0.02, "Hierarchical heads:  P(NORMAL | x) = 1 − P(abnormal | x);   "
            "P(DRUSEN | x) = P(abnormal | x) P(DRUSEN | abnormal, x);   "
            "P(CNV | x) = P(abnormal | x) P(CNV | abnormal, x)", ha="center", fontsize=8.5, style="italic",
            color="#333333")
    fig.tight_layout()
    fig.savefig(out / "12_method_diagram.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    out = cfg["output_dir"]
    data = Data(cfg, 0, 0.10)
    method_diagram(out)
    training_curves(cfg, out)
    per_class(cfg, out)
    roc_curves(cfg, data, out)
    clusters(cfg, data, out)
    print("saved outputs/12_training_curves.png, 12_roc_curves.png, 12_per_class.png, 12_clusters.png, "
          "12_method_diagram.png")


if __name__ == "__main__":
    main()
