"""08_report.py - tables and figures from the evaluation.

Reads outputs/results.csv, outputs/metrics/*.json and outputs/runs/*/rounds.csv and writes
RESULTS.md plus three figures in outputs/:
  08_comparison.png  accuracy / macro-F1 / under-grading rate per method
  08_confusions.png  confusion matrices of supervised, ITS and ITS+ (seed 0)
  08_rounds.png      validation macro-F1 and pseudo-label quality per teacher-student round

Usage: python 08_report.py [--fraction 0.10]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, load_config
from ssl_core import run_name

MAIN = [("supervised", "Supervised"), ("its", "ITS (Paper 1)"), ("its_plus", "ITS+ (ours)")]
COLORS = {"supervised": "#8a8f98", "supervised_hier": "#b5b9c0", "its": "#3b78c2", "its_plus_A": "#9ad0b3",
          "its_plus_B": "#6dbb93", "its_plus_C": "#43a274", "its_plus": "#1b7f55"}
PUBLISHED = """| Labels | Supervised (Paper 1) | ITS (Paper 1) | Supervised (Paper 2) | TRIAGE (Paper 2) |
|---|---|---|---|---|
| 5% | 54.78 | 64.15 | 60.28 | 76.88 |
| 20% | 77.05 | 88.56 | 79.25 | 89.66 |
| 100% | 87.14 | - | 88.50 | - |"""


def summary(df):
    """Mean and SD over seeds per (fraction, method)."""
    g = df.groupby(["fraction", "method", "label"], sort=False)
    s = g[["accuracy", "macro_f1", "macro_auc", "under_grading_rate"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["seeds"] = g.size()
    return s.reset_index()


def cell(mean, sd):
    return f"{mean * 100:.2f}" + (f" +/- {sd * 100:.2f}" if pd.notna(sd) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    args = ap.parse_args()
    cfg = load_config()
    out = cfg["output_dir"]
    df = pd.read_csv(out / "results.csv")
    summ = summary(df)
    classes = cfg["classes"]
    f = args.fraction

    # ---- figure 1: comparison at the main label fraction
    cur = summ[np.isclose(summ["fraction"], f)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [("accuracy", "Accuracy (%)"), ("macro_f1", "Macro-F1 (%)"),
              ("under_grading_rate", "Under-grading rate (%), lower is better")]
    for ax, (col, title) in zip(axes, panels):
        vals = (cur[f"{col}_mean"] * 100).to_numpy()
        err = (cur[f"{col}_std"].fillna(0) * 100).to_numpy()
        bars = ax.bar(range(len(cur)), vals, yerr=err, capsize=3,
                      color=[COLORS.get(m, "#888888") for m in cur["method"]])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(cur)))
        ax.set_xticklabels(cur["label"], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if col != "under_grading_rate":
            ax.set_ylim(max(0, vals.min() - 15), min(100, vals.max() + 6))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Test set, {f:.0%} of training labels (bars = mean over seeds, whiskers = SD)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "08_comparison.png", dpi=120)
    plt.close(fig)

    # ---- figure 2: confusion matrices (seed 0)
    have = [(m, l) for m, l in MAIN if (out / "metrics" / f"{run_name(m, f, 0)}.json").exists()]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(4.3 * len(have), 4), squeeze=False)
        for ax, (m, l) in zip(axes[0], have):
            res = json.loads((out / "metrics" / f"{run_name(m, f, 0)}.json").read_text())
            cm = np.array(res["confusion"])
            share = cm / cm.sum(1, keepdims=True)
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                            color="white" if share[i, j] > 0.55 else "black")
            ax.set_xticks(range(3))
            ax.set_xticklabels(classes, fontsize=8)
            ax.set_yticks(range(3))
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{l}\naccuracy {res['accuracy']:.1%}, macro-F1 {res['macro_f1']:.1%}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "08_confusions.png", dpi=120)
        plt.close(fig)

    # ---- figure 3: rounds (seed 0)
    rounds = {}
    for m, l in MAIN[1:]:
        p = out / "runs" / run_name(m, f, 0) / "rounds.csv"
        if p.exists():
            r = pd.read_csv(p)
            r["round"] = r["round"].astype(str)
            r = r[r["round"].str.isdigit() & ((r["round"] == "0") | (r["kept"].astype(str) == "True"))]
            rounds[m] = (l, r)
    if rounds:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for m, (l, r) in rounds.items():
            x = r["round"].astype(int).to_numpy()
            axes[0].plot(x, r["val_macro_f1"] * 100, marker="o", color=COLORS[m], label=l)
            axes[1].plot(x[1:], r["full_precision"].to_numpy()[1:] * 100, marker="o", color=COLORS[m], label=l)
            axes[2].plot(x[1:], r["coverage"].to_numpy()[1:] * 100, marker="o", color=COLORS[m], label=l)
        titles = ["Validation macro-F1 (%) per round (0 = supervised teacher)",
                  "Pseudo-label precision (%) per round", "Share of unlabelled scans admitted (%)"]
        for ax, t in zip(axes, titles):
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("round")
            ax.set_xticks(range(0, 5))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "08_rounds.png", dpi=120)
        plt.close(fig)

    # ---- RESULTS.md
    md = ["# Results", "",
          "Dataset: NEH_UT_2021 (Noor Eye Hospital), 3 classes. Test set: 4,326 scans from 109 cases that no "
          "model saw during training or model selection. All numbers are scan-level, in %.", "",
          "Generated by `08_report.py` from `outputs/results.csv`.", ""]
    table_lines = []
    for frac in sorted(summ["fraction"].unique()):
        s = summ[np.isclose(summ["fraction"], frac)]
        table_lines += [f"## {frac:.0%} of training labels", "",
                        "| Method | Seeds | Accuracy | Macro-F1 | Macro-AUC | Under-grading rate |",
                        "|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            table_lines.append(
                f"| {r['label']} | {r['seeds']} | {cell(r['accuracy_mean'], r['accuracy_std'])} | "
                f"{cell(r['macro_f1_mean'], r['macro_f1_std'])} | {cell(r['macro_auc_mean'], r['macro_auc_std'])} | "
                f"{cell(r['under_grading_rate_mean'], r['under_grading_rate_std'])} |")
        table_lines.append("")
    md += table_lines
    md += ["Values are mean +/- SD over seeds where more than one seed was run. Under-grading rate = share of "
           "diseased scans predicted NORMAL (lower is better).", "",
           "## Per-run results with 95% confidence intervals (cases resampled)", "",
           "| Method | Labels | Seed | Best round | Accuracy [95% CI] | Macro-F1 [95% CI] |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['label']} | {r['fraction']:.0%} | {r['seed']} | {r['best_round']} | "
                  f"{r['accuracy'] * 100:.2f} [{r['acc_lo'] * 100:.1f}-{r['acc_hi'] * 100:.1f}] | "
                  f"{r['macro_f1'] * 100:.2f} [{r['f1_lo'] * 100:.1f}-{r['f1_hi'] * 100:.1f}] |")
    md += ["", "## Figures", "", "![comparison](outputs/08_comparison.png)", "",
           "![confusion matrices](outputs/08_confusions.png)", "", "![rounds](outputs/08_rounds.png)", "",
           "## Published accuracy on the same dataset (for context only)", "", PUBLISHED, "",
           "The papers use different train/test splits and Paper 2 groups patients differently, so these are "
           "not directly comparable with the numbers above. No paper reports a 10% result.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(table_lines))
    print("saved RESULTS.md, outputs/08_comparison.png, outputs/08_confusions.png, outputs/08_rounds.png")


if __name__ == "__main__":
    main()
