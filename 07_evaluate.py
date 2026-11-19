"""07_evaluate.py - test-set evaluation of every trained model in outputs/runs/.

For each run: accuracy, macro-F1, macro-AUC, under-grading rate, per-class precision / recall and the
confusion matrix on the held-out test cases, plus 95% bootstrap confidence intervals obtained by
resampling whole cases (scans of one patient case are not independent).

Usage: python 07_evaluate.py
Output: outputs/results.csv, outputs/metrics/<run>.json, outputs/07_evaluation.txt
"""
import json
import re

import numpy as np
import pandas as pd

from common import load_config
from ssl_core import Data, Log, load_model, metrics, predict

LABELS = {"supervised": "Supervised", "supervised_hier": "Supervised (two-head)", "its": "ITS (Paper 1)",
          "its_plus_A": "ITS+ A: two-head", "its_plus_B": "ITS+ B: + adaptive/partial",
          "its_plus_C": "ITS+ C: + volume check", "its_plus": "ITS+ (full)"}
ORDER = list(LABELS)


def case_bootstrap(y, pred, cases, n_boot=1000, seed=0):
    """95% CI of accuracy and macro-F1 when test cases (not single scans) are resampled."""
    ids, inv = np.unique(cases, return_inverse=True)
    cm = np.zeros((len(ids), 3, 3))
    np.add.at(cm, (inv, y, pred), 1)
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(n_boot):
        c = cm[rng.integers(0, len(ids), len(ids))].sum(0)
        tp = np.diag(c)
        acc.append(tp.sum() / c.sum())
        f1.append(np.mean(2 * tp / np.maximum(c.sum(0) + c.sum(1), 1)))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(acc), q(f1)


def main():
    cfg = load_config()
    out = cfg["output_dir"]
    (out / "metrics").mkdir(exist_ok=True)
    log = Log(out / "07_evaluation.txt")
    data = Data(cfg, 0, 0.10)          # the test set is the same for every seed and label fraction
    y = data.y[data.test]
    cases = data.df.loc[data.test, "case_id"].to_numpy()
    log(f"test set: {len(y)} scans from {len(set(cases))} cases "
        f"({', '.join(f'{c} {int((y == i).sum())}' for i, c in enumerate(data.classes))})\n")

    rows = []
    for d in sorted((out / "runs").iterdir()):
        m = re.fullmatch(r"(.+)_f(\d+)_s(\d+)", d.name)
        if not m or not (d / "best.pt").exists():
            continue
        model, ck = load_model(cfg, d / "best.pt")
        p = predict(data, model, data.test)
        res = metrics(y, p)
        res["accuracy_ci95"], res["macro_f1_ci95"] = case_bootstrap(y, p.argmax(1), cases)
        res.update(run=d.name, method=m.group(1), fraction=int(m.group(2)) / 100, seed=int(m.group(3)),
                   best_round=str(ck.get("round", 0)), val_macro_f1=ck["val"]["macro_f1"])
        (out / "metrics" / f"{d.name}.json").write_text(json.dumps(res, indent=2))
        rows.append({"method": res["method"], "label": LABELS.get(res["method"], res["method"]),
                     "fraction": res["fraction"], "seed": res["seed"], "best_round": res["best_round"],
                     "val_macro_f1": res["val_macro_f1"], "accuracy": res["accuracy"],
                     "acc_lo": res["accuracy_ci95"][0], "acc_hi": res["accuracy_ci95"][1],
                     "macro_f1": res["macro_f1"], "f1_lo": res["macro_f1_ci95"][0],
                     "f1_hi": res["macro_f1_ci95"][1], "macro_auc": res["macro_auc"],
                     "under_grading_rate": res["under_grading_rate"],
                     **{f"recall_{c}": res["recall"][i] for i, c in enumerate(data.classes)}})
        del model

    df = pd.DataFrame(rows)
    df["order"] = df["method"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    df = df.sort_values(["fraction", "order", "seed"]).drop(columns="order")
    df.to_csv(out / "results.csv", index=False)

    log("TEST RESULTS (scan level; accuracy and macro-F1 with 95% case-bootstrap interval)")
    log(f"{'method':<28}{'labels':>7}{'seed':>5}{'round':>13}{'accuracy':>22}{'macro-F1':>22}{'AUC':>8}{'UGR':>8}")
    for _, r in df.iterrows():
        log(f"{r['label']:<28}{r['fraction']:>7.0%}{r['seed']:>5}{r['best_round']:>13}"
            f"{r['accuracy']:>9.4f} [{r['acc_lo']:.3f}-{r['acc_hi']:.3f}]"
            f"{r['macro_f1']:>9.4f} [{r['f1_lo']:.3f}-{r['f1_hi']:.3f}]"
            f"{r['macro_auc']:>8.4f}{r['under_grading_rate']:>8.4f}")
    log("\nUGR = under-grading rate: share of diseased scans (Drusen or CNV) predicted NORMAL; lower is better.")
    log("saved outputs/results.csv and outputs/metrics/*.json")


if __name__ == "__main__":
    main()
