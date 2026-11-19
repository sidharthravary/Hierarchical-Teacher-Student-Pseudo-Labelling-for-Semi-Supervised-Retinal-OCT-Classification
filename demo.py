"""demo.py - show in the terminal that the trained models work (takes well under a minute).

Loads the saved models for one seed and label fraction, classifies the held-out test scans and prints
the comparison: supervised baseline vs ITS (Paper 1) vs ITS+ (ours), the confusion matrix of ITS+,
and a few individual predictions. Nothing is trained here.

Usage: python demo.py [--fraction 0.10] [--seed 0]
"""
import argparse

import numpy as np

from common import load_config
from ssl_core import DEVICE, Data, load_model, metrics, predict, run_dir, run_name

METHODS = [("supervised", "Supervised (labelled only)"), ("its", "ITS (Paper 1)"), ("its_plus", "ITS+ (ours)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    data = Data(cfg, args.seed, args.fraction)
    y = data.y[data.test]
    print(f"\nNEH_UT_2021 OCT, 3 classes {data.classes}, device {DEVICE}")
    print(f"labelled scans {len(data.lab)} ({args.fraction:.0%} of training cases), "
          f"unlabelled scans {len(data.pool)}, test scans {len(y)} (never used in training)\n")
    print(f"{'model':<28}{'accuracy':>10}{'macro-F1':>10}{'macro-AUC':>11}{'under-grading':>15}")
    print("-" * 74)
    last = None
    for method, label in METHODS:
        path = run_dir(cfg, run_name(method, args.fraction, args.seed)) / "best.pt"
        if not path.exists():
            print(f"{label:<28}  (not trained yet)")
            continue
        model, _ = load_model(cfg, path)
        p = predict(data, model, data.test)
        m = metrics(y, p)
        print(f"{label:<28}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['macro_auc']:>11.4f}"
              f"{m['under_grading_rate']:>15.4f}")
        last = (label, p, m)
    if last is None:
        return
    label, p, m = last
    print(f"\nConfusion matrix of {label} (rows = true class, columns = predicted)")
    print(f"{'':>10}" + "".join(f"{c:>9}" for c in data.classes) + f"{'recall':>9}")
    for i, c in enumerate(data.classes):
        print(f"{c:>10}" + "".join(f"{v:>9}" for v in m["confusion"][i]) + f"{m['recall'][i]:>9.3f}")
    print(f"\nTen test scans classified by {label}:")
    rng = np.random.default_rng(args.seed)
    for j in rng.choice(len(y), 10, replace=False):
        row = data.df.loc[data.test[j]]
        pred = int(p[j].argmax())
        print(f"  {row['image_path']:<28} true {data.classes[y[j]]:<7} predicted {data.classes[pred]:<7}"
              f" confidence {p[j, pred]:.2f}  {'correct' if pred == y[j] else 'WRONG'}")


if __name__ == "__main__":
    main()
