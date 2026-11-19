"""04_supervised.py - supervised baseline: EfficientNet-B0 trained on the labelled scans only.

This is also the round-0 teacher of ITS (Paper 1, step 1).

Usage: python 04_supervised.py [--fraction 0.10] [--seed 0]      (--fraction 1.0 = all training labels)
Output: outputs/runs/supervised_f010_s0/ (best.pt, history.csv, log.txt)
"""
import argparse

from common import load_config
from ssl_core import Data, Log, run_dir, run_name, supervised_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    name = run_name("supervised", args.fraction, args.seed)
    log = Log(run_dir(cfg, name) / "log.txt")
    log(f"=== {name}: supervised EfficientNet-B0, {args.fraction:.0%} of training labels, seed {args.seed} ===")
    data = Data(cfg, args.seed, args.fraction)
    data.describe(log)
    _, val = supervised_teacher(cfg, data, "flat", log)
    log(f"\n{name}: val macro-F1 {val['macro_f1']:.4f}, val accuracy {val['accuracy']:.4f}")


if __name__ == "__main__":
    main()
