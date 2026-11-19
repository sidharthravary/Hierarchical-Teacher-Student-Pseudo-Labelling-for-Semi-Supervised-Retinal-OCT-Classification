"""05_its_faithful.py - Iterative Teacher-Student (ITS) as published by Alizadeh et al. (Paper 1).

Round 0: the supervised model is the teacher. Each round: the teacher predicts the unlabelled pool,
scans with confidence >= tau get a hard pseudo-label, a student trains on labelled + pseudo-labelled
scans with plain cross-entropy, and the student becomes the next teacher. tau is swept in round 1
and the best value (validation macro-F1) is kept for the later rounds.

Usage: python 05_its_faithful.py [--fraction 0.10] [--seed 0]
Output: outputs/runs/its_f010_s0/ (best.pt, rounds.csv, log.txt)
"""
import argparse

from common import load_config
from ssl_core import Data, Log, run_dir, run_its, run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    name = run_name("its", args.fraction, args.seed)
    log = Log(run_dir(cfg, name) / "log.txt")
    log(f"=== {name}: ITS as published, {args.fraction:.0%} labels, seed {args.seed} ===")
    data = Data(cfg, args.seed, args.fraction)
    run_its(cfg, data, name, "flat", log, rounds=cfg["its"]["rounds"], taus=cfg["its"]["taus"])


if __name__ == "__main__":
    main()
