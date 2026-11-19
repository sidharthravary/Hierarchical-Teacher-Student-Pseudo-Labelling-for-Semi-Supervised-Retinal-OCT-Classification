"""06_its_plus.py - ITS+ : the same teacher-student loop with additions from Papers 2 and 3.

Variants (each adds one thing to the one before, for the ablation table):
  A  two-head network (Normal/Abnormal + Drusen/CNV), fixed threshold 0.95            [Paper 2, 3.2]
  B  A + class-adaptive thresholds and full / partial / reject admission               [Paper 2, 3.3]
  C  B + agreement with the average of 5 neighbouring B-scans in the same volume       [Paper 2, 3.4]
  D  C + strong augmentation and ramped loss weight for pseudo-labels, then a final
     fine-tune on the true labels = the full ITS+ model                                [Papers 2, 3]

Usage: python 06_its_plus.py [--variant D] [--fraction 0.10] [--seed 0]
Output: outputs/runs/its_plus_f010_s0/ (variant D) or its_plus_A_f010_s0/ etc.
"""
import argparse

from common import load_config
from ssl_core import Data, Log, run_dir, run_its, run_name

VARIANTS = {
    "A": dict(taus=[0.95]),
    "B": dict(adaptive=True, partial=True),
    "C": dict(adaptive=True, partial=True, volume=True),
    "D": dict(adaptive=True, partial=True, volume=True, plus=True, finetune=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="D")
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    method = "its_plus" if args.variant == "D" else f"its_plus_{args.variant}"
    name = run_name(method, args.fraction, args.seed)
    log = Log(run_dir(cfg, name) / "log.txt")
    log(f"=== {name}: ITS+ variant {args.variant} {VARIANTS[args.variant]}, "
        f"{args.fraction:.0%} labels, seed {args.seed} ===")
    data = Data(cfg, args.seed, args.fraction)
    run_its(cfg, data, name, "hier", log, rounds=cfg["its_plus"]["rounds"], **VARIANTS[args.variant])


if __name__ == "__main__":
    main()
