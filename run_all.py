"""run_all.py - run the whole project in order. Finished trainings are skipped, so it can be restarted.

Usage:
  python run_all.py                     main experiment: 10% labels, seed 0
  python run_all.py --seeds 0 1 2       more seeds
  python run_all.py --ablation          also the ITS+ ablation variants A, B, C
  python run_all.py --fractions 0.05 0.10 0.20 --reference
                                        other label fractions and the 100%-label supervised reference
"""
import argparse
import subprocess
import sys

from common import load_config
from ssl_core import run_dir, run_name


def step(script, *args, done=None):
    if done is not None and done.exists():
        print(f"[skip] {script} {' '.join(args)} (already done)")
        return
    print(f"\n[run ] {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, script, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.10])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--reference", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    out = cfg["output_dir"]
    best = lambda method, f, s: run_dir(cfg, run_name(method, f, s)) / "best.pt"

    step("01_data_audit.py", done=out / "manifest.csv")
    step("02_make_splits.py", done=out / "splits.csv")
    step("03_data_check.py", done=out / "03_data_check.txt")
    for s in args.seeds:
        for f in args.fractions:
            fs = ["--fraction", str(f), "--seed", str(s)]
            step("04_supervised.py", *fs, done=best("supervised", f, s))
            step("05_its_faithful.py", *fs, done=best("its", f, s))
            step("06_its_plus.py", "--variant", "D", *fs, done=best("its_plus", f, s))
            if args.ablation:
                for v in "ABC":
                    step("06_its_plus.py", "--variant", v, *fs, done=best(f"its_plus_{v}", f, s))
    if args.reference:
        step("04_supervised.py", "--fraction", "1.0", "--seed", "0", done=best("supervised", 1.0, 0))
    step("07_evaluate.py")
    step("08_report.py")
    step("09_make_notebook.py")
    step("10_unsupervised.py", done=out / "unsupervised_results.csv")
    step("11_comparison.py")


if __name__ == "__main__":
    main()
