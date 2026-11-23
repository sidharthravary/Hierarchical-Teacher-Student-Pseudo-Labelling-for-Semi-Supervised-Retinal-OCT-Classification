"""live_demo.py - live training demonstration (a few epochs), best-checkpoint saving and test benchmarking.

What it does, step by step:
  1. Loads the data split (10% labelled scans, unlabelled pool, validation and test sets).
  2. Loads the trained round-0 teacher checkpoint (hierarchical EfficientNet-B0 trained on the labelled scans).
  3. Teacher pseudo-labels the unlabelled pool with the ITS+ admission rule
     (class-adaptive thresholds, full / partial / reject, neighbouring-scan agreement).
  4. Trains a new student LIVE for a few epochs on labelled + pseudo-labelled scans, printing loss and
     validation accuracy / macro-F1 after every epoch.
  5. Saves the best epoch (highest validation macro-F1) as a checkpoint: outputs/live_demo/best_model.pt
  6. Reloads that checkpoint from disk and benchmarks it on the unseen test set
     (accuracy, macro-F1, macro-AUC, per-class precision / recall, confusion matrix).
  7. Benchmarks the stored best ITS+ checkpoint from the full experiment on the same test set.
  8. Saves loss / accuracy curves of the live run: outputs/live_demo/training_curves.png

Usage: python live_demo.py [--epochs 3] [--steps 60] [--seed 0]
       (--steps = training batches per epoch; lower it for a faster demo)
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import load_config
from ssl_core import (DEVICE, Data, Log, admit, audit, load_model, metrics, predict, run_dir, run_name,
                      set_seed, train)


def show_test(log, name, m, classes):
    """Print the test-set benchmark of one model."""
    log(f"\n  TEST RESULTS - {name}")
    log(f"    accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | macro-AUC {m['macro_auc']:.4f} | "
        f"under-grading rate {m['under_grading_rate']:.4f}")
    log(f"    {'class':<8}{'precision':>11}{'recall':>9}")
    for i, c in enumerate(classes):
        log(f"    {c:<8}{m['precision'][i]:>11.3f}{m['recall'][i]:>9.3f}")
    log("    confusion matrix (rows = true, columns = predicted):")
    log(f"    {'':>8}" + "".join(f"{c:>8}" for c in classes))
    for i, c in enumerate(classes):
        log(f"    {c:>8}" + "".join(f"{v:>8}" for v in m["confusion"][i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3, help="number of live training epochs")
    ap.add_argument("--steps", type=int, default=60, help="training batches per epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    out = cfg["output_dir"] / "live_demo"
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt")
    set_seed(args.seed)
    t_start = time.time()

    # ---- 1. data
    log(f"=== LIVE DEMO: one teacher-student round, {args.epochs} epochs x {args.steps} batches, seed {args.seed} ===")
    log(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    data = Data(cfg, args.seed, 0.10)
    data.describe(log)

    # ---- 2. teacher checkpoint
    teacher_path = run_dir(cfg, run_name("supervised_hier", 0.10, args.seed)) / "best.pt"
    teacher, ck = load_model(cfg, teacher_path)
    log(f"\n[1] teacher loaded from {teacher_path.relative_to(cfg['output_dir'].parent).as_posix()} "
        f"(validation macro-F1 {ck['val']['macro_f1']:.4f})")

    # ---- 3. pseudo-labelling
    probs = predict(data, teacher, data.pool)
    del teacher
    idx, tgt, taus = admit(cfg, data, probs, adaptive=True, partial=True, volume=True)
    a = audit(data, idx, tgt)
    log(f"[2] teacher pseudo-labelled the unlabelled pool: thresholds NORMAL/DRUSEN/CNV = "
        f"{'/'.join(f'{t:.3f}' for t in taus)}")
    log(f"    admitted {a['admitted']} of {len(data.pool)} scans ({a['coverage']:.1%}), of which "
        f"{a['n_PARTIAL']} partial ('abnormal, type unknown'); full-label precision {a['full_precision']:.4f}")

    # ---- 4. live student training (short schedule for the demonstration)
    pc = cfg["its_plus"]
    n_pse = cfg["train"]["batch_size"] - pc["labelled_per_batch"]
    rng = np.random.default_rng(args.seed)
    keep = rng.choice(len(idx), min(len(idx), args.steps * n_pse), replace=False)
    cfg["train"]["max_epochs"] = args.epochs
    cfg["train"]["min_steps_per_epoch"] = args.steps
    pc["ramp_start"], pc["ramp_end"] = 0, max(args.epochs - 1, 1)   # pseudo-label weight 0 -> 1 within the demo
    log(f"\n[3] training the student live: each batch = {pc['labelled_per_batch']} labelled (weak augmentation) + "
        f"{n_pse} pseudo-labelled (strong augmentation); pseudo-label loss weight ramps 0 -> 1")
    state, val, history = train(cfg, data, "hier", log, idx[keep], tgt[keep], plus=True, seed=args.seed * 100 + 1)

    # ---- 5. save the best checkpoint
    ckpt = out / "best_model.pt"
    best_epoch = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    torch.save({"kind": "hier", "state": state, "val": val, "round": f"live demo, epoch {best_epoch}"}, ckpt)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    log(f"\n[4] best epoch = {best_epoch} (validation macro-F1 {val['macro_f1']:.4f}) -> "
        f"checkpoint saved: outputs/live_demo/best_model.pt ({ckpt.stat().st_size / 2**20:.1f} MB)")

    # ---- 6. reload the checkpoint from disk and benchmark it on the unseen test set
    model, _ = load_model(cfg, ckpt)
    log("[5] checkpoint reloaded from disk: outputs/live_demo/best_model.pt")
    m_live = metrics(data.y[data.test], predict(data, model, data.test))
    show_test(log, f"live-demo student (best epoch {best_epoch}), {len(data.test)} unseen test scans", m_live,
              data.classes)

    # ---- 7. benchmark the stored best checkpoint of the full experiment
    full_path = run_dir(cfg, run_name("its_plus", 0.10, args.seed)) / "best.pt"
    model, ck = load_model(cfg, full_path)
    m_full = metrics(data.y[data.test], predict(data, model, data.test))
    log(f"\n[6] stored best checkpoint of the full experiment: "
        f"{full_path.relative_to(cfg['output_dir'].parent).as_posix()} (best round {ck['round']})")
    show_test(log, "full ITS+ model (4 rounds, early stopping)", m_full, data.classes)

    # ---- 8. curves of the live run
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(h["epoch"], h["train_loss"], marker="o", color="#b5532a")
    axes[0].set_title("Training loss (class-weighted NLL)")
    axes[1].plot(h["epoch"], h["val_accuracy"] * 100, marker="o", label="validation accuracy", color="#1b7f55")
    axes[1].plot(h["epoch"], h["val_macro_f1"] * 100, marker="s", label="validation macro-F1", color="#3b78c2")
    axes[1].axvline(best_epoch, color="#888888", linestyle="--", linewidth=1)
    axes[1].text(best_epoch, axes[1].get_ylim()[0], " best checkpoint", color="#555555", fontsize=8, va="bottom")
    axes[1].set_title("Validation accuracy / macro-F1 (%)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xticks(h["epoch"])
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)
    log(f"\n[7] saved outputs/live_demo/training_curves.png and history.csv  "
        f"(total time {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
