"""03_data_check.py - check the data pipeline before any training.

Prints the labelled / unlabelled / val / test sizes for one seed and label fraction, saves a picture of
original, weakly and strongly augmented scans, and times a few training steps on the GPU.

Usage: python 03_data_check.py [--fraction 0.10] [--seed 0]
Output: outputs/03_data_check.txt, outputs/03_augmentations.png
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import load_config
from ssl_core import DEVICE, Data, Log, Net, class_weights, nll, normalise, set_seed, strong_aug, weak_aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    log = Log(cfg["output_dir"] / "03_data_check.txt")
    set_seed(0)

    log(f"device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'no GPU'})")
    data = Data(cfg, args.seed, args.fraction)
    log(f"image cache: {tuple(data.images.shape)} {data.images.dtype}")
    log(f"seed {args.seed}, label fraction {args.fraction:.0%}")
    data.describe(log)
    lab_cases = data.df.loc[data.lab, "case_id"].nunique()
    pool_cases = data.df.loc[data.pool, "case_id"].nunique()
    log(f"labelled cases {lab_cases}, unlabelled cases {pool_cases}, "
        f"overlap {len(set(data.df.loc[data.lab, 'case_id']) & set(data.df.loc[data.pool, 'case_id']))}")

    # picture: one row per class, columns = original, 2 weak views, 3 strong views
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.3))
    for r, name in enumerate(data.classes):
        row = data.lab[data.y[data.lab] == r][0]
        x = data.batch([row])
        views = [x, weak_aug(x), weak_aug(x), strong_aug(x), strong_aug(x), strong_aug(x)]
        titles = ["original", "weak", "weak", "strong", "strong", "strong"]
        for c, (v, t) in enumerate(zip(views, titles)):
            axes[r, c].imshow(v[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"{name} - {t}" if c == 0 else t, fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(cfg["output_dir"] / "03_augmentations.png", dpi=110)
    log("saved outputs/03_augmentations.png")

    # speed: 20 training steps of each network at the training batch size
    w = class_weights(data.y[data.lab])
    y = torch.as_tensor(data.y)
    for kind in ("flat", "hier"):
        model = Net(cfg, kind).to(DEVICE).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
        bs = cfg["train"]["batch_size"]
        for step in range(25):
            if step == 5:
                torch.cuda.synchronize() if DEVICE == "cuda" else None
                t0 = time.time()
            idx = data.lab[torch.randint(0, len(data.lab), (bs,)).numpy()]
            x = torch.cat([weak_aug(data.batch(idx[:20])), strong_aug(data.batch(idx[20:]))])
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logp = model(normalise(x))
            loss = nll(logp, y[idx].to(DEVICE), w)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        rate = 20 * bs / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else 0
        log(f"{kind} network: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, "
            f"{rate:.0f} images/s training, peak GPU memory {mem:.0f} MB, loss {loss.item():.3f}")
        del model, opt


if __name__ == "__main__":
    main()
