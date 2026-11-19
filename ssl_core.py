"""ssl_core.py - shared code: data, augmentation, network, training loop, teacher-student rounds, metrics.

Used by 03_data_check.py, 04_supervised.py, 05_its_faithful.py, 06_its_plus.py, 07_evaluate.py and demo.py.
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from common import ROOT, load_config, load_manifest

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARTIAL = 3  # pseudo-target meaning "abnormal, subtype unknown"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- logging / names
class Log:
    """Print a line and append it to a log file straight away."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def run_name(method, fraction, seed):
    return f"{method}_f{round(fraction * 100):03d}_s{seed}"


def run_dir(cfg, name):
    d = cfg["output_dir"] / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
class Data:
    """Images in memory plus the row indices of the labelled set, unlabelled pool, val and test."""

    def __init__(self, cfg, seed, fraction):
        out = cfg["output_dir"]
        df = load_manifest(cfg).merge(pd.read_csv(out / "splits.csv"), on="case_id")
        assert (df["row"].values == np.arange(len(df))).all(), "manifest rows must match the image cache"
        self.classes = cfg["classes"]
        df["y"] = df["label"].map({c: i for i, c in enumerate(self.classes)})
        train = df["split"] == "train"
        if fraction >= 1.0:
            labelled = train
        else:
            subs = pd.read_csv(out / "labelled_subsets.csv")
            cases = subs.loc[(subs["seed"] == seed) & (subs["min_fraction"] <= fraction + 1e-9), "case_id"]
            labelled = train & df["case_id"].isin(set(cases))
        self.df = df
        self.y = df["y"].to_numpy().copy()
        self.lab = df.index[labelled].to_numpy()
        self.pool = df.index[train & ~labelled].to_numpy()   # labels hidden; used only for the audit
        self.val = df.index[df["split"] == "val"].to_numpy()
        self.test = df.index[df["split"] == "test"].to_numpy()
        self.images = torch.from_numpy(np.load(cfg["image_cache"]))  # N x 224 x 224 uint8
        assert len(self.images) == len(df)
        self.seed, self.fraction = seed, fraction

    def batch(self, idx):
        """uint8 rows -> float tensor B x 1 x 224 x 224 in [0, 1] on the GPU."""
        return self.images[torch.as_tensor(np.array(idx))].to(DEVICE, non_blocking=True).unsqueeze(1).float() / 255.0

    def describe(self, log):
        counts = lambda idx: {c: int((self.y[idx] == i).sum()) for i, c in enumerate(self.classes)}
        log(f"labelled {len(self.lab)} {counts(self.lab)} | unlabelled pool {len(self.pool)} | "
            f"val {len(self.val)} | test {len(self.test)}")


# ----------------------------------------------------------------------------- augmentation (on the GPU)
def _affine(x, rot, scale, trans, shear):
    b, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(b, device=dev).uniform_(lo, hi)
    ang, sh = u(-rot, rot) * math.pi / 180, u(-shear, shear) * math.pi / 180
    sc = u(*scale)                                   # < 1 zooms in, which acts as a random crop
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    cos, sin, t = torch.cos(ang), torch.sin(ang), torch.tan(sh)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = sc * cos * flip
    theta[:, 0, 1] = sc * (cos * t - sin)
    theta[:, 1, 0] = sc * sin * flip
    theta[:, 1, 1] = sc * (sin * t + cos)
    theta[:, 0, 2], theta[:, 1, 2] = u(-trans, trans), u(-trans, trans)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def weak_aug(x):
    """Paper 1's augmentation: random crop, horizontal flip, small rotation, translation and shear."""
    return _affine(x, rot=10, scale=(0.85, 1.0), trans=0.06, shear=5)


def strong_aug(x):
    """Strong view for pseudo-labelled scans (Paper 2 uses RandAugment + Cutout; this is a GPU version
    with the same kinds of operations: stronger geometry, brightness / contrast / gamma, and Cutout)."""
    b, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)
    x = _affine(x, rot=20, scale=(0.70, 1.10), trans=0.10, shear=10)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mean) * u(0.6, 1.4) + mean + u(-0.15, 0.15)).clamp(0, 1)
    x = x.pow(u(0.7, 1.5))
    size = x.shape[-1]
    side = torch.randint(int(0.2 * size), int(0.4 * size), (b, 1, 1), device=dev)
    cy = torch.randint(0, size, (b, 1, 1), device=dev)
    cx = torch.randint(0, size, (b, 1, 1), device=dev)
    ar = torch.arange(size, device=dev)
    hole = ((ar.view(1, -1, 1) - cy).abs() < side // 2) & ((ar.view(1, 1, -1) - cx).abs() < side // 2)
    return x.masked_fill(hole.unsqueeze(1), 0.0)


def normalise(x):
    return (x.expand(-1, 3, -1, -1) - MEAN.to(x.device)) / STD.to(x.device)


# ----------------------------------------------------------------------------- network
def pretrained_backbone(cfg):
    path = ROOT / cfg["pretrained_weights"]
    if not path.exists():  # one-off download of the ImageNet weights, saved inside the project
        torch.save(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).state_dict(), path)
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.features


class Net(nn.Module):
    """EfficientNet-B0 feature extractor with either
       'flat': one 3-way head (Paper 1), or
       'hier': a Normal/Abnormal head and a Drusen/CNV head, multiplied back to 3 classes (Paper 2).
    forward() returns log-probabilities over [NORMAL, DRUSEN, CNV] in both cases."""

    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind
        self.features = pretrained_backbone(cfg)
        self.drop = nn.Dropout(cfg["train"]["dropout"])
        if kind == "flat":
            self.head = nn.Linear(1280, 3)
        else:
            self.binary = nn.Linear(1280, 2)
            self.fine = nn.Linear(1280, 2)

    def forward(self, x):
        f = self.drop(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)).float()
        if self.kind == "flat":
            return F.log_softmax(self.head(f), dim=1)
        b = F.log_softmax(self.binary(f), dim=1)   # [normal, abnormal]
        g = F.log_softmax(self.fine(f), dim=1)     # [drusen, cnv] given abnormal
        return torch.cat([b[:, :1], b[:, 1:] + g], dim=1)


def nll(logp, y, w):
    """Class-weighted negative log-likelihood. y in {0, 1, 2} is a normal target; y == PARTIAL means
    'abnormal, subtype unknown' and uses -log(p_drusen + p_cnv). For the 'hier' net this is exactly
    binary cross-entropy plus masked fine cross-entropy (Paper 2, Eq. 5)."""
    full = -logp.gather(1, y.clamp(max=2).unsqueeze(1)).squeeze(1)
    part = -torch.logsumexp(logp[:, 1:], dim=1)
    loss = torch.where(y == PARTIAL, part, full)
    return (loss * w[y]).sum() / w[y].sum()


def class_weights(y_lab):
    """Inverse-frequency weights from the labelled set (Paper 1 adjusts for class imbalance)."""
    n = np.bincount(y_lab, minlength=3).astype(float)
    w = n.sum() / (3 * n)
    return torch.tensor([*w, (w[1] + w[2]) / 2], dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def predict(data, model, idx, batch=256):
    """Class probabilities (len(idx) x 3) on un-augmented images."""
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(normalise(data.batch(idx[i:i + batch]))).exp().float().cpu())
    return torch.cat(out).numpy()


def load_model(cfg, path):
    ck = torch.load(path, map_location="cpu")
    model = Net(cfg, ck["kind"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    return model.eval(), ck


# ----------------------------------------------------------------------------- metrics
def metrics(y, p):
    pred = p.argmax(1)
    prec, rec, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    diseased = y > 0
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_auc": float(roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class="ovr", average="macro")),
        "under_grading_rate": float((pred[diseased] == 0).mean()),  # diseased scans called NORMAL (Paper 2)
        "precision": [float(v) for v in prec],
        "recall": [float(v) for v in rec],
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


# ----------------------------------------------------------------------------- training loop
def train(cfg, data, kind, log, pseudo_idx=None, pseudo_y=None, plus=False, init_state=None,
          lr=None, max_epochs=None, seed=0):
    """Train one network and return (best_state, best_val_metrics, history).

    Labelled scans always use weak augmentation.
    plus=False: labelled and pseudo-labelled scans are one pool, plain loss (supervised / ITS as published).
    plus=True : each batch has a fixed labelled share, pseudo-labelled scans get strong augmentation and
                their loss is multiplied by a weight that ramps from 0 to 1 (ITS+).
    The model kept is the epoch with the best validation macro-F1."""
    tc, pc = cfg["train"], cfg["its_plus"]
    set_seed(seed)
    model = Net(cfg, kind).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    lr = lr or tc["lr"]
    max_epochs = max_epochs or tc["max_epochs"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=tc["lr_patience"])
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    w = class_weights(data.y[data.lab])
    y_all = torch.as_tensor(data.y)
    rng = np.random.default_rng(seed)

    has_pseudo = pseudo_idx is not None and len(pseudo_idx) > 0
    plus = plus and has_pseudo
    bs = tc["batch_size"]
    if plus:
        n_lab = pc["labelled_per_batch"]
        n_pse = bs - n_lab
        steps = max(math.ceil(len(pseudo_idx) / n_pse), tc["min_steps_per_epoch"])
    else:
        idx = np.concatenate([data.lab, pseudo_idx]) if has_pseudo else data.lab
        tgt = torch.cat([y_all[data.lab], torch.as_tensor(pseudo_y)]) if has_pseudo else y_all[data.lab]
        steps = max(math.ceil(len(idx) / bs), tc["min_steps_per_epoch"])

    def sampler(n, size):  # endless stream of shuffled positions, 'size' at a time
        while True:
            order = rng.permutation(n)
            for i in range(0, n - size + 1, size) if n >= size else [0]:
                yield order[i:i + size] if n >= size else rng.choice(n, size)

    if plus:
        lab_stream, pse_stream = sampler(len(data.lab), n_lab), sampler(len(pseudo_idx), n_pse)
        pse_t = torch.as_tensor(pseudo_y)
    else:
        stream = sampler(len(idx), min(bs, len(idx)))

    best, best_state, history, bad = None, None, [], 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        lam = 1.0
        if plus:
            lam = float(np.clip((epoch - 1 - pc["ramp_start"]) / max(pc["ramp_end"] - pc["ramp_start"], 1), 0, 1))
        total = 0.0
        for _ in range(steps):
            if plus:
                li, pi = next(lab_stream), next(pse_stream)
                x = torch.cat([weak_aug(data.batch(data.lab[li])), strong_aug(data.batch(pseudo_idx[pi]))])
                y_l, y_p = y_all[data.lab[li]].to(DEVICE), pse_t[pi].to(DEVICE)
            else:
                bi = next(stream)
                x, yb = weak_aug(data.batch(idx[bi])), tgt[bi].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logp = model(normalise(x))
            if plus:
                loss = nll(logp[:len(li)], y_l, w) + lam * nll(logp[len(li):], y_p, w)
            else:
                loss = nll(logp, yb, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        m = metrics(data.y[data.val], predict(data, model, data.val))
        sched.step(m["macro_f1"])
        row = {"epoch": epoch, "train_loss": total / steps, "val_macro_f1": m["macro_f1"],
               "val_accuracy": m["accuracy"], "lr": opt.param_groups[0]["lr"], "lambda": lam,
               "seconds": time.time() - t0}
        history.append(row)
        improved = best is None or m["macro_f1"] > best["macro_f1"]
        if improved:
            best, bad = m, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"    epoch {epoch:2d}  loss {row['train_loss']:.4f}  val acc {m['accuracy']:.4f}  "
            f"val macro-F1 {m['macro_f1']:.4f}{' *' if improved else '  '}  lr {row['lr']:.0e}  "
            f"{row['seconds']:.0f}s")
        if bad >= tc["stop_patience"]:
            log(f"    early stop (no improvement for {bad} epochs)")
            break
    return best_state, best, history


def save_run(cfg, name, kind, state, extra, history=None):
    d = run_dir(cfg, name)
    torch.save({"kind": kind, "state": state, **extra}, d / "best.pt")
    if history is not None:
        pd.DataFrame(history).to_csv(d / "history.csv", index=False)


def supervised_teacher(cfg, data, kind, log):
    """Round-0 teacher = network trained on the labelled scans only. Reused if already trained."""
    name = run_name("supervised" if kind == "flat" else "supervised_hier", data.fraction, data.seed)
    path = run_dir(cfg, name) / "best.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu")
        log(f"round 0 teacher: reusing {name} (val macro-F1 {ck['val']['macro_f1']:.4f})")
        return ck["state"], ck["val"]
    log(f"round 0 teacher: training {name} on {len(data.lab)} labelled scans")
    state, val, hist = train(cfg, data, kind, log, seed=data.seed)
    save_run(cfg, name, kind, state, {"val": val}, hist)
    return state, val


# ----------------------------------------------------------------------------- pseudo-labelling
def window_mean(data, probs, window):
    """Mean of the teacher's probabilities over each pool scan and its neighbours in the same volume."""
    df = data.df.loc[data.pool, ["volume_id", "bscan_index"]].copy()
    df["pos"] = np.arange(len(df))
    out = np.empty_like(probs)
    half = window // 2
    for _, g in df.sort_values(["volume_id", "bscan_index"]).groupby("volume_id", sort=False):
        pos = g["pos"].to_numpy()
        p = probs[pos]
        csum = np.vstack([np.zeros((1, 3)), np.cumsum(p, 0)])
        lo = np.maximum(np.arange(len(pos)) - half, 0)
        hi = np.minimum(np.arange(len(pos)) + half + 1, len(pos))
        out[pos] = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    return out


def admit(cfg, data, probs, tau=None, adaptive=False, partial=False, volume=False):
    """Turn teacher probabilities on the pool into pseudo-labels.
    Returns (row indices, targets, per-class thresholds). Targets are 0/1/2 or PARTIAL."""
    pc = cfg["its_plus"]
    top, conf = probs.argmax(1), probs.max(1)
    if adaptive:
        # Learning status of each class: share of its expected scans the teacher is already sure of.
        # Classes the teacher finds hard get the low threshold, easy ones the high threshold (Paper 2).
        prior = np.bincount(data.y[data.lab], minlength=3) / len(data.lab)
        sure = np.array([((top == c) & (conf >= pc["tau_high"])).sum() for c in range(3)])
        status = sure / np.maximum(prior * len(probs), 1)
        taus = pc["tau_low"] + (pc["tau_high"] - pc["tau_low"]) * status / max(status.max(), 1e-9)
    else:
        taus = np.full(3, tau)
    p_abn = probs[:, 1] + probs[:, 2]
    fine_conf = np.maximum(probs[:, 1], probs[:, 2]) / np.maximum(p_abn, 1e-9)
    full = conf >= taus[top]
    if partial:
        full &= (top == 0) | (fine_conf >= pc["tau_fine"])
    part = np.zeros(len(probs), bool)
    if partial:
        part = ~full & (p_abn >= min(taus[1], taus[2]))
    if volume:
        ctx = window_mean(data, probs, pc["window"])
        ctx_abn = ctx[:, 1] + ctx[:, 2]
        agree_full = ctx.argmax(1) == top
        agree_abn = ctx_abn >= 0.5
        if partial:   # a confident abnormal scan whose subtype the neighbours dispute becomes partial
            part |= full & ~agree_full & (top > 0) & agree_abn
            part &= agree_abn
        full &= agree_full
        part &= ~full
    targets = np.where(full, top, PARTIAL)
    keep = full | part
    return data.pool[keep], targets[keep], taus


def audit(data, idx, targets):
    """Pseudo-label quality against the hidden true labels. Analysis only; never used for any decision."""
    true = data.y[idx]
    row = {"admitted": len(idx), "coverage": len(idx) / max(len(data.pool), 1)}
    full = targets != PARTIAL
    row["full_precision"] = float((targets[full] == true[full]).mean()) if full.any() else float("nan")
    for c, name in enumerate(data.classes):
        sel = targets == c
        row[f"n_{name}"] = int(sel.sum())
        row[f"precision_{name}"] = float((true[sel] == c).mean()) if sel.any() else float("nan")
    part = targets == PARTIAL
    row["n_PARTIAL"] = int(part.sum())
    row["precision_PARTIAL"] = float((true[part] > 0).mean()) if part.any() else float("nan")
    return row


def run_its(cfg, data, name, kind, log, rounds, taus=None, adaptive=False, partial=False, volume=False,
            plus=False, finetune=False):
    """Iterative teacher-student loop.
    Round 0: teacher trained on labelled scans. Each round: the teacher pseudo-labels the whole pool,
    a new student (ImageNet start) trains on labelled + pseudo-labelled scans, and becomes the teacher.
    The round with the best validation macro-F1 is saved as outputs/runs/<name>/best.pt."""
    data.describe(log)
    state, val = supervised_teacher(cfg, data, kind, log)
    best = {"state": state, "val": val, "round": 0}
    rows = [{"round": 0, "tau": "", "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]}]
    candidates = list(taus) if taus else [None]
    for r in range(1, rounds + 1):
        model = Net(cfg, kind).to(DEVICE)
        model.load_state_dict(state)
        probs = predict(data, model, data.pool)
        del model
        trials = []
        for tau in candidates:
            idx, tgt, used = admit(cfg, data, probs, tau, adaptive, partial, volume)
            a = audit(data, idx, tgt)
            label = "class-adaptive " + "/".join(f"{t:.3f}" for t in used) if adaptive else f"{tau:.2f}"
            log(f"\nround {r}  threshold {label}: admitted {a['admitted']} of {len(data.pool)} "
                f"({a['coverage']:.1%}), partial {a['n_PARTIAL']}, full-label precision {a['full_precision']:.4f}")
            log("  per class (count, precision): " + ", ".join(
                f"{c} {a['n_' + c]} / {a['precision_' + c]:.3f}" for c in data.classes))
            s, v, _ = train(cfg, data, kind, log, idx, tgt, plus=plus, seed=data.seed * 100 + r)
            log(f"  student val macro-F1 {v['macro_f1']:.4f}  val accuracy {v['accuracy']:.4f}")
            trials.append((v["macro_f1"], tau, s, v, a, label))
        f1, tau, state, val, a, label = max(trials, key=lambda t: t[0])
        if len(candidates) > 1:
            log(f"\nround {r}: best threshold {label} (val macro-F1 {f1:.4f}); kept for the next rounds")
            for t in trials:
                rows.append({"round": r, "tau": t[5], "val_macro_f1": t[0], "val_accuracy": t[3]["accuracy"],
                             "kept": t[1] == tau, **t[4]})
            candidates = [tau]
        else:
            rows.append({"round": r, "tau": label, "val_macro_f1": f1, "val_accuracy": val["accuracy"],
                         "kept": True, **a})
        if f1 > best["val"]["macro_f1"]:
            best = {"state": state, "val": val, "round": r}
        pd.DataFrame(rows).to_csv(run_dir(cfg, name) / "rounds.csv", index=False)

    if finetune:
        pc = cfg["its_plus"]
        log(f"\nfinal fine-tune of round {best['round']} on the labelled scans only (lr {pc['finetune_lr']})")
        s, v, _ = train(cfg, data, kind, log, init_state=best["state"], lr=pc["finetune_lr"],
                        max_epochs=pc["finetune_epochs"], seed=data.seed * 100 + 99)
        kept = v["macro_f1"] > best["val"]["macro_f1"]
        log(f"  fine-tuned val macro-F1 {v['macro_f1']:.4f} vs {best['val']['macro_f1']:.4f} -> "
            f"{'kept' if kept else 'discarded'}")
        rows.append({"round": "fine-tune", "tau": "", "val_macro_f1": v["macro_f1"],
                     "val_accuracy": v["accuracy"], "kept": kept})
        if kept:
            best = {"state": s, "val": v, "round": f"{best['round']}+fine-tune"}
        pd.DataFrame(rows).to_csv(run_dir(cfg, name) / "rounds.csv", index=False)

    save_run(cfg, name, kind, best["state"], {"val": best["val"], "round": best["round"]})
    log(f"\nsaved {name}: best round {best['round']}, val macro-F1 {best['val']['macro_f1']:.4f}, "
        f"val accuracy {best['val']['accuracy']:.4f}")
    return best
