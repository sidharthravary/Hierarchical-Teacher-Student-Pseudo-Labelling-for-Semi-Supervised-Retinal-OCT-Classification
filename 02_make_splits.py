"""02_make_splits.py - case-level train/val/test split and nested labelled subsets.

Split unit: the duplicate group from 01 (a case, or cases that share byte-identical images),
stratified by class folder. Layout-outlier cases are left out of every split.

Labelled subsets: drawn inside the training cases only, stratified by class folder, one random
order per seed; each fraction takes a prefix of that order, so 5% is inside 10% is inside 20%.
Training images that are not labelled form the unlabelled pool (their labels stay in the manifest
for auditing pseudo-label quality only).

Outputs: outputs/splits.csv (case_id, split), outputs/labelled_subsets.csv (case_id, seed, min_fraction),
outputs/02_split_report.txt. The split is written once; rerun with --force to overwrite it.
"""
import sys

import numpy as np
import pandas as pd

from common import Report, load_config, load_manifest

pd.set_option("display.width", 200)


def split_groups(groups, fractions, seed):
    """groups: DataFrame indexed by group_id with columns stratum, n_cases. Returns Series group_id -> split."""
    rng = np.random.default_rng(seed)
    names = list(fractions)
    assigned = {}
    for _, sub in groups.groupby("stratum"):
        order = rng.permutation(sub.index.to_numpy())
        target = {s: fractions[s] * sub["n_cases"].sum() for s in names}
        filled = dict.fromkeys(names, 0)
        for gid in order:  # give each group to the split furthest below its target
            s = max(names, key=lambda k: (target[k] - filled[k]) / target[k])
            assigned[gid] = s
            filled[s] += sub.at[gid, "n_cases"]
    return pd.Series(assigned)


def labelled_subsets(train_cases, fractions, seeds):
    """train_cases: DataFrame with case_id, class_folder, group_id. Returns rows (case_id, seed, min_fraction)."""
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _, sub in train_cases.groupby("class_folder"):
            gids = rng.permutation(sub["group_id"].unique())
            order = [c for g in gids for c in sorted(sub.loc[sub["group_id"] == g, "case_id"])]
            taken = 0
            for frac in sorted(fractions):
                n = max(1, round(frac * len(order)))
                rows += [(c, seed, frac) for c in order[taken:n]]
                taken = max(taken, n)
    return pd.DataFrame(rows, columns=["case_id", "seed", "min_fraction"])


def main():
    cfg = load_config()
    out_splits = cfg["output_dir"] / "splits.csv"
    out_subsets = cfg["output_dir"] / "labelled_subsets.csv"
    if out_splits.exists() and "--force" not in sys.argv:
        sys.exit(f"{out_splits.name} already exists; the split is fixed. Pass --force to regenerate it.")

    rep = Report()
    df = load_manifest(cfg)
    cases = df.groupby("case_id").agg(class_folder=("class_folder", "first"), group_id=("group_id", "first"),
                                      layout_outlier=("layout_outlier", "first")).reset_index()
    excluded_groups = set(cases.loc[cases["layout_outlier"], "group_id"]) if cfg["exclude_layout_outliers"] else set()
    kept = cases[~cases["group_id"].isin(excluded_groups)]

    groups = kept.groupby("group_id").agg(stratum=("class_folder", "first"), n_cases=("case_id", "size"))
    group_split = split_groups(groups, cfg["split_fractions"], cfg["split_seed"])
    cases["split"] = cases["group_id"].map(group_split).fillna("excluded")
    cases[["case_id", "split"]].to_csv(out_splits, index=False)

    train_cases = cases[cases["split"] == "train"]
    subsets = labelled_subsets(train_cases, cfg["label_fractions"], cfg["label_seeds"])
    subsets.to_csv(out_subsets, index=False)

    # ---- report ----
    df = df.merge(cases[["case_id", "split"]], on="case_id")
    order = ["train", "val", "test", "excluded"]
    rep("1. CASES PER SPLIT (by class folder)")
    rep.table(pd.crosstab(cases["split"], cases["class_folder"], margins=True).reindex(order + ["All"]))
    rep("\n2. IMAGES PER SPLIT (by per-B-scan label = training target)")
    rep.table(pd.crosstab(df["split"], df["label"], margins=True).reindex(order + ["All"]))
    rep("\n3. VOLUMES PER SPLIT")
    rep.table(df.groupby("split")["volume_id"].nunique().reindex(order).to_frame("volumes"))

    rep("\n4. LABELLED SUBSETS (inside train; the rest of train is the unlabelled pool)")
    train_df = df[df["split"] == "train"]
    rows = []
    for seed in cfg["label_seeds"]:
        for frac in sorted(cfg["label_fractions"]):
            ids = subsets.loc[(subsets["seed"] == seed) & (subsets["min_fraction"] <= frac), "case_id"]
            lab = train_df[train_df["case_id"].isin(ids)]
            counts = lab["label"].value_counts()
            rows.append({"seed": seed, "fraction": frac, "cases": ids.nunique(), "images": len(lab),
                         **{c: int(counts.get(c, 0)) for c in cfg["classes"]},
                         "unlabelled_images": len(train_df) - len(lab)})
    rep.table(pd.DataFrame(rows).set_index(["seed", "fraction"]))

    rep("\n5. LEAKAGE CHECKS")
    main_df = df[df["split"] != "excluded"]
    checks = {
        "every case is in exactly one split": cases["case_id"].is_unique and cases["split"].notna().all(),
        "no duplicate group spans two splits": main_df.groupby("group_id")["split"].nunique().max() == 1,
        "no identical image (md5) appears in two splits": main_df.groupby("md5")["split"].nunique().max() == 1,
        "labelled cases are all training cases": set(subsets["case_id"]) <= set(train_cases["case_id"]),
        "subsets are nested and every class is present": all(r[c] > 0 for r in rows for c in cfg["classes"]),
    }
    for name, ok in checks.items():
        rep(f"  [{'OK' if ok else 'FAIL'}] {name}")
    rep.save(cfg["output_dir"] / "02_split_report.txt")
    if not all(checks.values()):
        sys.exit("leakage check failed")


if __name__ == "__main__":
    main()
