"""01_data_audit.py - audit the dataset and build this project's manifest.

Checks:
  1. Class (patient folder) vs Label (per-B-scan) - which column is the target
  2. cases, volumes and images per case
  3. label mix inside a case
  4. byte-identical images (md5): within a case, across cases, and with conflicting labels
  5. duplicate groups: cases linked by shared images must stay in the same split
  6. image cache alignment

Outputs: outputs/manifest.csv, outputs/01_audit_report.txt
"""
import numpy as np
import pandas as pd

from common import Report, load_config, manifest_path

pd.set_option("display.width", 200)


def duplicate_groups(df):
    """Union-find over cases that share at least one byte-identical image."""
    parent = {c: c for c in df["case_id"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cases in df.groupby("md5")["case_id"].unique():
        for other in cases[1:]:
            ra, rb = find(cases[0]), find(other)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return df["case_id"].map(find)


def main():
    cfg = load_config()
    rep = Report()
    src = pd.read_csv(cfg["source_manifest"])
    raw = pd.read_csv(cfg["source_csv"])

    df = pd.DataFrame({
        "row": np.arange(len(src)),                       # row in the image cache
        "image_path": src["image_path"],
        "case_id": src["patient_id"],                     # e.g. CNV-001 = class folder + patient number
        "class_folder": src["class_folder"].str.upper(),
        "label": src["bscan_label"].str.upper(),          # per-B-scan label = training target
        "eye": src["eye"],
        "bscan_index": src["bscan_index"],
        "md5": src["md5"],
        "layout_outlier": src["excluded"].notna(),
    })
    df["volume_id"] = df["case_id"] + "-" + df["eye"]

    rep("1. SOURCE")
    rep(f"original CSV rows: {len(raw)}; verified images in manifest: {len(df)} "
        f"(difference {len(raw) - len(df)}: duplicate CSV rows and files missing from the archive)")

    rep("\n2. CLASS (folder) vs LABEL (per B-scan)")
    rep.table(pd.crosstab(df["class_folder"], df["label"], margins=True))
    off = (df["class_folder"] != df["label"]).mean()
    rep(f"B-scans whose label differs from their folder: {off:.1%} -> the target must be `label`, not the folder")

    rep("\n3. CASES AND VOLUMES")
    per_case = df.groupby("case_id").agg(class_folder=("class_folder", "first"), images=("row", "size"),
                                         volumes=("volume_id", "nunique"), labels=("label", "nunique"))
    summary = per_case.groupby("class_folder").agg(cases=("images", "size"), images=("images", "sum"),
                                                   volumes=("volumes", "sum"), min_img=("images", "min"),
                                                   median_img=("images", "median"), max_img=("images", "max"))
    summary.loc["All"] = [len(per_case), per_case["images"].sum(), per_case["volumes"].sum(),
                          per_case["images"].min(), per_case["images"].median(), per_case["images"].max()]
    rep.table(summary.astype(int))
    rep(f"unique raw patient numbers ignoring the class folder: {src['case_number'].nunique()} "
        f"(grouping on this alone merges unrelated cases)")

    rep("\n4. LABEL MIX INSIDE A CASE (number of distinct labels per case)")
    rep.table(pd.crosstab(per_case["class_folder"], per_case["labels"]))

    rep("\n5. BYTE-IDENTICAL IMAGES (md5)")
    g = df.groupby("md5").agg(n=("row", "size"), cases=("case_id", "nunique"), labels=("label", "nunique"))
    dup = g[g["n"] > 1]
    rep(f"md5 values shared by more than one image: {len(dup)} (covering {int(dup['n'].sum())} images)")
    rep(f"  within a single case: {(dup['cases'] == 1).sum()}")
    rep(f"  across different cases: {(dup['cases'] > 1).sum()}")
    rep(f"  identical image with conflicting labels: {(dup['labels'] > 1).sum()}")

    df["group_id"] = duplicate_groups(df)
    sizes = df.groupby("group_id")["case_id"].nunique()
    multi = sizes[sizes > 1]
    rep(f"\n6. DUPLICATE GROUPS (cases linked by shared images; split unit)")
    rep(f"groups: {len(sizes)} for {df['case_id'].nunique()} cases; groups with more than one case: {len(multi)}")
    for gid in multi.index:
        members = sorted(df.loc[df["group_id"] == gid, "case_id"].unique())
        shared = df[df["group_id"] == gid].groupby("md5")["case_id"].nunique().gt(1).sum()
        rep(f"  {' + '.join(members)}: {shared} shared images")

    rep("\n7. LAYOUT OUTLIERS (excluded from every split)")
    out = df[df["layout_outlier"]]
    rep(f"{out['case_id'].nunique()} cases, {len(out)} images: {sorted(out['case_id'].unique())}")

    rep("\n8. IMAGE CACHE")
    cache = np.load(cfg["image_cache"], mmap_mode="r")
    index = pd.read_csv(cfg["image_cache_index"])
    aligned = len(index) == len(df) and (index["image_path"].values == df["image_path"].values).all()
    rep(f"cache shape {cache.shape} {cache.dtype}; rows aligned with manifest: {aligned}")
    assert aligned and cache.shape[0] == len(df), "image cache is not aligned with the manifest"

    df.to_csv(manifest_path(cfg), index=False)
    rep(f"\nmanifest: {len(df)} images, {df['case_id'].nunique()} cases -> {manifest_path(cfg).name}")
    rep.save(cfg["output_dir"] / "01_audit_report.txt")


if __name__ == "__main__":
    main()
