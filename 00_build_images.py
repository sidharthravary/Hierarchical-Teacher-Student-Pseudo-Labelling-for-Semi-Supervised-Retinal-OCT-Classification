"""00_build_images.py - build data/images.npy from the original NEH_UT_2021 dataset.

data/images.npy is not on GitHub (805 MB). This script recreates it from the raw dataset, exactly as the
original preprocessing did: grayscale, black out the burned-in scale bar, crop the retina window recorded
in data/index.csv (crop_top..crop_bottom, full width), resize to 224 x 224 (bilinear), uint8.

The source can be the extracted dataset folder (e.g. Kaggle's "archive" folder), the Mendeley zip, or a
folder that contains the dataset zip. Image paths are matched from the class folder onward
(CNV/..., DRUSEN/..., NORMAL/...), so the folder layout above that does not matter.

Usage: python 00_build_images.py archive            (or the path to the dataset folder / zip)
Output: data/images.npy  (16803 x 224 x 224 uint8, row order of data/index.csv)
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLASSES = {"NORMAL", "DRUSEN", "CNV"}
EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}
SCALE_BAR = (slice(430, 492), slice(0, 86))   # rows, cols blacked out before cropping (as in the original)
SIZE = 224


def rel_path(path):
    """Part of the path from the class folder onward, e.g. CNV/1/000_Drusen.jpg."""
    parts = PurePosixPath(path).parts
    i = next((k for k, p in enumerate(parts) if p.upper() in CLASSES), None)
    return None if i is None else "/".join(parts[i:])


def add_zip(zf, readers):
    """Register every image in a zip, opening one level of nested zips."""
    for info in zf.infolist():
        name = info.filename
        if name.lower().endswith(".zip"):
            add_zip(zipfile.ZipFile(io.BytesIO(zf.read(info))), readers)
        elif PurePosixPath(name).suffix.lower() in EXTS:
            rel = rel_path(name)
            if rel:
                readers[rel] = (lambda z=zf, n=info: z.read(n))


def find_images(source):
    """Map rel path -> function returning the image bytes."""
    readers = {}
    src = Path(source)
    if src.is_file() and src.suffix.lower() == ".zip":
        add_zip(zipfile.ZipFile(src), readers)
    elif src.is_dir():
        for p in src.rglob("*"):
            if p.suffix.lower() in EXTS:
                rel = rel_path(p.relative_to(src).as_posix())
                if rel:
                    readers[rel] = (lambda q=p: q.read_bytes())
            elif p.suffix.lower() == ".zip":
                print(f"reading images inside {p.name} ...", flush=True)
                add_zip(zipfile.ZipFile(p), readers)
    else:
        sys.exit(f"ERROR: '{source}' is not a folder or a .zip file.")
    return readers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="archive", help="dataset folder or zip (default: archive)")
    ap.add_argument("--out", default=str(ROOT / "data" / "images.npy"))
    args = ap.parse_args()

    index = pd.read_csv(ROOT / "data" / "index.csv")
    print(f"looking for {len(index)} images in {args.source} ...", flush=True)
    readers = find_images(args.source)
    missing = [p for p in index["image_path"] if p not in readers]
    if missing:
        sys.exit(f"ERROR: {len(missing)} of {len(index)} images not found, e.g. {missing[:3]}. "
                 "Point the script at the folder that contains the CNV, DRUSEN and NORMAL folders.")

    out = Path(args.out)
    tmp = out.with_suffix(".npy.tmp")
    images = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(index), SIZE, SIZE))
    for i, r in enumerate(index.itertuples()):
        with Image.open(io.BytesIO(readers[r.image_path]())) as img:
            gray = np.asarray(img.convert("L")).copy()
        gray[SCALE_BAR] = 0
        crop = gray[int(r.crop_top):int(r.crop_bottom)]
        images[i] = np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.BILINEAR))
        if (i + 1) % 2000 == 0 or i + 1 == len(index):
            print(f"  {i + 1} / {len(index)}", flush=True)
    images.flush()
    del images
    tmp.replace(out)
    print(f"saved {out} ({len(index)} x {SIZE} x {SIZE} uint8)")


if __name__ == "__main__":
    main()
