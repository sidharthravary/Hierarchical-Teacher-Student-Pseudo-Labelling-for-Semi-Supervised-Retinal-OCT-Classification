"""Shared helpers: config, paths, report writer."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("source_manifest", "source_csv", "image_cache", "image_cache_index", "output_dir"):
        cfg[key] = (ROOT / cfg[key]).resolve()
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def manifest_path(cfg):
    return cfg["output_dir"] / "manifest.csv"


def load_manifest(cfg):
    return pd.read_csv(manifest_path(cfg))


class Report:
    """Print and collect lines, then save them as a text report."""

    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text)
        self.lines.append(text)

    def table(self, df):
        self(df.to_string())

    def save(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\nsaved {path}")
