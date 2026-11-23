# Code guide

What every file does, where each part of the method lives in the code, and what each output file contains.
Line links point to the current version of each file.

## 1. How the pieces connect

```
config.yaml ──► common.py (load_config) ──► every script
                                     │
data/ (images.npy, index.csv, ...)   ▼
   │                          ssl_core.py  (all shared logic: data, network, training, pseudo-labelling, ITS loop)
   ▼                                 │
01_data_audit.py ─► outputs/manifest.csv
02_make_splits.py ─► outputs/splits.csv, outputs/labelled_subsets.csv
03_data_check.py ─► sanity checks, augmentation picture, GPU speed
04_supervised.py ─► outputs/runs/supervised_*            (also the round-0 teacher of ITS)
05_its_faithful.py ─► outputs/runs/its_*                 (Paper 1)
06_its_plus.py ─► outputs/runs/its_plus_*                (ours; variants A, B, C, D)
07_evaluate.py ─► outputs/results.csv, outputs/metrics/*.json   (test set, every run)
08_report.py ─► RESULTS.md, outputs/08_*.png
09_make_notebook.py ─► results.ipynb (executed)
10_unsupervised.py ─► outputs/unsupervised_results.csv   (k-means reference)
11_comparison.py ─► COMPARISON.md
12_extra_figures.py ─► training curves, ROC curves, per-class scores, cluster plot, method diagram

run_all.py runs 01 → 12 in order and skips anything already finished.
demo.py loads the saved best checkpoints and prints the test results in the terminal.
live_demo.py trains a student live for a few epochs, saves the best checkpoint, reloads it and benchmarks it on the test set.
00_build_images.py rebuilds data/images.npy from the raw NEH_UT_2021 download (only needed if images.npy is missing).
```

## 2. Files

| File | What it does | Main output |
|---|---|---|
| [config.yaml](config.yaml) | All paths and settings: data files, split fractions, label fractions, seeds, training settings, ITS and ITS+ settings | – |
| [common.py](common.py) | Loads `config.yaml` and turns its paths into absolute paths; reads the manifest; a small report writer | – |
| [ssl_core.py](ssl_core.py) | Shared library used by every training and evaluation script (section 3) | – |
| [01_data_audit.py](01_data_audit.py) | Checks the dataset: per-scan label vs folder class, cases and volumes, duplicate images (md5), cases linked by duplicates, layout outliers, alignment of the image file with the metadata | `outputs/manifest.csv`, `outputs/01_audit_report.txt` |
| [02_make_splits.py](02_make_splits.py) | Splits cases into train / validation / test (60 / 15 / 25%), keeping linked cases together; draws nested labelled subsets (5% ⊂ 10% ⊂ 20%) for seeds 0, 1, 2; runs leakage checks | `outputs/splits.csv`, `outputs/labelled_subsets.csv`, `outputs/02_split_report.txt` |
| [03_data_check.py](03_data_check.py) | Prints set sizes, saves examples of weak and strong augmentation, times training on the GPU | `outputs/03_data_check.txt`, `outputs/03_augmentations.png` |
| [04_supervised.py](04_supervised.py) | Supervised baseline: EfficientNet-B0 trained on the labelled scans only | `outputs/runs/supervised_f010_s<seed>/` |
| [05_its_faithful.py](05_its_faithful.py) | ITS as published (Paper 1): flat 3-class network, one confidence threshold (0.95 or 0.99 chosen in round 1), 4 rounds | `outputs/runs/its_f010_s<seed>/` |
| [06_its_plus.py](06_its_plus.py) | Our method, ITS+. `--variant A/B/C/D` switches components on one at a time for the ablation; D is the full model | `outputs/runs/its_plus_f010_s<seed>/`, `its_plus_A_...`, `_B_`, `_C_` |
| [07_evaluate.py](07_evaluate.py) | Scores every trained model on the test set: accuracy, macro-F1, macro-AUC, under-grading rate, per-class precision and recall, confusion matrix, 95% confidence intervals by resampling patient cases | `outputs/results.csv`, `outputs/metrics/<run>.json`, `outputs/07_evaluation.txt` |
| [08_report.py](08_report.py) | Mean ± SD tables over seeds and three figures | `RESULTS.md`, `outputs/08_comparison.png`, `08_confusions.png`, `08_rounds.png` |
| [09_make_notebook.py](09_make_notebook.py) | Builds and runs a notebook with the demo, tables, figures and logs, so it opens with outputs already shown | `results.ipynb` |
| [10_unsupervised.py](10_unsupervised.py) | Unsupervised reference: ImageNet EfficientNet-B0 features → PCA (50) → k-means (3 clusters), no training labels; validation labels only name the clusters | `outputs/unsupervised_results.csv`, `outputs/10_unsupervised.txt` |
| [11_comparison.py](11_comparison.py) | Builds the full comparison page: all models, per seed, per class, ablation, pseudo-label quality, published results, supervised vs unsupervised vs semi-supervised | `COMPARISON.md` |
| [12_extra_figures.py](12_extra_figures.py) | Training-loss and validation curves per round (parsed from the logs), ROC curves, per-class precision / recall / F1, PCA + k-means cluster plot with variance retained, method diagram | `outputs/12_*.png` |
| [live_demo.py](live_demo.py) | Live demonstration: teacher pseudo-labels the pool, a student trains for a few epochs (`--epochs`, `--steps`), the best epoch is saved as `outputs/live_demo/best_model.pt`, reloaded and benchmarked on the unseen test set together with the stored best ITS+ checkpoint | `outputs/live_demo/` |
| [00_build_images.py](00_build_images.py) | Rebuilds `data/images.npy` from the raw dataset folder or zip (grayscale, scale-bar mask, retina crop, 224 x 224) | `data/images.npy` |
| [demo.py](demo.py) | Terminal demo: loads the supervised, ITS and ITS+ models for one seed, classifies the test set, prints the comparison, the ITS+ confusion matrix and ten example predictions. Trains nothing | printed output |
| [run_all.py](run_all.py) | Runs the whole pipeline in order; flags `--seeds`, `--fractions`, `--ablation`, `--reference` | – |

## 3. Inside `ssl_core.py`

| Lines | Part | What it does |
|---|---|---|
| [26](ssl_core.py#L26) | `Log` | Prints a line and appends it to a log file at once |
| [41](ssl_core.py#L41), [45](ssl_core.py#L45) | `run_name`, `run_dir` | Run names such as `its_plus_f010_s0` and their folders in `outputs/runs/` |
| [58](ssl_core.py#L58) | `Data` | Loads all images into memory and holds the row numbers of the labelled set, unlabelled pool, validation and test sets for one seed and label fraction |
| [95](ssl_core.py#L95) | `_affine` | Random rotation, zoom, shift, shear and flip on the GPU |
| [112](ssl_core.py#L112) | `weak_aug` | Light augmentation (Paper 1), used for labelled scans |
| [117](ssl_core.py#L117) | `strong_aug` | Heavy augmentation: stronger geometry, brightness, contrast, gamma and a Cutout square (Paper 2), used for pseudo-labelled scans in ITS+ |
| [140](ssl_core.py#L140) | `pretrained_backbone` | EfficientNet-B0 with ImageNet weights, downloaded once |
| [149](ssl_core.py#L149) | `Net` | `flat`: one 3-class head (Paper 1). `hier`: hierarchical classifier with a normal/abnormal head and a drusen/CNV head, multiplied into 3 class probabilities (Paper 2) |
| [175](ssl_core.py#L175) | `nll` | Class-weighted loss. A partial label ("abnormal, type unknown") uses −log(P(drusen) + P(CNV)), so it trains only the normal/abnormal decision |
| [185](ssl_core.py#L185) | `class_weights` | Inverse-frequency class weights from the labelled set |
| [193](ssl_core.py#L193) | `predict` | Class probabilities for a set of scans, no augmentation |
| [211](ssl_core.py#L211) | `metrics` | Accuracy, macro-F1, macro-AUC, under-grading rate, per-class precision and recall, confusion matrix |
| [227](ssl_core.py#L227) | `train` | Trains one network with Adam, learning-rate reduction and early stopping on validation macro-F1; keeps the best epoch. In ITS+ mode each batch is 20 labelled + 44 pseudo-labelled scans and the pseudo-label loss is ramped in ([line 279](ssl_core.py#L279), Paper 3) |
| [329](ssl_core.py#L329) | `supervised_teacher` | Round-0 teacher: trains on the labelled scans, or reuses an already trained one |
| [344](ssl_core.py#L344) | `window_mean` | Averages the teacher's predictions over each scan and its neighbours in the same volume |
| [360](ssl_core.py#L360) | `admit` | Turns teacher predictions into pseudo-labels: fixed or class-adaptive thresholds ([365](ssl_core.py#L365)), full / partial / reject ([376](ssl_core.py#L376)), volume-context agreement ([382](ssl_core.py#L382)) |
| [397](ssl_core.py#L397) | `audit` | Measures pseudo-label accuracy against the hidden true labels, for reporting only; never used to make a decision |
| [413](ssl_core.py#L413) | `run_its` | The iterative teacher–student loop: teacher predicts the pool, `admit` picks pseudo-labels, a new student trains, the student becomes the teacher; optional final fine-tune ([455](ssl_core.py#L455), Paper 3); saves the best round |

## 4. Where each part of the method is

| Method component | Source | Code |
|---|---|---|
| Iterative teacher–student loop | Paper 1 | [`run_its`](ssl_core.py#L413) |
| Confidence threshold for pseudo-labels | Paper 1 | [`admit`](ssl_core.py#L373) with a fixed `tau` |
| Hierarchical classifier | Paper 2 | [`Net`](ssl_core.py#L149), `kind="hier"` |
| Class-adaptive thresholds | Paper 2 | [`admit`](ssl_core.py#L365), `adaptive=True` |
| Full / partial / reject admission | Paper 2 | [`admit`](ssl_core.py#L376) and [`nll`](ssl_core.py#L175) |
| Volume-context agreement | Paper 2 (simplified) | [`window_mean`](ssl_core.py#L344), [`admit`](ssl_core.py#L382) |
| Weak / strong augmentation | Paper 2 | [`weak_aug`, `strong_aug`](ssl_core.py#L112), used in [`train`](ssl_core.py#L285) |
| Pseudo-label loss ramp-up | Paper 3 | [`train`](ssl_core.py#L279) |
| Final fine-tuning | Paper 3 | [`run_its`](ssl_core.py#L455) |
| Ablation switches | – | [`VARIANTS` in 06_its_plus.py](06_its_plus.py#L18) |

## 5. Outputs

| Path | Contents |
|---|---|
| `outputs/runs/<run>/best.pt` | Trained model (best round, best epoch) |
| `outputs/runs/<run>/log.txt` | Full training log: every round and epoch |
| `outputs/runs/<run>/history.csv` | Per-epoch loss and validation scores (supervised runs) |
| `outputs/runs/<run>/rounds.csv` | Per round: threshold, pseudo-labels admitted, their accuracy per class, partial labels, validation score |
| `outputs/metrics/<run>.json` | All test metrics for one run |
| `outputs/results.csv` | One row per run with the main test metrics |
| `outputs/unsupervised_results.csv` | Unsupervised k-means test metrics per seed |
| `outputs/*.txt` | Reports of steps 01, 02, 03, 07 and 10 |
| `outputs/*.png` | Figures |

Run names: `<method>_f<label %>_s<seed>`, for example `its_plus_f010_s1` = full ITS+, 10% labels, seed 1.
Methods: `supervised`, `supervised_hier` (round-0 teacher of ITS+), `its`, `its_plus_A`, `its_plus_B`, `its_plus_C`, `its_plus`.
