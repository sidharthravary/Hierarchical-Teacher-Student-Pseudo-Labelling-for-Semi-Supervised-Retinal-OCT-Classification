# Hierarchical Teacher–Student Pseudo-Labelling for Semi-Supervised Retinal OCT Classification

**Machine Learning course project – final review | Group 7**

| Team member | UEN |
|---|---|
| Sidharth Ravary | 2024UG000252 |
| Shivani Sharath | 2024UG000242 |
| Nithyashree | 2024UG000277 |

**Task.** Classify a retinal OCT B-scan as **NORMAL**, **DRUSEN** or **CNV** when only **10% of the training patients
are labelled** (940 scans) and 8,933 scans are unlabelled. Semi-supervised method: iterative teacher–student
pseudo-labelling (Paper 1) extended with a hierarchical classifier, class-adaptive partial-label admission,
neighbouring-scan agreement (Paper 2), a pseudo-label loss ramp-up and a final fine-tune (Paper 3).

**Result (test set, 4,326 scans from 109 unseen patients, mean ± SD over 3 seeds):** ITS+ (ours) **85.90 ± 2.84%**
accuracy, 85.64% macro-F1, 94.79% macro-AUC, vs ITS (Paper 1) 82.15%, supervised 80.64%, unsupervised k-means 50.62%.

---

## 1. Links: code, dataset and model checkpoints

All links are **public** (no login or access request needed). The submission zip holds the code, README, report,
presentation, screenshots and results; the dataset and the trained checkpoints are too large for the 35 MB upload
limit and are downloaded from these links.

| What | Link |
|---|---|
| Code repository (GitHub) | https://github.com/sidharthravary/Hierarchical-Teacher-Student-Pseudo-Labelling-for-Semi-Supervised-Retinal-OCT-Classification |
| Full project package, code + dataset + all checkpoints (Google Drive, 972 MB zip) | https://drive.google.com/file/d/14JeVbn09mX8-dJMCaF1Y3CdlKKfvavqz/view?usp=sharing |
| GitHub release v1.0 (dataset + checkpoints) | https://github.com/sidharthravary/Hierarchical-Teacher-Student-Pseudo-Labelling-for-Semi-Supervised-Retinal-OCT-Classification/releases/tag/v1.0 |
| Preprocessed dataset `images.npy` (805 MB) | https://github.com/sidharthravary/Hierarchical-Teacher-Student-Pseudo-Labelling-for-Semi-Supervised-Retinal-OCT-Classification/releases/download/v1.0/images.npy |
| All 21 trained checkpoints `checkpoints.zip` (302 MB) | https://github.com/sidharthravary/Hierarchical-Teacher-Student-Pseudo-Labelling-for-Semi-Supervised-Retinal-OCT-Classification/releases/download/v1.0/checkpoints.zip |
| Original raw dataset NEH_UT_2021 (Mendeley Data) | https://data.mendeley.com/datasets/8kt969dhx6/1 (doi:10.17632/8kt969dhx6.1) |

**Where to put the downloaded files** (needed before running the code):

1. `images.npy` → `OCT-Semi-Supervised/data/images.npy`
2. `checkpoints.zip` → unzip it inside `OCT-Semi-Supervised/` (it creates `outputs/runs/<run>/best.pt`)

Or download the Google Drive zip, which already contains everything in place. The ImageNet weights of EfficientNet-B0
(20 MB) are downloaded automatically by the code on first use. The raw Mendeley download can be converted to
`images.npy` with `python 00_build_images.py <dataset folder or zip>`.

## 2. Contents

```
OCT-Semi-Supervised/
├── README.md                 this file
├── requirements.txt          Python dependencies (tested versions)
├── config.yaml               all paths and hyper-parameters
├── *.py                      source code (section 5)
├── data/                     metadata CSVs; put images.npy here (section 1)
├── outputs/
│   ├── runs/<run>/           training logs (log.txt, history.csv, rounds.csv); put best.pt checkpoints here (section 1)
│   ├── live_demo/            log and curves written by live_demo.py
│   ├── metrics/              test metrics of every checkpoint (JSON)
│   ├── results.csv           one row per checkpoint
│   └── *.png, *.txt          figures and reports of every pipeline step
├── report/ML_Project_Report.pdf                      project report (IEEE format, 5 pages + references)
├── presentation/ML_Project_Presentation.pptx / .pdf  presentation slides (PPTX and PDF)
├── screenshots/              Command Prompt screenshots: live training, checkpoints, evaluation, demo
├── RESULTS.md, COMPARISON.md result tables; CODE_GUIDE.md: every file and function with line links
└── results.ipynb             executed notebook (outputs embedded)
```

The three papers used are cited in the report (references [1]–[3]).

## 3. Setup

Requirements: Python 3.10 or newer; an NVIDIA GPU is recommended (CPU works for the demos but training is slow).
Tested on Windows 11, Python 3.14.3, PyTorch 2.14 (CUDA 12.6), NVIDIA RTX 3050 4 GB.

**Windows (Command Prompt)**

```bat
cd OCT-Semi-Supervised
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Linux / macOS**

```bash
cd OCT-Semi-Supervised
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check the GPU: `python -c "import torch; print(torch.cuda.is_available())"`

## 4. How to run

| Step | Command | Time (RTX 3050) | What it shows |
|---|---|---|---|
| **Live training demo** | `python live_demo.py --epochs 3` | ~2 min | Teacher pseudo-labels the unlabelled pool, a student is **trained live for 3 epochs** (loss and validation accuracy / macro-F1 printed every epoch), the **best epoch is saved** as `outputs/live_demo/best_model.pt`, **reloaded from disk and benchmarked on the unseen test set** (accuracy, macro-F1, AUC, per-class precision / recall, confusion matrix), and compared with the stored best ITS+ checkpoint. Curves: `outputs/live_demo/training_curves.png` |
| **Benchmark the stored best checkpoints** | `python demo.py --seed 0` | ~1 min | Loads the best checkpoints of supervised, ITS and ITS+ and prints test results, the confusion matrix and example predictions (`--seed 1` / `--seed 2` for the other runs) |
| **Evaluate every checkpoint** | `python 07_evaluate.py` | ~5 min | Test metrics + 95% case-bootstrap confidence intervals for all 21 checkpoints → `outputs/results.csv`, `outputs/07_evaluation.txt` |
| **Figures** | `python 08_report.py`, `python 11_comparison.py`, `python 12_extra_figures.py` | ~2 min | Result tables, confusion matrices, training curves, ROC curves, cluster plot |
| **Full pipeline** | `python run_all.py --seeds 0 1 2 --ablation` | skips finished models | Runs 01 → 12 in order; existing checkpoints are reused. Delete `outputs/runs` to retrain everything from scratch (about a day on a laptop GPU) |
| **Train one model** | `python 04_supervised.py --seed 0`, `python 05_its_faithful.py --seed 0`, `python 06_its_plus.py --variant D --seed 0` | 0.5–1.5 h each | Supervised baseline, ITS (Paper 1), ITS+ (ours); `--variant A/B/C` = ablation |

Model selection uses only the validation set (best epoch and best round by validation macro-F1); the test set is used
once, to benchmark the saved best checkpoints.

## 5. Modules

| File | Description |
|---|---|
| `config.yaml` | Paths, split fractions, label fractions, seeds, training and method hyper-parameters |
| `common.py` | Loads the configuration, resolves paths, report writer |
| `ssl_core.py` | Core library: `Data` (image cache and splits), `weak_aug` / `strong_aug` (GPU augmentation), `Net` (EfficientNet-B0 with flat or hierarchical heads), `nll` (class-weighted loss with partial labels), `train` (training loop, early stopping, best-epoch selection), `admit` (pseudo-label admission: thresholds, full / partial / reject, volume agreement), `run_its` (teacher–student loop), `metrics` |
| `00_build_images.py` | Builds `data/images.npy` from the raw dataset (grayscale, scale-bar mask, retina crop, 224 x 224) |
| `01_data_audit.py` | Dataset audit: per-scan vs folder labels, duplicates, outliers, image-cache alignment |
| `02_make_splits.py` | Patient-level train / validation / test split, nested labelled subsets per seed, leakage checks |
| `03_data_check.py` | Set sizes, augmentation examples, GPU speed test |
| `04_supervised.py` | Supervised baseline (labelled scans only) |
| `05_its_faithful.py` | ITS as published in Paper 1 |
| `06_its_plus.py` | ITS+ (ours); `--variant A/B/C/D` for the ablation |
| `07_evaluate.py` | Test-set evaluation of every checkpoint with confidence intervals |
| `08_report.py` | `RESULTS.md` and result figures |
| `09_make_notebook.py` | Builds and executes `results.ipynb` |
| `10_unsupervised.py` | Unsupervised reference: PCA + k-means on ImageNet features |
| `11_comparison.py` | `COMPARISON.md`: all models, seeds, classes, ablation, published results |
| `12_extra_figures.py` | Training curves, ROC curves, per-class scores, cluster plot, method diagram |
| `live_demo.py` | Live training demo with best-checkpoint saving and test benchmarking |
| `demo.py` | Benchmarks the stored best checkpoints in the terminal |
| `run_all.py` | Runs the whole pipeline in order |

`CODE_GUIDE.md` maps every function and every method component to its line in the code.

## 6. Checkpoints

| Checkpoint | Model |
|---|---|
| `outputs/runs/supervised_f010_s<seed>/best.pt` | Supervised, flat head |
| `outputs/runs/supervised_hier_f010_s<seed>/best.pt` | Supervised, hierarchical heads (round-0 teacher of ITS+) |
| `outputs/runs/its_f010_s<seed>/best.pt` | ITS (Paper 1), best round |
| `outputs/runs/its_plus_A/B/C_f010_s<seed>/best.pt` | Ablation variants |
| `outputs/runs/its_plus_f010_s<seed>/best.pt` | **ITS+ (ours), final model** |
| `outputs/live_demo/best_model.pt` | Best epoch of the live demo (overwritten on each run) |

Each checkpoint stores the network type, weights, validation metrics and the selected round.

## 7. Results summary

| Learning type | Model | Accuracy | Macro-F1 | Macro-AUC | Under-grading rate ↓ |
|---|---|---|---|---|---|
| Unsupervised | k-means on ImageNet features | 50.62 ± 0.01 | 49.27 ± 0.03 | 69.41 ± 0.01 | 24.98 ± 0.10 |
| Supervised | EfficientNet-B0 | 80.64 ± 0.39 | 79.31 ± 1.14 | 91.29 ± 0.58 | 23.94 ± 5.35 |
| Semi-supervised | ITS (Paper 1) | 82.15 ± 0.18 | 81.03 ± 1.86 | 93.53 ± 0.54 | 20.84 ± 10.42 |
| Semi-supervised | **ITS+ (ours)** | **85.90 ± 2.84** | **85.64 ± 3.37** | **94.79 ± 1.59** | **14.64 ± 5.33** |

Full tables: `RESULTS.md` and `COMPARISON.md`. Discussion, error analysis and limitations: `report/ML_Project_Report.pdf`.

## 8. Screenshots

Screen captures of Windows Command Prompt after running each command in the project folder:

| File | Command | Shows |
|---|---|---|
| `01_live_training_best_checkpoint_test.png` | `python live_demo.py --epochs 3` | Live training for 3 epochs, best checkpoint saved, reloaded and benchmarked on the test set |
| `02_saved_checkpoints.png` | `dir /s /b outputs\*.pt`, `dir outputs\live_demo` | All saved checkpoints on disk, including the one just written by the live demo |
| `03_training_supervised_seed0.png` | `type outputs\runs\supervised_f010_s0\log.txt` | Training log of the supervised baseline |
| `04`–`06_training_its_plus_*.png` | log of `its_plus_f010_s0`, in three pages | Every round and epoch of ITS+, ending with the saved best round |
| `07_training_its_paper1_part1`–`3.png` | log of `its_f010_s0`, in three pages | Every round and epoch of ITS (Paper 1) |
| `08_test_evaluation_all_checkpoints.png` | `python 07_evaluate.py` | Test results of all 21 checkpoints with 95% confidence intervals |
| `09_unsupervised_baseline.png` | `python 10_unsupervised.py` | k-means baseline |
| `10_demo_best_checkpoints.png` | `python demo.py --seed 0` | Stored best checkpoints benchmarked on the test set |

The training logs (03–07) were written while the models trained, which took several hours each; the screenshots show
those log files printed in Command Prompt.

## 9. References

1. A. Alizadeh et al., *Investigating the impact of semi-supervised learning methods to improve the quality of diagnosis of retinal diseases from OCT images*, Diagnostics 16 (2026) 656.
2. M. A. H. Akash et al., *TRIAGE: Risk-controlled pseudo-label admission for annotation-efficient semi-supervised retinal OCT classification*, arXiv:2608.14321 (2026).
3. Z. Tan et al., *OCT retinopathy classification via a semi-supervised pseudo-label sub-domain adaptation and fine-tuning method*, Mathematics 12 (2024) 347.
4. S. Sotoudeh-Paima, F. Hajizadeh, H. Soltanian-Zadeh, *Labeled retinal OCT dataset for classification of normal, drusen, and CNV cases*, Mendeley Data (2021), doi:10.17632/8kt969dhx6.1.
