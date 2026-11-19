# Implementation plan: ITS+ (improved iterative teacher-student) on NEH_UT_2021

## 1. Dataset cross-check

- The shared dataset is **not Kermany**. It is **NEH_UT_2021 (Noor Eye Hospital, Tehran)**: 16,822 OCT B-scans, 3 classes (Normal / Drusen / CNV).
- Paper 1, Alizadeh et al., *Diagnostics* 2026 (ITS, the foundation): Noor, 441 patients, EfficientNet-B0.
- Paper 2, Akash et al., arXiv 2608.14321 (TRIAGE): the same Noor data, EfficientNet-B0, plus OCT-C8 for transfer.
- Paper 3, Tan et al., *Mathematics* 2024 (TLSDA): three datasets. One is an older Noor set (AMD/DME/Normal), one is a Kermany subset. It does not use NEH_UT_2021, so it is a source of method ideas only, not of comparison numbers.
- Kermany appears only in the attached plan's separate unsupervised track.

## 2. What is already in place

- `01_data_audit.py`, `02_make_splits.py`, `config.yaml`, and the saved splits. These are kept and never regenerated.
- 16,803 verified images, 441 cases, split by case: train 259 cases / 9,873 images, val 66 / 2,396, test 109 / 4,326. 7 layout-outlier cases are excluded.
- Labelled subsets (nested, 3 seeds): 5% = 14 cases (~540 images), **10% = 25 cases (~960 images)**, 20% = 52 cases (~1,940 images). The rest of train is the unlabelled pool (~8,900 images at 10%).
- Target is the per-B-scan `label`, because 24.8% of scans differ from their folder class.
- Images come from the 224x224 cache of the sibling OCT project. The manifest has `volume_id` and `bscan_index`, which the volume-context step needs.
- Environment: torch 2.14 + torchvision 0.29 on an RTX 3050 4 GB. `timm` is not installed, so the backbone is torchvision's `efficientnet_b0`.
- The saved split is 60/15/25. The attached plan suggested 65/15/20. I keep the saved one.

## 3. The three deliverables

1. **Supervised (10%)**: EfficientNet-B0 trained on the 10% labelled cases only.
2. **Semi-supervised (10% + 90%)**: the faithful ITS replication, then ITS+ (this plan).
3. **Unsupervised**: stays as the separate track in the attached plan. Not changed here.

## 4. Foundation: ITS as published (Paper 1)

1. Train a teacher (EfficientNet-B0, ImageNet weights, all layers fine-tuned) on the labelled set.
2. The teacher predicts every unlabelled scan.
3. Scans whose top probability is at or above a threshold tau get a hard pseudo-label. The rest are set aside.
4. A student trains on labelled + pseudo-labelled scans with plain cross-entropy.
5. The student becomes the next teacher. Repeat, up to 4 rounds.

Published settings: Adam, lr 1e-3, batch 64, cross-entropy, ReduceLROnPlateau (factor 0.1, patience 10), early stopping (patience 15), up to 100 epochs, tau in {0, 0.90, 0.95, 0.99}, augmentation = random crop, horizontal flip, small rotation, affine.

Known weaknesses, which the additions target:
- One global threshold. Drusen, the hardest class, is admitted least, so the student sees a skewed pseudo-set.
- All-or-nothing admission. A scan that is clearly diseased but unclear between Drusen and CNV is thrown away.
- Each B-scan is judged alone, although neighbours in the same volume carry evidence.
- Student and teacher see the same easy views, so errors are reinforced (the paper itself reports that accuracy drops after round 2).

## 5. ITS+ : the model

Every addition is taken from Paper 2 or Paper 3. The teacher-to-student-to-teacher loop is unchanged.

### 5.1 Network (addition A, TRIAGE section 3.2)

- Input: 224x224 greyscale scan repeated to 3 channels, ImageNet normalisation.
- Backbone: EfficientNet-B0, ImageNet weights, fully fine-tuned, giving a 1,280-d feature vector.
- Dropout 0.2, then two linear heads instead of one 3-way softmax:
  - **Binary head**: Normal vs Abnormal.
  - **Fine head**: Drusen vs CNV, trained only on abnormal scans (masked cross-entropy).
- The 3-class output is recovered exactly:
  - p(Normal) = p_bin(Normal)
  - p(Drusen) = p_bin(Abnormal) x p_fine(Drusen)
  - p(CNV) = p_bin(Abnormal) x p_fine(CNV)
- Why: the fine head lets a scan teach "abnormal" without forcing a subtype guess.

### 5.2 One round of ITS+

**Step 1. Teacher prediction.** The teacher predicts the whole unlabelled pool on un-augmented images. The pool is re-labelled from scratch every round, so an early mistake is not locked in. (The paper does not say whether labels accumulate. This is my choice and will be stated.)

**Step 2. Volume context (addition C, TRIAGE section 3.4, simplified).** For each scan, average the teacher's probabilities over a window of 5 neighbouring B-scans (itself and 2 on each side, same `volume_id`, ordered by `bscan_index`). TRIAGE uses a small Transformer as a second teacher. I use the plain average because it needs no extra training and fits 4 GB. This is a simplification and will be stated.

**Step 3. Three-outcome admission (addition B, TRIAGE section 3.3 and its class-adaptive threshold).**
- Per-class threshold tau_c inside [0.90, 0.98]: classes the teacher is currently less sure of (in practice Drusen) get a lower threshold, classes it is sure of get a higher one.
- **Full label**: top-class confidence >= tau_c, fine-head confidence >= 0.80 if the class is abnormal, and the single-scan and volume-context predictions agree. The scan gets a hard 3-class label.
- **Partial label**: p(Abnormal) passes its threshold and both views agree it is abnormal, but the subtype is unsure or the two views disagree on it. The scan is labelled "abnormal" only.
- **Reject**: everything else. No loss.
- The volume average is used only as an agreement gate, never to overwrite a label, because labels really do change inside a volume (most CNV cases contain Drusen and Normal scans too).

**Step 4. Student training (addition D, TRIAGE weak/strong views + TLSDA Eq. 1 and 6).**
- A new student starts from ImageNet weights (not from the teacher, to limit inherited errors; the paper is silent on this).
- Labelled scans use the ITS augmentation (weak). Pseudo-labelled scans use strong augmentation: RandAugment (2 ops) + Cutout.
- Loss: `L = L_x + lambda(t) * L_u`
  - `L_x` = binary CE + masked fine CE on labelled scans.
  - `L_u` = the same CE for full labels, and `-log(p_Drusen + p_CNV)` for partial labels.
  - `lambda(t)` = 0 for the first epochs, then a linear ramp to 1, then constant. The student first fits the true labels and only then leans on pseudo-labels.
- Each batch mixes labelled and pseudo-labelled scans (1:2), with AMP, default memory format (not channels_last, which is ~14x slower on this GPU).

**Step 5. Hand-over.** Score the student on validation (macro-F1). It becomes the teacher for the next round. Run up to 4 rounds and keep the round with the best validation macro-F1.

**Step 6. Final fine-tune (TLSDA step 2).** Fine-tune the best student on the true labelled scans only, at a small learning rate (1e-4), for a few epochs. Keep it only if validation macro-F1 improves.

### 5.3 Optional extensions (only if time allows, each as its own ablation row)

- **Risk-based admission (TRIAGE section 3.3)**: replace confidence with expected clinical cost, using TRIAGE's cost matrix (calling CNV "Normal" costs 5, Drusen "Normal" costs 2), with the cut-off set on validation cases.
- **Sub-domain alignment (TLSDA Eq. 2)**: an LMMD loss that pulls labelled and unlabelled features of the same class together.
- An autoencoder is left out: none of the three papers uses one, so it would not meet the "must come from the papers" rule.

## 6. Experiments

Main setting: 10% labels, seeds 0/1/2. Each row adds one thing to the row above.

| Row | Method | Source |
|---|---|---|
| S0 | Supervised, 10% labels | Paper 1 |
| S1 | ITS as published (tau sweep 0 / 0.90 / 0.95 / 0.99, 4 rounds) | Paper 1 |
| A | S1 + hierarchical two-head network | Paper 2 |
| B | A + class-adaptive thresholds and full / partial / reject admission | Paper 2 |
| C | B + volume-context agreement gate | Paper 2 |
| D | C + strong augmentation, ramped pseudo-loss, final fine-tune = **ITS+** | Papers 2, 3 |

Also run: S0, S1 and ITS+ at 5% and 20% (the fractions both papers publish), and a 100%-label supervised reference.

Metrics, on the untouched test set:
- Accuracy, macro-F1, per-class precision and recall, macro-AUC, confusion matrix.
- Under-grading rate (share of diseased scans called Normal), from Paper 2.
- Scan level and volume level (mean of a volume's scan probabilities).
- Mean and SD over 3 seeds, plus a bootstrap confidence interval resampled by case.

Pseudo-label audit: every round, log per class how many scans were admitted (full / partial / rejected) and how many were correct against the hidden labels. This is for analysis only and never feeds any decision.

Published numbers to place next to ours (different splits, so context, not a ranking):

| Labels | Supervised (P1) | ITS (P1) | Supervised (P2) | TRIAGE (P2) |
|---|---|---|---|---|
| 5% | 54.78 | 64.15 | 60.28 | 76.88 |
| 20% | 77.05 | 88.56 | 79.25 | 89.66 |
| 100% | 87.14 | - | 88.50 | - |

No paper publishes a 10% figure.

## 7. Build order

| # | File | What it does | Done when |
|---|---|---|---|
| 1 | `03_data.py` | Dataset over the image cache, weak and strong augmentation, labelled / unlabelled loaders | Saved grid of augmented samples looks right; one epoch timed |
| 2 | `models.py`, `engine.py` | Flat and two-head EfficientNet-B0; train / eval loops, early stopping, checkpoints | A 2-epoch smoke run trains and reloads |
| 3 | `04_supervised.py` | S0 at 5 / 10 / 20 / 100% | Baseline numbers in `results.csv` |
| 4 | `05_its_faithful.py` | S1, as published | Per-round log and pseudo-label audit |
| 5 | `06_its_plus.py` | Rows A-D behind config switches | Ablation table at 10% |
| 6 | `07_evaluate.py` | All metrics, scan and volume level, bootstrap CIs | One metrics JSON per run |
| 7 | `08_report.py` | Tables, round-by-round curves, confusion matrices, comparison with the papers | Figures and tables for the report |

Rules: every setting lives in `config.yaml`; the test set is used only in step 6; all choices (threshold range, ramp length, rounds) are tuned on validation with seed 0 and then frozen.

## 8. Compute and risks

- Not yet measured: training time with strong augmentation. Earlier measurement: ~200 images/s for plain EfficientNet-B0 training, so a student round on ~10k images is roughly 1 minute per epoch. Rough total: 60-70 trainings, on the order of 20 GPU-hours. Step 1 times an epoch before anything is launched; if it is slow, the tau sweep runs on seed 0 only.
- 10% is only 25 labelled cases, so seed-to-seed spread will be large. Differences of 1-2 points are not conclusions.
- A labelled validation set of 66 cases is more labels than the 25 training cases. This flatters every SSL method equally and will be stated in the report.
- Paper 1's own numbers are inconsistent in places (split sizes do not sum to the total; 70% beats 100%), and Paper 2 groups by 161 patient IDs where the metadata gives 441 cases. Exact replication of either is not possible.
- ITS+ may not beat ITS on every seed. The ablation table reports whatever happens, row by row.
