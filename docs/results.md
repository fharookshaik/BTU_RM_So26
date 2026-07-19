# Results — Music Structure Analysis with SpecTNT

**Model:** SpecTNT(dim=96, n_blocks=5) — 1,193,768 params
**Data:** 912 songs from HarmonixSet, 4-fold cross-validation
**Training:** batch=16, grad_accum=8 (effective 128), max 100 epochs,
cosine LR schedule, patience=2, gradient clip norm=5.0
**Hardware:** NVIDIA RTX 2000 Ada (CUDA, 8GB VRAM)

---

## 1. Training

### 1.1 Variant A — No CTL

| Fold | Best Val Loss | Best Epoch | Epochs Trained |
|------|:-------------:|:----------:|:--------------:|
| 1    | 0.5148        | 9           | 12             |
| 2    | 0.5157        | 8           | 10             |
| 3    | 0.5287        | 7           | 9              |
| 4    | 0.5266        | 8           | 10             |
| **Mean ± Std** | **0.5215 ± 0.0058** | **8.0** | **10.25** |

### 1.2 Variant B — With CTL (blank=7)

| Fold | Best Val Loss | Best Epoch | Epochs Trained |
|------|:-------------:|:----------:|:--------------:|
| 1    | 0.6909        | 9           | 11             |
| 2    | 0.7042        | 8           | 10             |
| 3    | 0.6945        | 10          | 12             |
| 4    | 0.7065        | 8           | 10             |
| **Mean ± Std** | **0.6990 ± 0.0061** | **8.75** | **10.75** |

---

## 2. Evaluation — 4-Fold Cross-Validation

### 2.1 Variant Comparison

| Metric | Variant A (No CTL) | Variant B (With CTL) | Δ |
|--------|:------------------:|:--------------------:|:-:|
| HR@0.5 ↑ | **0.2767** | 0.2748 | −0.0019 |
| ACC ↑ | **0.4932** | 0.4577 | −0.0355 |
| PWF ↑ | **0.5544** | 0.5344 | −0.0200 |
| Sf ↑ | **0.2325** | 0.1970 | −0.0355 |
| CHR@0.5 ↑ | 0.1576 | **0.1604** | +0.0028 |
| CF1 ↑ | **0.6805** | 0.6574 | −0.0231 |
| Macro F1 ↑ | **0.1497** | 0.1422 | −0.0075 |

### 2.2 Effect of Viterbi Smoothing

**Variant A (No CTL):**

| Metric | Baseline | Smoothed | Δ |
|--------|:--------:|:---------:|:-:|
| HR@0.5 | **0.2767** | 0.2767 | 0.0000 |
| ACC | **0.4932** | 0.4896 | −0.0036 |
| PWF | **0.5544** | 0.5491 | −0.0053 |
| Sf | **0.2325** | 0.2082 | −0.0243 |
| CHR@0.5 | **0.1576** | 0.1518 | −0.0058 |
| CF1 | **0.6805** | 0.6795 | −0.0009 |
| Macro F1 | **0.1497** | 0.1420 | −0.0077 |

**Variant B (With CTL):**

| Metric | Baseline | Smoothed | Δ |
|--------|:--------:|:---------:|:-:|
| HR@0.5 | **0.2748** | 0.2748 | 0.0000 |
| ACC | 0.4577 | **0.4580** | +0.0004 |
| PWF | **0.5344** | 0.5212 | −0.0131 |
| Sf | **0.1970** | 0.1674 | −0.0295 |
| CHR@0.5 | **0.1604** | 0.1560 | −0.0044 |
| CF1 | **0.6574** | 0.6515 | −0.0059 |
| Macro F1 | **0.1422** | 0.1352 | −0.0070 |

---

## 3. Single-Song Analysis (0024_billionaire, Fold 2)

| Variant | HR@0.5 | ACC | PWF | Sf | CHR@0.5 | CF1 | MF1 | #Seg |
|---------|:------:|:---:|:---:|:--:|:-------:|:---:|:---:|:----:|
| Base | 0.2667 | 0.4564 | 0.6258 | 0.1736 | 0.1786 | 0.6397 | 0.1079 | 50 |
| Base+Viterbi | 0.2667 | 0.4560 | 0.5358 | 0.0594 | 0.1739 | 0.5575 | 0.1283 | 50 |
| CTL | 0.2540 | 0.5464 | 0.5244 | 0.0808 | 0.1455 | 0.5480 | 0.1581 | 53 |
| CTL+Viterbi | 0.2540 | 0.5864 | 0.5160 | 0.0981 | 0.1667 | 0.5470 | 0.1720 | 53 |
| **Ground Truth** | — | — | — | — | — | — | — | **9** |

GT: 9 segments, median duration 23.4s
Base: 50 segments, median 3.5s, 5.6x over-segmentation
CTL: 53 segments, median 3.2s, 5.9x over-segmentation

---

## 4. Summary of Findings

- **Base model outperforms CTL** on most metrics: ACC (+3.6pp), PWF (+2.0pp), Sf (+3.6pp), CF1 (+2.3pp). CTL only slightly improves CHR@0.5 (+0.3pp).
- **Boundary detection is the weakest link**: HR@0.5 ≈ 0.28 means ~72% of boundaries are missed or misaligned. CHR@0.5 ≈ 0.16 shows chorus boundaries are even harder.
- **Chorus discrimination is relatively strong**: CF1 ≈ 0.68 (binary chorus vs non-chorus) far exceeds PWF ≈ 0.55 (7-class), indicating the model learns chorus features well but struggles with fine-grained multi-class segmentation.
- **Viterbi smoothing consistently hurts** — it degrades Sf, PWF, CF1, and Macro F1 on both variants. The data-driven transition matrix may not align well with the 7-class taxonomy, or the self-loop prior (50.0) is too aggressive.
- **Over-segmentation is the dominant failure mode** — single-song analysis shows the model produces 5.6–5.9x more segments than ground truth, breaking long sections into short fragments.
- **Macro F1 < 0.15** reveals severe class imbalance issues — the model rarely predicts minority classes (bridge, inst, silence), reflected in the large gap between ACC (~0.49) and Macro F1 (~0.15).
- **Fast convergence with early stopping** — all folds converge within 9–12 epochs (patience=2), suggesting the model may benefit from longer training, larger batches, or stronger augmentation.
