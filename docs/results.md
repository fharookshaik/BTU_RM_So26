# Results — Music Structure Analysis with SpecTNT

**Model:** SpecTNT(dim=96, n_blocks=5) — 1,193,768 params
**Data:** 912 songs from HarmonixSet, 4-fold cross-validation
**Training:** batch=32, grad_accum=4 (effective 128), max 100 epochs,
cosine LR schedule, patience=2, gradient clip norm=5.0
**Hardware:** NVIDIA RTX 2000 Ada (CUDA, 8GB VRAM)

---

## 1. Training

### 1.1 Variant A — No CTL

| Fold | Best Val Loss | Best Epoch | Epochs Trained |
|------|:-------------:|:----------:|:--------------:|
| 1    | 0.5106        | 4           | 6              |
| 2    | 0.5180        | 5           | 7              |
| 3    | 0.5193        | 5           | 7              |
| 4    | 0.5249        | 3           | 5              |
| **Mean ± Std** | **0.5182 ± 0.0052** | — | 6.25 |

### 1.2 Variant B — With CTL (blank=7)

| Fold | Best Val Loss | Best Epoch | Epochs Trained |
|------|:-------------:|:----------:|:--------------:|
| 1    | 0.6903        | 5           | 7              |
| 2    | 0.6930        | 5           | 7              |
| 3    | 0.7084        | 3           | 5              |
| 4    | 0.7003        | 4           | 6              |
| **Mean ± Std** | **0.6980 ± 0.0071** | — | 6.25 |

---

## 2. Evaluation — 4-Fold Cross-Validation

### 2.1 Variant Comparison (All 6 Metrics)

| Metric | Variant A (No CTL) | Variant B (With CTL) | Δ |
|--------|:------------------:|:--------------------:|:-:|
| HR@0.5 ↑ | **0.2813** | 0.2763 | −0.0049 |
| ACC ↑ | **0.3948** | 0.3937 | −0.0011 |
| PWF ↑ | **0.4923** | 0.4859 | −0.0064 |
| Sf ↑ | 0.0536 | **0.0952** | **+0.0416** |
| CHR@0.5 ↑ | **0.1703** | 0.1620 | −0.0083 |
| CF1 ↑ | **0.7035** | 0.6775 | −0.0259 |
| Macro F1 ↑ | 0.0904 | **0.1172** | **+0.0268** |

### 2.2 Effect of Viterbi Smoothing

**Variant A (No CTL):**

| Metric | Baseline | Smoothed | Δ |
|--------|:--------:|:---------:|:-:|
| HR@0.5 | 0.2813 | 0.2813 | 0.0000 |
| ACC | **0.3948** | 0.3939 | −0.0009 |
| PWF | 0.4923 | **0.4929** | **+0.0006** |
| Sf | 0.0536 | **0.0756** | **+0.0220** |
| CHR@0.5 | 0.1703 | **0.1708** | **+0.0005** |
| CF1 | **0.7035** | 0.6939 | −0.0096 |
| Macro F1 | 0.0904 | **0.0955** | **+0.0051** |

**Variant B (With CTL):**

| Metric | Baseline | Smoothed | Δ |
|--------|:--------:|:---------:|:-:|
| HR@0.5 | 0.2763 | 0.2763 | 0.0000 |
| ACC | **0.3937** | 0.3859 | −0.0078 |
| PWF | **0.4859** | 0.4720 | −0.0139 |
| Sf | **0.0952** | 0.0898 | −0.0054 |
| CHR@0.5 | **0.1620** | 0.1616 | −0.0004 |
| CF1 | **0.6775** | 0.6487 | −0.0288 |
| Macro F1 | **0.1172** | 0.1073 | −0.0099 |

---

## 3. Single-Song Analysis (0095_firework, Variant A Fold 1)

| Metric | Value |
|--------|:-----:|
| HR@0.5 | 0.3846 |
| PWF | 0.5931 |
| SF | 0.3028 |
| Accuracy | 0.3924 |
| Macro F1 | 0.0837 |

GT: 7 segments, 29.5s median duration
Predicted: 17 segments, 8.5s median duration (2.4× over-segmentation)

---

## 4. Summary of Findings

- CTL improves Sf by 78% relative (+0.0416) and Macro F1 by 30% (+0.0268), but slightly hurts boundary metrics (HR@0.5, CHR@0.5) and pairwise metrics (PWF, CF1)
- Base model (no CTL) achieves better boundary detection (HR@0.5 0.2813 vs 0.2763) and chorus boundary detection (CHR@0.5 0.1703 vs 0.1620)
- CTL model achieves better function labeling (Sf 0.0952 vs 0.0536, Macro F1 0.1172 vs 0.0904)
- CF1 (chorus pairwise F-measure) is substantially higher than PWF (0.70 vs 0.49), indicating chorus/non-chorus discrimination is easier than 7-class segmentation
- Both variants over-segment (~2.4×) and rarely predict minority functional classes (Macro F1 < 0.12)
