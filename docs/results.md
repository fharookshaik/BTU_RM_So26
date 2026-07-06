# Results — Music Structure Analysis with SpecTNT

**Model:** SpecTNT(dim=96, n_blocks=5) — 1,193,768 params
**Data:** 912 songs from HarmonixSet, 4-fold cross-validation
**Training:** batch=32, grad_accum=4 (effective 128), max 100 epochs,
cosine LR schedule, patience=2, gradient clip norm=5.0
**Hardware:** Intel Arc 140V (XPU, 8GB VRAM), ~12–14 min/epoch

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

### 2.1 Variant Comparison

| Metric | Variant A (No CTL) | Variant B (With CTL) | Δ |
|--------|:------------------:|:--------------------:|:-:|
| HR@0.5 ↑ | **0.2812** | 0.2762 | −0.0050 |
| PWF ↑ | **0.4923** | 0.4859 | −0.0064 |
| SF ↑ | 0.0534 | **0.0950** | **+0.0416** |
| Accuracy ↑ | **0.3947** | 0.3936 | −0.0011 |

### 2.2 Effect of Viterbi Smoothing

**Variant A (No CTL):**

| Metric | Baseline | Smoothed | Δ |
|--------|:--------:|:---------:|:-:|
| HR@0.5 | 0.2812 | 0.2812 | 0.0000 |
| PWF | **0.4923** | 0.4929 | +0.0006 |
| SF | 0.0534 | **0.0754** | **+0.0220** |
| Acc | **0.3947** | 0.3938 | −0.0009 |

**Variant B (With CTL):**

| Metric | Baseline | Smoothed | Δ |
|--------|:--------:|:---------:|:-:|
| HR@0.5 | 0.2762 | 0.2762 | 0.0000 |
| PWF | **0.4859** | 0.4719 | −0.0139 |
| SF | **0.0950** | 0.0897 | −0.0054 |
| Acc | **0.3936** | 0.3860 | −0.0077 |

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

- CTL improves SF by 78% relative (+0.0416) but slightly hurts HR@0.5 and PWF
- Base model converges to lower validation loss (0.5182 vs 0.6980) — CTL adds a harder auxiliary task
- Viterbi smoothing benefits Variant A (SF +0.0220) but degrades Variant B across non-boundary metrics
- Accuracy is similar across variants (~39.4%) and weakly affected by smoothing
- Both variants over-segment (~2.4×) and rarely predict minority functional classes (Macro F1 = 0.08)
