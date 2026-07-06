# SpecTNT + CTL Loss — Implementation Plan

## Goal
Reimplement the SpecTNT model with and without Connectionist Temporal Localization (CTL) loss from Wang et al. 2022 ("To Catch A Chorus, Verse, Intro, or Anything Else") and evaluate on the Harmonix Set (912 pop tracks).

## Data Overview

| Asset | Count | Details |
|---|---|---|
| `data/harmonixset/melspecs/*-mel.npy` | 912 | Precomputed mel spectrograms. Shape: (80, T). SR=22050, hop=1024, n_mels=80. |
| `data/harmonixset/dataset/segments/*.txt` | 912 | Segment boundaries + raw labels. |
| `data/harmonixset/dataset/metadata.csv` | 912 | Song metadata (duration, BPM, genre, etc.). |
| `data/harmonixset/audio/*.wav` | 840 | Downloaded audio (incomplete). **Not used.** |

All 912 metadata entries have matching melspecs and segment files.

## Architecture (Paper §3.3)

```
Input: (80, T) mel spectrogram @ ~21.5 fps
    ↓
ResNet front-end (2D conv, 3×3 kernels) + temporal downsampling → ~5.2 fps
    ↓  feature maps × 5 SpecTNT blocks
╔═══════════════════════════════════════════╗
║  SpecTNT Block × 5 (96 feat maps each)   ║
║  ┌──────────────────────────────┐         ║
║  │ Spectral Encoder             │         ║
║  │   Multi-head Self-Attn (4 hd)│         ║
║  │   FFN                         │         ║
║  └──────────┬───────────────────┘         ║
║             ↓ (FCTs across time)          ║
║  ┌──────────────────────────────┐         ║
║  │ Temporal Encoder             │         ║
║  │   Multi-head Self-Attn (8 hd)│         ║
║  │   FFN                         │         ║
║  └──────────────────────────────┘         ║
╚═══════════════════════════════════════════╝
    ↓
Linear projection → (T, 8) outputs
    ↓
[boundary curve] + [7 function curves: intro, verse, chorus, bridge, outro, inst, silence]
```

## Project Structure

```
btu-rm-sose26/
├── utils/
│   ├── __init__.py
│   ├── label_conversion.py   # Algorithm 1: raw label → 7-class taxonomy
│   ├── target_generation.py  # Build boundary + function activation curves + token sequences
│   ├── dataset.py            # PyTorch Dataset: load melspecs, enumerate 24s chunks
│   ├── augmentations.py      # torchaudio augmentations (noise, gain, filters)
│   ├── spectnt.py            # ResNet front-end + SpecTNTBlock + full model
│   ├── losses.py             # Weighted BCE, CTL loss
│   ├── postprocessing.py     # Peak-picking + segment-level argmax
│   └── metrics.py            # mir_eval wrappers (HR.5F, ACC, PWF, Sf, CHR.5F, CFI)
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Explore dataset, visualize melspecs, test label conversion
│   ├── 02_target_generation.ipynb      # Build & visualize activation curves, test CTL token seqs
│   ├── 03_model_definition.ipynb       # Define SpecTNT, inspect architecture, test forward pass
│   ├── 04_training.ipynb               # Train SpecTNT (no CTL) + SpecTNT (with CTL)
│   ├── 05_evaluation.ipynb             # Load checkpoints, post-process, compute metrics
│   └── 06_results_visualization.ipynb  # Activation curve plots, paper-style figures
├── data/harmonixset/
├── pyproject.toml
└── plan.md
```

**Key principle**: Core logic lives in `utils/*.py` — notebooks import from these and handle orchestration, visualization, and results analysis.

## Implementation Phases

### Phase 1: Project Setup
- `uv add librosa mir_eval torchaudio tqdm matplotlib ipykernel`
- Create `utils/` and `notebooks/` directories

### Phase 2: Data Utilities (`utils/`)

**`label_conversion.py`** — Algorithm 1 from paper (§2)
- Substring matching rules: `pre-chorus→verse`, `refrain→chorus`, `out/coda/ending→outro`, `break/interlude/solo→inst`, etc.
- `convert_label(raw: str) -> str` — returns one of 7 classes or "end"
- `convert_segments(segments_df) -> list` — apply conversion to all segments in a song

**`target_generation.py`** — Activation curve creation (§3.1)
- **Boundary curve**: Binary pulses of 0.6s width at each segment boundary
- **Function curves**: Binary masks per class, smoothed with 2s Hann window (1s ramp-up, 1s ramp-down)
- All curves at ~5.2 fps resolution (downsampled from native 21.5 fps melspec via averaging)
- **Token sequence** (for CTL): Sequence of converted section labels from annotations

**`dataset.py`** — PyTorch Dataset
- `HarmonixDataset` loads melspecs + segment annotations from file IDs
- Enumerates all 24s chunks with 3s hop across all songs (§4.1)
- Random uniform sampling from the chunk pool
- Returns: `(chunk_melspec_tensor, boundary_target, function_targets_7, token_sequence)`
- Optional `fold` argument for 4-fold CV: filters songs by fold assignment

**`augmentations.py`** (§4.1)
- Random noise, gain, HP/LP filtering via `torchaudio` transforms
- Applied on-the-fly per chunk during training

### Phase 3: Model Utilities (`utils/`)

**`spectnt.py`** — SpecTNT implementation (§3.3)

- **ResNet front-end**: 2D conv layers (kernel_size=3), temporal downsampling via stride to ~5.2 fps
- **SpecTNTBlock**: SpectralEncoder (96 dim, 4 heads) → TemporalEncoder (96 dim, 8 heads) with residual + LayerNorm
- **SpecTNT**: ResNet → 5× SpecTNTBlock → Linear(96, 8)
- `forward(x) -> (boundary_logits, function_logits)` where both are (B, T, ...)

### Phase 4: Training Utilities (`utils/`)

**`losses.py`** (§3.5)
- `weighted_bce_loss` — BCE with per-class weights for boundary sparsity
- `function_bce_loss` — BCE over 7 function classes
- `combined_loss` = `0.9 * ℓ_boundary + 0.1 * ℓ_function`
- `CTLLoss` — CTC-based loss over token sequences, encourages temporal coherence (§3.5)

**`postprocessing.py`** (§3.4)
- `peak_picking(boundary_curve)` — detect boundary timestamps (Ullrich et al. 2014)
- `segment_labeling(boundaries, function_curves)` — assign function to each segment by max avg probability

**`metrics.py`** (§4.2)
- Wrappers around `mir_eval` for HR.5F, ACC, PWF, Sf, CHR.5F, CFI
- `evaluate_all` — compute all 6 metrics given predictions + ground truth

### Phase 5: Notebooks

**`01_data_exploration.ipynb`**
- Load metadata, count songs/genres
- Load a melspec, plot mel spectrogram
- Load segment annotations, inspect raw labels
- Test `label_conversion` on raw labels, show distribution of converted labels
- Overlay segment boundaries on mel spectrogram

**`02_target_generation.ipynb`**
- Build boundary activation curves, visualize
- Build function activation curves, visualize per-class
- Test temporal downsampling (21.5 fps → 5.2 fps)
- Generate CTL token sequences, visualize
- Sanity check: verify target curve shapes make sense

**`03_model_definition.ipynb`**
- Import SpecTNT from utils
- Print model architecture, parameter count
- Create dummy batch, run forward pass
- Verify output shapes: (B, T', 1) for boundary, (B, T', 7) for functions
- Test loss functions on dummy data

**`04_training.ipynb`**
- Create HarmonixDataset + DataLoader
- 4-fold cross-validation setup:
  - Stratify by song (random 75/25, repeat 4×)
  - For each fold: train on 3 folds, validate on held-out fold
- **Variant A**: Train SpecTNT without CTL loss
- **Variant B**: Train SpecTNT with CTL loss
- Training loop: Adam (lr=0.0005, wd=0.9), batch=128, 500 batches/epoch, 100 epochs, patience=2
- Save best checkpoint per fold per variant

**`05_evaluation.ipynb`**
- Load best checkpoints for both variants
- Run full inference on test folds
- Post-process each song: peak-picking → boundaries → segment labels
- Compute all 6 metrics per song, aggregate across folds
- Present results table (mean ± std across 4 folds):

| Variant | HR.5F | ACC | PWF | Sf | CHR.5F | CFI |
|---|---|---|---|---|---|---|
| SpecTNT (24s) | | | | | | |
| SpecTNT (24s, CTL) | | | | | | |

- Compare against paper Table 1 ablation results

**`06_results_visualization.ipynb`**
- Plot activation curves for selected test songs (like paper Fig. 2)
  - Top: melspec + raw boundary + function curves
  - Bottom: raw argmax labels vs post-processed labels vs ground truth
- Confusion matrix for function classification
- Per-genre breakdown of metrics (if interesting patterns emerge)

### Phase 6: Dependencies

```
# pyproject.toml additions
librosa
mir_eval
torchaudio
tqdm
matplotlib
ipykernel
```

### Phase 7: Commands

```bash
uv add librosa mir_eval torchaudio tqdm matplotlib ipykernel
uv run jupyter notebook notebooks/
```

## Key Design Decisions

1. **Time resolution**: Native melspec ~21.5 fps (hop=1024, SR=22050). ResNet front-end will downsample via strided conv/pooling to ~5.2 fps, matching paper's activation curve resolution.

2. **4-fold CV**: Since we only have HarmonixSet (no SALAMI/RWC/Isophonics), we do 4-fold CV like the paper's ablation study, but train only on HarmonixSet (paper trained on all 4 datasets).

3. **CTL loss**: Implemented as CTC loss (torch.nn.CTCLoss) over the 7 function classes + blank token. Target sequence = converted section labels. The CTL penalizes predictions that deviate from plausible label orderings.

4. **Checkpointing**: Save best model per fold based on validation combined loss. Each variant has 4 checkpoints (one per fold).
