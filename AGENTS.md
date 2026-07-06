# BTU RM SoSe26 — Music Structure Analysis

## Package manager
- `uv sync` installs deps (not `pip`). Lockfile: `uv.lock`.
- PyTorch is pulled from `https://download.pytorch.org/whl/xpu` (Intel GPU / XPU) on Windows/Linux. Code must use `torch.xpu` APIs, **not** `torch.cuda`.

## Dataset
- `data/harmonixset/` is a vendored fork of [urinieto/harmonixset](https://github.com/urinieto/harmonixset) (912 pop tracks with beat, downbeat, and functional segment annotations).
- Actual data files are **not committed** — `.gitignore` excludes `data/harmonixset/`. Handle missing data gracefully.

## Hardware
- **Intel Arc 140V GPU (8GB)** available via `torch.xpu` (detected as `xpu` device).
- All notebooks use `"xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"`.
- XPU has 8GB VRAM — training with batch=128 may require gradient checkpointing or smaller batch.

## Project state
- Full pipeline implemented: EDA → target generation → model definition → training → evaluation → visualization.
- `utils/*.py` contains core logic; `notebooks/*.ipynb` orchestrates experiments.
- SpecTNT: 1,193,768 params, ResNet front-end + 5× SpecTNTBlock (spectral + temporal self-attention).
- Two training variants: **A** (no CTL) and **B** (with CTL via CTC loss, blank=7).
- 4-fold CV on HarmonixSet only (912 songs, melspecs + segments 100% available).
- `checkpoints/` directory created for model saves.
- **No commits, no CI, no tests, no linter/formatter/typechecker** configured.

## Quick commands
- `uv sync` — install all dependencies
- `uv run jupyter notebook notebooks/` — launch notebook server
- `uv run python -m jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb` — run notebook headless
- `uv add <pkg>` / `uv remove <pkg>` — manage dependencies
