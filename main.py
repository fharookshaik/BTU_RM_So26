import os, sys, json
import numpy as np
import torch
from sklearn.model_selection import KFold

from src.data_utils import (
    get_songs_with_audio, load_all_annotations, FUNCTION_LABELS,
)
from src.features import precompute_features, load_features
from src.dataset import get_dataloader, CHUNK_FRAMES
from src.models.harmonic_cnn import HarmonicCNN
from src.models.spectnt import SpecTNT
from src.training import Trainer, MODELS_DIR, OUTPUTS_DIR
from src.postprocessing import postprocess_song
from src.metrics import compute_metrics

RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BATCH_SIZE = 128
MAX_EPOCHS = 100


def predict_song_instant(model, sid, device, chunk_frames=CHUNK_FRAMES):
    feat = load_features(sid)
    T = feat.shape[1]

    if T <= chunk_frames:
        feat_t = torch.from_numpy(feat).float().unsqueeze(0)
        feat_pad = torch.nn.functional.pad(feat_t, (0, chunk_frames - T))
        with torch.no_grad():
            out = model(feat_pad.to(device))
        b = np.full(T, out[0, 0].item())
        f = np.tile(out[0, 1:].cpu().numpy(), (T, 1))
        return b, f

    half = chunk_frames // 2
    boundary_curve = np.zeros(T)
    func_curve = np.zeros((T, len(FUNCTION_LABELS)))
    count = np.zeros(T)

    all_chunks = []
    positions = []
    for center in range(0, T):
        start = max(0, center - half)
        end = min(feat.shape[1], start + chunk_frames)
        if end - start < chunk_frames:
            start = max(0, end - chunk_frames)
        all_chunks.append(torch.from_numpy(feat[:, start:end]).float())
        positions.append(center)

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_t = torch.stack(batch_chunks).float().to(device)
        with torch.no_grad():
            batch_out = model(batch_t)
        for j, center in enumerate(positions[i:i + BATCH_SIZE]):
            boundary_curve[center] += batch_out[j, 0].item()
            func_curve[center] += batch_out[j, 1:].cpu().numpy()
            count[center] += 1

    count = np.maximum(count, 1)
    boundary_curve /= count
    func_curve /= count[:, np.newaxis]
    return boundary_curve, func_curve


def predict_song_multipoint(model, sid, device, chunk_hop=62):
    feat = load_features(sid)
    T = feat.shape[1]
    if T <= chunk_hop * 2:
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(feat_t)
        return out[0, :, 0].cpu().numpy(), out[0, :, 1:].cpu().numpy()

    boundary_curve = np.zeros(T)
    func_curve = np.zeros((T, 7))
    count = np.zeros(T)
    for start in range(0, T - chunk_hop + 1, chunk_hop):
        end = min(start + chunk_hop * 2, T)
        feat_chunk = torch.from_numpy(feat[:, start:end]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(feat_chunk)
        b_chunk = out[0, :, 0].cpu().numpy()
        f_chunk = out[0, :, 1:].cpu().numpy()
        n = min(len(b_chunk), end - start)
        boundary_curve[start:start+n] += b_chunk[:n]
        func_curve[start:start+n] += f_chunk[:n]
        count[start:start+n] += 1.0
    count = np.maximum(count, 1)
    return boundary_curve / count, func_curve / count[:, np.newaxis]


def run_fold(
    model_class,
    model_name,
    train_ids,
    val_ids,
    annotations,
    fold: int,
    max_epochs=MAX_EPOCHS,
    use_ctl=False,
):
    is_instant = model_name in ("HarmonicCNN",)
    print(f"\n{'='*60}")
    print(f"Fold {fold} — {model_name}" + (" (CTL)" if use_ctl else ""))
    print(f"Train: {len(train_ids)} songs, Val: {len(val_ids)} songs")
    print(f"{'='*60}")

    train_loader = get_dataloader(train_ids, annotations, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_loader = get_dataloader(val_ids, annotations, batch_size=BATCH_SIZE, shuffle=False)

    run_name = f"{model_name}/fold_{fold}"
    model = model_class()
    trainer = Trainer(model, device=DEVICE, lr=0.0005, weight_decay=0.9,
                      use_ctl=use_ctl, run_name=run_name)

    resume_path = os.path.join(MODELS_DIR, f"{model_name}_fold{fold}_checkpoint.pt")
    if os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)
        print(f"Resuming {model_name} fold {fold} from epoch {trainer.epoch}")

    trainer.fit(train_loader, val_loader, max_epochs=max_epochs,
                ckpt_path=resume_path)

    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR, "best_model.pt"),
            map_location=DEVICE,
            weights_only=True,
        )["model_state"]
    )
    model.eval()

    all_metrics = []
    est_seg_counts = []
    ref_seg_counts = []
    for sid in val_ids:
        ann = annotations[sid]
        duration = ann["duration"]

        if is_instant:
            boundary_curve, func_curve = predict_song_instant(model, sid, DEVICE)
        else:
            boundary_curve, func_curve = predict_song_multipoint(model, sid, DEVICE)

        est_segments = postprocess_song(boundary_curve, func_curve, duration)
        est_seg_counts.append(len(est_segments))
        ref_seg_counts.append(len(ann["segments"]))

        metrics = compute_metrics(ann["segments"], est_segments, duration)
        all_metrics.append(metrics)

    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    print(f"\nFold {fold} results ({model_name}):")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    est_mean = np.mean(est_seg_counts)
    ref_mean = np.mean(ref_seg_counts)
    print(f"  Segments: est={est_mean:.1f} ref={ref_mean:.1f}")

    if trainer.writer:
        for k, v in avg_metrics.items():
            trainer.writer.add_scalar(f"eval/{k}", v, 0)
        trainer.writer.add_scalar("eval/num_estimated_segments", est_mean, 0)
        trainer.writer.add_scalar("eval/num_reference_segments", ref_mean, 0)

        text = f"Fold {fold} — {model_name}"
        for k, v in avg_metrics.items():
            text += f"\n  {k}: {v:.4f}"
        text += f"\n  est segments: {est_mean:.1f}  ref segments: {ref_mean:.1f}"
        trainer.writer.add_text("eval/summary", text, 0)

    return avg_metrics


def main(model_name=None, fold_idx=None):
    print(f"Device: {DEVICE}")
    songs = get_songs_with_audio()
    print(f"Songs with audio: {len(songs)}")
    precompute_features(songs, force=False)
    annotations = load_all_annotations()
    print(f"Loaded {len(annotations)} annotations")

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    song_ids = np.array(songs)

    model_configs = [
        ("HarmonicCNN", HarmonicCNN, False),
        ("SpecTNT", SpecTNT, False),
        ("SpecTNT_CTL", SpecTNT, True),
    ]

    selected_configs = model_configs
    if model_name is not None:
        lookup = {
            "harmonic_cnn": model_configs[0],
            "spectnt": model_configs[1],
            "spectnt_ctl": model_configs[2],
        }
        selected_configs = [lookup[model_name]]

    all_results = {}
    for model_label, model_class, use_ctl in selected_configs:
        fold_metrics = []
        folds_to_run = [fold_idx] if fold_idx is not None else range(4)
        for fold in folds_to_run:
            train_idx, val_idx = list(kf.split(song_ids))[fold]
            metrics = run_fold(
                model_class, model_label, song_ids[train_idx].tolist(),
                song_ids[val_idx].tolist(), annotations, fold, use_ctl=use_ctl,
            )
            fold_metrics.append(metrics)

        avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        all_results[model_label] = {
            "per_fold": [{k: float(v) for k, v in m.items()} for m in fold_metrics],
            "average": avg,
        }
        print(f"\n{'='*60}")
        print(f"{model_label} — Average across {len(fold_metrics)} folds:")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["harmonic_cnn", "spectnt", "spectnt_ctl"], default=None)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3], default=None)
    args = parser.parse_args()
    main(model_name=args.model, fold_idx=args.fold)
