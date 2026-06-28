import os, sys, json, time
import numpy as np
import torch
from sklearn.model_selection import KFold

from src.data_utils import (
    get_songs_with_audio, load_all_annotations, load_segments,
    get_duration_from_segments, SEGMENT_DIR, AUDIO_DIR, TARGET_HOP_TIME,
    FUNCTION_LABELS, HARMONIX_DIR,
)
from src.features import precompute_features, load_features, FEATURES_DIR
from src.dataset import get_dataloader, CHUNK_FRAMES
from src.models.harmonic_cnn import HarmonicCNN
from src.models.spectnt import SpecTNT
from src.training import Trainer, MODELS_DIR, OUTPUTS_DIR
from src.postprocessing import postprocess_song
from src.metrics import compute_metrics

RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
MAX_EPOCHS = 60


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

    train_loader = get_dataloader(train_ids, annotations, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_ids, annotations, batch_size=BATCH_SIZE, shuffle=False)

    model = model_class()
    trainer = Trainer(model, device=DEVICE, lr=0.0005, weight_decay=0.9, use_ctl=use_ctl)
    trainer.fit(train_loader, val_loader, max_epochs=max_epochs)

    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_fold{fold}_best.pt")
    trainer.save_checkpoint(ckpt_path)

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
    one_hot_labels = set()
    for sid in val_ids:
        ann = annotations[sid]
        duration = ann["duration"]

        if is_instant:
            boundary_curve, func_curve = predict_song_instant(model, sid, DEVICE)
        else:
            boundary_curve, func_curve = predict_song_multipoint(model, sid, DEVICE)

        est_segments = postprocess_song(boundary_curve, func_curve, duration)
        est_seg_counts.append(len(est_segments))
        ref_segments = ann["segments"]
        for s, e, l in ref_segments:
            one_hot_labels.add(l)

        metrics = compute_metrics(ref_segments, est_segments, duration)
        all_metrics.append(metrics)

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\nFold {fold} results ({model_name}):")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  [DBG] est seg counts: min={min(est_seg_counts)} max={max(est_seg_counts)} mean={np.mean(est_seg_counts):.1f} <=1={sum(1 for c in est_seg_counts if c<=1)}")
    print(f"  [DBG] ref labels: {sorted(one_hot_labels)}")

    return avg_metrics


def main():
    print(f"Device: {DEVICE}")
    print(f"Songs with audio: {len(get_songs_with_audio())}")

    songs = get_songs_with_audio()
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

    all_results = {}
    for model_name, model_class, use_ctl in model_configs:
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(song_ids)):
            train_ids = song_ids[train_idx].tolist()
            val_ids = song_ids[val_idx].tolist()
            metrics = run_fold(
                model_class, model_name, train_ids, val_ids,
                annotations, fold, use_ctl=use_ctl,
            )
            fold_metrics.append(metrics)

        avg_across_folds = {
            k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]
        }
        all_results[model_name] = {
            "per_fold": [{k: float(v) for k, v in m.items()} for m in fold_metrics],
            "average": avg_across_folds,
        }
        print(f"\n{'='*60}")
        print(f"{model_name} — Average across 4 folds:")
        for k, v in avg_across_folds.items():
            print(f"  {k}: {v:.4f}")

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
