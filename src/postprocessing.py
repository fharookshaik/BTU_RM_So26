import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.data_utils import TARGET_HOP_TIME, FUNCTION_LABELS


def peak_picking(
    boundary_curve: np.ndarray,
    min_distance: float = 1.5,
    sigma: float = 3.0,
    threshold_factor: float = 0.3,
) -> list[float]:
    smooth = gaussian_filter1d(boundary_curve.astype(np.float64), sigma=sigma, mode="constant")
    threshold = np.median(smooth) + threshold_factor * np.std(smooth)

    min_dist_frames = int(round(min_distance / TARGET_HOP_TIME))
    peaks = []
    i = 0
    while i < len(smooth):
        if smooth[i] > threshold:
            start = max(0, i - min_dist_frames)
            end = min(len(smooth), i + min_dist_frames + 1)
            local_window = smooth[start:end]
            local_max_idx = np.argmax(local_window) + start
            if local_max_idx == i:
                peaks.append(i * TARGET_HOP_TIME)
            i = end
        else:
            i += 1

    if not peaks or peaks[0] > 0.5:
        peaks.insert(0, 0.0)
    return peaks


def assign_functions(
    func_curves: np.ndarray,
    boundaries: list[float],
    duration: float,
) -> list[tuple[float, float, str]]:
    n_frames = func_curves.shape[0]
    segments = []
    for i in range(len(boundaries) - 1):
        start_t = boundaries[i]
        end_t = boundaries[i + 1]
        if end_t - start_t < 0.01:
            continue
        start_f = int(round(start_t / TARGET_HOP_TIME))
        end_f = int(round(end_t / TARGET_HOP_TIME))
        end_f = min(end_f, n_frames)
        if end_f <= start_f:
            end_f = start_f + 1
        segment_means = func_curves[start_f:end_f, :].mean(axis=0)
        label_idx = int(np.argmax(segment_means))
        segments.append((start_t, min(end_t, duration), FUNCTION_LABELS[label_idx]))
    if not segments:
        mid = n_frames // 2
        label_idx = int(np.argmax(func_curves[mid]))
        segments.append((0.0, duration, FUNCTION_LABELS[label_idx]))
    return segments


def postprocess_song(
    boundary_curve: np.ndarray,
    func_curves: np.ndarray,
    duration: float,
) -> list[tuple[float, float, str]]:
    prob_funcs = 1.0 / (1.0 + np.exp(-func_curves))
    boundaries = peak_picking(boundary_curve)
    if not boundaries or abs(boundaries[-1] - duration) > 0.01:
        boundaries.append(duration)
    segments = assign_functions(prob_funcs, boundaries, duration)
    return segments


def curves_to_segments(
    boundary_logits: np.ndarray,
    func_logits: np.ndarray,
    duration: float,
) -> list[tuple[float, float, str]]:
    return postprocess_song(boundary_logits, func_logits, duration)
