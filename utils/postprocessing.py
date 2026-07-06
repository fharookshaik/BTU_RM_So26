"""Post-processing: peak picking and segment labeling."""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def peak_picking(boundary_curve, threshold=0.3, min_distance=5, smooth_sigma=1.0):
    """Detect boundary timestamps from a boundary activation curve.

    Args:
        boundary_curve: (T,) float32
        threshold: minimum peak height
        min_distance: minimum frames between peaks
        smooth_sigma: Gaussian smoothing sigma (frames)

    Returns:
        list of boundary timestamps (seconds) including 0 and song end
    """
    curve = gaussian_filter1d(boundary_curve.astype(np.float64), smooth_sigma)

    peaks = []
    i = 0
    while i < len(curve):
        if curve[i] > threshold:
            peak_end = i
            while peak_end < len(curve) and curve[peak_end] > threshold:
                peak_end += 1
            peak_pos = i + np.argmax(curve[i:peak_end])
            if not peaks or (peak_pos - peaks[-1]) >= min_distance:
                peaks.append(peak_pos)
            i = peak_end
        else:
            i += 1

    return peaks


def segment_labeling(boundaries, function_curves, class_names=None):
    """Assign a function label to each segment.

    Args:
        boundaries: list of int frame indices (including 0 and total_frames)
        function_curves: (T, 7) float32
        class_names: list of 7 strings (optional)

    Returns:
        labels: list of str — predicted label per segment
    """
    if class_names is None:
        from .label_conversion import CLASSES as class_names

    labels = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment_probs = function_curves[start:end].mean(axis=0)
        cls_idx = int(np.argmax(segment_probs))
        labels.append(class_names[cls_idx])
    return labels


def postprocess(boundary_logits, function_logits, fps=5.38, **pk_kwargs):
    """Full post-processing: peak picking + segment labeling.

    Args:
        boundary_logits: (T,) or (T, 1) float32
        function_logits: (T, 7) float32
        fps: target frames per second

    Returns:
        boundaries_sec: list of float — boundary timestamps in seconds
        pred_labels: list of str — predicted segment labels
        boundaries_frames: list of int — boundary frame indices
    """
    if boundary_logits.ndim == 2 and boundary_logits.shape[-1] == 1:
        boundary_logits = boundary_logits.squeeze(-1)
    boundary_curve = 1.0 / (1.0 + np.exp(-boundary_logits))

    peak_frames = peak_picking(boundary_curve, **pk_kwargs)

    if len(peak_frames) < 2:
        peak_frames = [0, len(boundary_curve) - 1]

    peak_frames = sorted(set([0] + peak_frames + [len(boundary_curve)]))
    boundaries_sec = [f / fps for f in peak_frames]

    function_probs = 1.0 / (1.0 + np.exp(-function_logits))
    pred_labels = segment_labeling(peak_frames, function_probs)

    return boundaries_sec, pred_labels, peak_frames
