"""Build boundary/function activation curves and CTL token sequences."""

import numpy as np
from .label_conversion import convert_segments, CLASSES, CLASS_TO_IDX

SR = 22050
HOP_LENGTH = 1024
NATIVE_FPS = SR / HOP_LENGTH
DOWNSAMPLE_FACTOR = 4
TARGET_FPS = NATIVE_FPS / DOWNSAMPLE_FACTOR


def seconds_to_native_frames(seconds):
    return int(round(seconds * NATIVE_FPS))


def seconds_to_target_frames(seconds):
    return int(round(seconds * TARGET_FPS))


def boundary_curve(boundaries, total_frames):
    """Binary pulse of 0.6s width at each segment boundary.

    Args:
        boundaries: list of boundary timestamps in seconds
        total_frames: number of target-resolution frames

    Returns:
        np.ndarray of shape (total_frames,) — 0/1 pulse train
    """
    curve = np.zeros(total_frames, dtype=np.float32)
    pulse_width = seconds_to_target_frames(0.6)
    for b in boundaries:
        center = seconds_to_target_frames(b)
        start = max(0, center - pulse_width // 2)
        end = min(total_frames, center + pulse_width // 2 + 1)
        curve[start:end] = 1.0
    return curve


def function_curves(boundaries, labels, total_frames):
    """Binary mask per class smoothed with 2s Hann window.

    Args:
        boundaries: list of segment boundary timestamps (seconds), length N+1.
                    The last boundary is the song end.
        labels: list of raw label strings, length N.
        total_frames: number of target-resolution frames.

    Returns:
        np.ndarray of shape (total_frames, 7) — smoothed activation per class.
    """
    hann_length = int(round(2.0 * TARGET_FPS))
    if hann_length % 2 == 0:
        hann_length += 1
    window = np.hanning(hann_length)
    window /= window.sum()

    masks = np.zeros((total_frames, 7), dtype=np.float32)
    for i, lbl in enumerate(labels):
        cls = CLASS_TO_IDX.get(lbl, -1)
        if cls < 0:
            continue
        start_s = boundaries[i]
        end_s = boundaries[i + 1]
        start_f = seconds_to_target_frames(start_s)
        end_f = seconds_to_target_frames(end_s)
        masks[start_f:end_f, cls] = 1.0

    curves = np.zeros_like(masks)
    for c in range(7):
        curves[:, c] = np.convolve(masks[:, c], window, mode="same")
    return curves


def build_targets(boundaries, labels):
    """Build all targets for a song.

    Args:
        boundaries: list of boundary timestamps including song start and end
        labels: list of raw label strings (one per segment)

    Returns:
        boundary_curve: (T,) float32
        function_curves: (T, 7) float32
        token_seq: list of int (class indices for CTL)
        token_seq_clean: list of int (excluding 'end')
    """
    total_duration = boundaries[-1]
    total_frames = seconds_to_target_frames(total_duration)
    if total_frames == 0:
        total_frames = 1

    b_curve = boundary_curve(boundaries, total_frames)
    f_curves = function_curves(boundaries, labels, total_frames)

    converted, token_seq = convert_segments(boundaries[:-1], labels)
    return b_curve, f_curves, token_seq
