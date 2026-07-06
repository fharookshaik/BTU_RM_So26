"""Temporal smoothing decoder: Viterbi-based label smoothing with data-driven transition matrix."""

import os
import numpy as np
from .label_conversion import CLASSES, CLASS_TO_IDX, convert_label
from .postprocessing import peak_picking


def compute_transition_matrix(song_ids, segment_dir, self_loop_prior=50.0):
    """Estimate a 7x7 row-stochastic transition matrix from training segment bigrams.

    Args:
        song_ids: list of song ID strings used for training.
        segment_dir: path to directory containing *-segments.txt files.
        self_loop_prior: Dirichlet pseudo-count added to diagonal (higher -> more stickiness).

    Returns:
        (7, 7) float64 transition matrix where row i -> P(next = j | current = i).
    """
    C = len(CLASSES)
    counts = np.zeros((C, C), dtype=np.float64)

    for sid in song_ids:
        path = os.path.join(segment_dir, f"{sid}.txt")
        if not os.path.exists(path):
            continue
        boundaries = []
        labels = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    boundaries.append(float(parts[0]))
                    labels.append(parts[1].strip())
        if not labels:
            continue

        indices = []
        for lbl in labels:
            converted = convert_label(lbl)
            if converted in CLASS_TO_IDX:
                indices.append(CLASS_TO_IDX[converted])

        for i in range(len(indices) - 1):
            counts[indices[i], indices[i + 1]] += 1.0

    for c in range(C):
        counts[c, c] += self_loop_prior

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    transition_matrix = counts / row_sums

    return transition_matrix


def viterbi_decode(frame_probs, transition_matrix):
    """Viterbi decoding for optimal state sequence.

    Args:
        frame_probs: (T, C) float32 — per-frame class probabilities.
        transition_matrix: (C, C) float64 — row-stochastic.

    Returns:
        path: (T,) int32 — most likely state sequence.
    """
    T, C = frame_probs.shape
    log_emit = np.log(np.maximum(frame_probs, 1e-12))
    log_trans = np.log(np.maximum(transition_matrix, 1e-12))

    delta = np.zeros((T, C), dtype=np.float64)
    psi = np.zeros((T, C), dtype=np.int32)

    delta[0] = log_emit[0] - np.log(C)

    for t in range(1, T):
        for j in range(C):
            scores = delta[t - 1] + log_trans[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + log_emit[t, j]

    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def smoothed_postprocess(boundary_logits, function_logits, transition_matrix,
                         fps=5.38, **pk_kwargs):
    """Post-processing with Viterbi label smoothing.

    Boundaries are detected via peak-picking (identical to baseline).
    Labels are assigned by Viterbi within each segment then majority-voted.

    Args:
        boundary_logits: (T,) or (T, 1) float32.
        function_logits: (T, 7) float32.
        transition_matrix: (7, 7) float64.
        fps: target frames per second.

    Returns:
        boundaries_sec: list of float.
        pred_labels: list of str.
        boundaries_frames: list of int.
    """
    if boundary_logits.ndim == 2 and boundary_logits.shape[-1] == 1:
        boundary_logits = boundary_logits.squeeze(-1)
    boundary_curve = 1.0 / (1.0 + np.exp(-boundary_logits))
    function_probs = 1.0 / (1.0 + np.exp(-function_logits))

    peak_frames = peak_picking(boundary_curve, **pk_kwargs)

    if len(peak_frames) < 2:
        peak_frames = [0, len(boundary_curve) - 1]

    peak_frames = sorted(set([0] + peak_frames + [len(boundary_curve)]))
    boundaries_sec = [f / fps for f in peak_frames]

    labels = []
    for i in range(len(peak_frames) - 1):
        start = peak_frames[i]
        end = peak_frames[i + 1]
        seg_probs = function_probs[start:end]
        if len(seg_probs) == 0:
            labels.append("inst")
            continue
        path = viterbi_decode(seg_probs, transition_matrix)
        majority_class = int(np.bincount(path).argmax())
        labels.append(CLASSES[majority_class])

    return boundaries_sec, labels, peak_frames
