"""Evaluation metrics: mir_eval wrappers for structure analysis."""

import numpy as np
import mir_eval
from .label_conversion import CLASSES


METRIC_KEYS = ["hr.5f", "acc", "pwf", "sf", "chr.5f", "cf1", "macro_f1"]


def _to_intervals(boundaries_sec):
    """Convert boundary list to intervals array.

    Args:
        boundaries_sec: list of float — sorted boundary timestamps

    Returns:
        (N, 2) array of [start, end] intervals
    """
    b = np.asarray(boundaries_sec, dtype=np.float64)
    return np.column_stack([b[:-1], b[1:]])


def _deduplicate_boundaries(boundaries, labels):
    """Remove consecutive duplicate boundaries and the labels they create."""
    b = list(boundaries)
    l = list(labels)
    i = 1
    while i < len(b):
        if b[i] <= b[i - 1]:
            b.pop(i)
            l.pop(i - 1)
        else:
            i += 1
    return b, l


def _chorus_intervals(intervals, labels):
    """Filter intervals to only chorus-labeled segments.

    Args:
        intervals: (N, 2) array of [start, end]
        labels: list of N str

    Returns:
        (M, 2) array of chorus intervals (M <= N)
    """
    mask = np.array([l == "chorus" for l in labels])
    return intervals[mask]


def _binary_labels(labels):
    """Convert labels to binary chorus/non-chorus.

    Args:
        labels: list of str

    Returns:
        list of str — "chorus" or "non-chorus"
    """
    return ["chorus" if l == "chorus" else "non-chorus" for l in labels]


def evaluate_song(boundaries_pred, labels_pred, boundaries_ref, labels_ref):
    """Compute all metrics for one song.

    Args:
        boundaries_pred: list of float — predicted boundaries (seconds)
        labels_pred: list of str — predicted segment labels
        boundaries_ref: list of float — ground truth boundaries (seconds)
        labels_ref: list of str — ground truth segment labels

    Returns:
        dict with keys: hr.5f, acc, pwf, sf, chr.5f, cf1, macro_f1
    """
    boundaries_pred, labels_pred = _deduplicate_boundaries(boundaries_pred, labels_pred)
    boundaries_ref, labels_ref = _deduplicate_boundaries(boundaries_ref, labels_ref)

    est_intervals = _to_intervals(boundaries_pred)
    ref_intervals = _to_intervals(boundaries_ref)

    scores = mir_eval.segment.evaluate(
        ref_intervals, labels_ref, est_intervals, labels_pred,
    )

    acc = _frame_accuracy(ref_intervals, labels_ref, est_intervals, labels_pred)
    mf1 = _macro_f1(ref_intervals, labels_ref, est_intervals, labels_pred, CLASSES)

    # CHR.5F: boundary hit rate at 0.5s on chorus-only segments
    ref_chorus = _chorus_intervals(ref_intervals, labels_ref)
    est_chorus = _chorus_intervals(est_intervals, labels_pred)
    if len(ref_chorus) > 0 and len(est_chorus) > 0:
        _, _, chr5f = mir_eval.segment.detection(
            ref_chorus, est_chorus, window=0.5,
        )
    else:
        chr5f = 0.0

    # CF1: pairwise frame F-measure on binary chorus/non-chorus labels
    # Adjust intervals to match time ranges (mir_eval requires matching end times)
    adj_ref_intervals, ref_labels_adj = mir_eval.util.adjust_intervals(
        ref_intervals, labels=labels_ref, t_min=0.0,
    )
    adj_est_intervals, est_labels_adj = mir_eval.util.adjust_intervals(
        est_intervals, labels=labels_pred, t_min=0.0, t_max=ref_intervals.max(),
    )
    ref_binary = _binary_labels(ref_labels_adj)
    est_binary = _binary_labels(est_labels_adj)
    _, _, cf1 = mir_eval.segment.pairwise(
        adj_ref_intervals, ref_binary, adj_est_intervals, est_binary,
    )

    return {
        "hr.5f": float(scores["F-measure@0.5"]),
        "pwf": float(scores["Pairwise F-measure"]),
        "sf": float(scores["V-measure"]),
        "acc": float(acc),
        "chr.5f": float(chr5f),
        "cf1": float(cf1),
        "macro_f1": float(mf1),
    }


def _macro_f1(ref_intervals, ref_labels, est_intervals, est_labels, classes, hop=0.1):
    """Per-class F1 macro-averaged over frame-level labels."""
    total_dur = max(ref_intervals[-1, 1], est_intervals[-1, 1] if len(est_intervals) else 0)
    frames = np.arange(0, total_dur, hop)

    ref_frames = np.full(len(frames), "", dtype=object)
    for (s, e), lbl in zip(ref_intervals, ref_labels):
        mask = (frames >= s) & (frames < e)
        ref_frames[mask] = lbl

    est_frames = np.full(len(frames), "", dtype=object)
    for (s, e), lbl in zip(est_intervals, est_labels):
        mask = (frames >= s) & (frames < e)
        est_frames[mask] = lbl

    valid = (ref_frames != "") & (est_frames != "")
    if valid.sum() == 0:
        return 0.0

    ref_valid = ref_frames[valid]
    est_valid = est_frames[valid]

    f1_scores = []
    for cls in classes:
        ref_bin = ref_valid == cls
        est_bin = est_valid == cls
        tp = np.sum(ref_bin & est_bin)
        fp = np.sum(~ref_bin & est_bin)
        fn = np.sum(ref_bin & ~est_bin)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def _frame_accuracy(ref_intervals, ref_labels, est_intervals, est_labels, hop=0.1):
    """Frame-level labeling accuracy."""
    total_dur = max(ref_intervals[-1, 1], est_intervals[-1, 1] if len(est_intervals) else 0)
    frames = np.arange(0, total_dur, hop)

    ref_frames = np.full(len(frames), "", dtype=object)
    for (s, e), lbl in zip(ref_intervals, ref_labels):
        mask = (frames >= s) & (frames < e)
        ref_frames[mask] = lbl

    est_frames = np.full(len(frames), "", dtype=object)
    for (s, e), lbl in zip(est_intervals, est_labels):
        mask = (frames >= s) & (frames < e)
        est_frames[mask] = lbl

    valid = (ref_frames != "") & (est_frames != "")
    if valid.sum() == 0:
        return 0.0
    return float((ref_frames[valid] == est_frames[valid]).sum() / valid.sum())


def evaluate_all(boundaries_pred_list, labels_pred_list, boundaries_ref_list, labels_ref_list):
    """Compute metrics averaged across songs.

    Args:
        boundaries_pred_list: list of list[float]
        labels_pred_list: list of list[str]
        boundaries_ref_list: list of list[float]
        labels_ref_list: list of list[str]

    Returns:
        dict mapping metric name -> mean value
    """
    all_metrics = []
    for bp, lp, br, lr in zip(boundaries_pred_list, labels_pred_list, boundaries_ref_list, labels_ref_list):
        try:
            m = evaluate_song(bp, lp, br, lr)
            all_metrics.append(m)
        except Exception:
            pass

    if not all_metrics:
        return {k: 0.0 for k in METRIC_KEYS}

    keys = all_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
