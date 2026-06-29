import numpy as np
import mir_eval

from src.data_utils import TARGET_HOP_TIME


def segments_to_arrays(segments):
    if not segments:
        return np.empty((0, 2)), np.array([])
    filtered = [(s, e, l) for s, e, l in segments if e - s > 0.01]
    if not filtered:
        return np.empty((0, 2)), np.array([])
    intervals = np.array([[s, e] for s, e, _ in filtered])
    labels = np.array([l for _, _, l in filtered])
    return intervals, labels


def compute_metrics(
    ref_segments: list[tuple[float, float, str]],
    est_segments: list[tuple[float, float, str]],
    duration: float,
) -> dict[str, float]:
    n_frames = int(np.ceil(duration / TARGET_HOP_TIME))

    ref_intervals, ref_labels = segments_to_arrays(ref_segments)
    est_intervals, est_labels = segments_to_arrays(est_segments)

    ref_frames = np.full(n_frames, "?", dtype=object)
    for (s, e), l in zip(ref_intervals, ref_labels):
        sf = int(round(s / TARGET_HOP_TIME))
        ef = int(round(e / TARGET_HOP_TIME))
        ref_frames[sf:ef] = l

    est_frames = np.full(n_frames, "?", dtype=object)
    for (s, e), l in zip(est_intervals, est_labels):
        sf = int(round(s / TARGET_HOP_TIME))
        ef = int(round(e / TARGET_HOP_TIME))
        est_frames[sf:ef] = l

    ref_intervals = filter_intervals(ref_intervals)
    est_intervals = filter_intervals(est_intervals)

    # 1. HR.5F — boundary detection F-measure at 0.5s
    hr_p = hr_r = hr5f = 0.0
    if len(ref_intervals) > 0 and len(est_intervals) > 0:
        hr_p, hr_r, hr_f = mir_eval.segment.detection(
            ref_intervals, est_intervals, window=0.5
        )
        hr5f = hr_f

    # 2. ACC — frame-wise accuracy
    mask = (ref_frames != "?") & (est_frames != "?")
    acc = float(np.mean(ref_frames[mask] == est_frames[mask])) if mask.sum() > 0 else 0.0

    # 3. PWF — Pair-wise frame clustering
    if len(ref_labels) > 0 and len(est_labels) > 0:
        pw_p, pw_r, pw_f = mir_eval.segment.pairwise(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        pwf = pw_f
    else:
        pwf = 0.0

    # 4. Sf — V-measure (homogeneity + completeness)
    if len(ref_labels) > 0 and len(est_labels) > 0:
        try:
            h, c, v = mir_eval.segment.vmeasure(
                ref_intervals, ref_labels,
                est_intervals, est_labels,
            )
            sf = v
        except Exception:
            sf = 0.0
    else:
        sf = 0.0

    # 5. CHR.5F — Chorus boundary detection
    chr_p = chr_r = chr5f = 0.0
    chr_intervals_r = chorus_intervals(ref_segments)
    chr_intervals_e = chorus_intervals(est_segments)
    if len(chr_intervals_r) > 0 and len(chr_intervals_e) > 0:
        chr_p, chr_r, chr_f = mir_eval.segment.detection(
            chr_intervals_r, chr_intervals_e, window=0.5
        )
        chr5f = chr_f

    # 6. CFI — Chorus vs non-chorus pairwise F
    ref_binary = make_chorus_binary(ref_segments, n_frames)
    est_binary = make_chorus_binary(est_segments, n_frames)
    cfi = pairwise_f_binary(ref_binary, est_binary)

    return {
        "HR.5F": round(hr5f, 3),
        "HR.5F_P": round(hr_p, 3),
        "HR.5F_R": round(hr_r, 3),
        "CHR.5F": round(chr5f, 3),
        "CHR.5F_P": round(chr_p, 3),
        "CHR.5F_R": round(chr_r, 3),
        "ACC": round(acc, 3),
        "PWF": round(pwf, 3),
        "Sf": round(sf, 3),
        "CFI": round(cfi, 3),
    }


def filter_intervals(intervals):
    if len(intervals) == 0:
        return intervals
    mask = (intervals[:, 1] - intervals[:, 0]) > 0.01
    return intervals[mask]


def chorus_intervals(segments):
    intervals = []
    for s, e, l in segments:
        if l.lower() == "chorus" and e - s > 0.01:
            intervals.append([s, e])
    return np.array(intervals) if intervals else np.empty((0, 2))


def make_chorus_binary(segments, n_frames):
    binary = np.zeros(n_frames, dtype=bool)
    for s, e, l in segments:
        if l.lower() == "chorus":
            sf = int(round(s / TARGET_HOP_TIME))
            ef = int(round(e / TARGET_HOP_TIME))
            binary[sf:ef] = True
    return binary


def pairwise_f_binary(ref, est):
    n = len(ref)
    if n < 2:
        return 0.0
    matched = ref_pos = est_pos = 0
    for i in range(n):
        for j in range(i + 1, n):
            sr = ref[i] == ref[j]
            se = est[i] == est[j]
            if sr and se:
                matched += 1
            if sr:
                ref_pos += 1
            if se:
                est_pos += 1
    prec = matched / est_pos if est_pos > 0 else 0.0
    rec = matched / ref_pos if ref_pos > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
