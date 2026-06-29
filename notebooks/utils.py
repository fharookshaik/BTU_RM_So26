import os, csv, ast, functools
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.ndimage import gaussian_filter1d

HARMONIX_DIR = "data/harmonixset"
SEGMENT_DIR = os.path.join(HARMONIX_DIR, "dataset", "segments")
AUDIO_DIR = os.path.join(HARMONIX_DIR, "audio")
FEATURES_DIR = os.path.join(HARMONIX_DIR, "features")
SAMPLE_RATE = 16000
HOP_LENGTH = 512
TARGET_HOP = 6
TARGET_HOP_TIME = TARGET_HOP * HOP_LENGTH / SAMPLE_RATE
N_MELS = 96
FUNCTION_LABELS = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence"]
LABEL_TO_IDX = {l: i for i, l in enumerate(FUNCTION_LABELS)}
NUM_FUNCTIONS = len(FUNCTION_LABELS)
NUM_OUTPUTS = 1 + NUM_FUNCTIONS

SUBBSTRINGS = [
    ("silence", "silence"), ("pre-chorus", "verse"), ("prechorus", "verse"),
    ("refrain", "chorus"), ("chorus", "chorus"), ("theme", "chorus"),
    ("stutter", "chorus"), ("verse", "verse"), ("rap", "verse"),
    ("section", "verse"), ("slow", "verse"), ("build", "verse"),
    ("dialog", "verse"), ("intro", "intro"), ("raden", "intro"),
    ("opening", "intro"), ("bridge", "bridge"), ("trans", "bridge"),
    ("out", "outro"), ("coda", "outro"), ("ending", "outro"),
    ("break", "inst"), ("inst", "inst"), ("interlude", "inst"),
    ("improv", "inst"), ("solo", "inst"),
]

def convert_label(label):
    if label == "end":
        return "end"
    for s1, s2 in SUBBSTRINGS:
        if s1 in label.lower():
            return s2
    return "inst"

def load_segments(filepath):
    segments = []
    with open(filepath) as f:
        lines = f.read().strip().splitlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        start = float(parts[0])
        label = " ".join(parts[1:])
        converted = convert_label(label)
        if converted == "end":
            continue
        end = float(lines[i + 1].split()[0]) if i + 1 < len(lines) else start
        segments.append((start, end, converted))
    if not segments:
        return segments
    offset = segments[0][0]
    segments = [(s - offset, e - offset, l) for s, e, l in segments]
    return segments

def get_duration_from_segments(filepath):
    offset = 0.0
    with open(filepath) as f:
        lines = f.read().strip().splitlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            if i == 0 and parts[-1].lower() != "end":
                offset = float(parts[0])
            if parts[-1].lower() == "end":
                return float(parts[0]) - offset
    return 0.0

def get_songs_with_audio():
    audio_files = set()
    if os.path.isdir(AUDIO_DIR):
        audio_files = {os.path.splitext(f)[0] for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")}
    seg_files = set()
    if os.path.isdir(SEGMENT_DIR):
        seg_files = {os.path.splitext(f)[0] for f in os.listdir(SEGMENT_DIR) if f.endswith(".txt")}
    return sorted(audio_files & seg_files)

def generate_target_curves(segments, duration, boundary_width=0.6, hann_width=2.0):
    n_frames = int(np.ceil(duration / TARGET_HOP_TIME))
    boundary = np.zeros(n_frames, dtype=np.float32)
    functions = np.zeros((n_frames, NUM_FUNCTIONS), dtype=np.float32)

    b_frames = int(np.ceil(boundary_width / TARGET_HOP_TIME))
    onset_times = [seg[0] for seg in segments] + [duration]
    for t in onset_times:
        center = int(round(t / TARGET_HOP_TIME))
        s = max(0, center - b_frames // 2)
        e = min(n_frames, center + b_frames // 2)
        boundary[s:e] = 1.0

    h_f = int(round(hann_width / TARGET_HOP_TIME))
    hann = np.hanning(h_f * 2 + 1)
    ramp_up = hann[:h_f]
    ramp_down = hann[-h_f:]

    for start_t, end_t, label in segments:
        if label not in LABEL_TO_IDX:
            continue
        idx = LABEL_TO_IDX[label]
        on = int(round(start_t / TARGET_HOP_TIME))
        off = int(round(end_t / TARGET_HOP_TIME))

        ru_s = max(0, on - h_f)
        ru_l = min(h_f, on - ru_s)
        for j in range(ru_l):
            if ru_s + j < n_frames:
                functions[ru_s + j, idx] = max(functions[ru_s + j, idx], 1.0 - ramp_up[h_f - ru_l + j])
        for j in range(on, min(off, n_frames)):
            functions[j, idx] = 1.0
        rd_l = min(h_f, n_frames - off)
        for j in range(rd_l):
            functions[off + j, idx] = max(functions[off + j, idx], ramp_down[j] if j < len(ramp_down) else 0)

    return boundary, functions

def load_all_annotations():
    annotations = {}
    for sid in get_songs_with_audio():
        seg_path = os.path.join(SEGMENT_DIR, f"{sid}.txt")
        segments = load_segments(seg_path)
        duration = get_duration_from_segments(seg_path)
        boundary, functions = generate_target_curves(segments, duration)
        annotations[sid] = {"segments": segments, "duration": duration, "boundary": boundary, "functions": functions}
    return annotations

def get_section_tokens(segments):
    return [LABEL_TO_IDX[lbl] for _, _, lbl in segments if lbl in LABEL_TO_IDX]

def load_audio(filepath):
    waveform, sr = torchaudio.load(filepath)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform

def compute_mel_spectrogram(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=0, f_max=SAMPLE_RATE // 2, power=2.0, normalized=True,
    )
    spec = mel_spec(waveform)
    return torch.log(torch.clamp(spec, min=1e-10))

def compute_harmonic_features(waveform):
    mel = compute_mel_spectrogram(waveform).squeeze(0)
    T_s = mel.shape[1]
    T_t = T_s // TARGET_HOP
    mel = mel[:, :T_t * TARGET_HOP]
    return mel.view(N_MELS, T_t, TARGET_HOP).mean(dim=2)

def precompute_features(song_ids, force=False):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    paths = {}
    for sid in song_ids:
        out_path = os.path.join(FEATURES_DIR, f"{sid}.npy")
        if os.path.exists(out_path) and not force:
            paths[sid] = out_path
            continue
        audio_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
        if not os.path.exists(audio_path):
            continue
        wave = load_audio(audio_path)
        feat = compute_harmonic_features(wave)
        np.save(out_path, feat.numpy())
        paths[sid] = out_path
    return paths

@functools.lru_cache(maxsize=256)
def _load_npy(path):
    return np.load(path)

def load_features(sid):
    return _load_npy(os.path.join(FEATURES_DIR, f"{sid}.npy"))

def peak_picking(boundary_curve, min_distance=1.5, sigma=3.0, threshold_factor=0.3):
    smooth = gaussian_filter1d(boundary_curve.astype(np.float64), sigma=sigma, mode="constant")
    threshold = np.median(smooth) + threshold_factor * np.std(smooth)
    min_dist_frames = int(round(min_distance / TARGET_HOP_TIME))
    peaks = []
    i = 0
    while i < len(smooth):
        if smooth[i] > threshold:
            s = max(0, i - min_dist_frames)
            e = min(len(smooth), i + min_dist_frames + 1)
            local_max = np.argmax(smooth[s:e]) + s
            if local_max == i:
                peaks.append(i * TARGET_HOP_TIME)
            i = e
        else:
            i += 1
    if not peaks or peaks[0] > 0.5:
        peaks.insert(0, 0.0)
    return peaks

def assign_functions(func_curves, boundaries, duration):
    n_frames = func_curves.shape[0]
    segments = []
    for i in range(len(boundaries) - 1):
        s_t, e_t = boundaries[i], boundaries[i + 1]
        if e_t - s_t < 0.01:
            continue
        s_f = int(round(s_t / TARGET_HOP_TIME))
        e_f = min(int(round(e_t / TARGET_HOP_TIME)), n_frames)
        if e_f <= s_f:
            e_f = s_f + 1
        label_idx = int(np.argmax(func_curves[s_f:e_f, :].mean(axis=0)))
        segments.append((s_t, min(e_t, duration), FUNCTION_LABELS[label_idx]))
    if not segments:
        mid = n_frames // 2
        label_idx = int(np.argmax(func_curves[mid]))
        segments.append((0.0, duration, FUNCTION_LABELS[label_idx]))
    return segments

def postprocess_song(boundary_curve, func_curves, duration):
    prob_funcs = 1.0 / (1.0 + np.exp(-func_curves))
    boundaries = peak_picking(boundary_curve)
    if not boundaries or abs(boundaries[-1] - duration) > 0.01:
        boundaries.append(duration)
    return assign_functions(prob_funcs, boundaries, duration)

def filter_intervals(intervals):
    if len(intervals) == 0:
        return intervals
    mask = (intervals[:, 1] - intervals[:, 0]) > 0.01
    return intervals[mask]

def segments_to_arrays(segments):
    if not segments:
        return np.empty((0, 2)), np.array([])
    filtered = [(s, e, l) for s, e, l in segments if e - s > 0.01]
    if not filtered:
        return np.empty((0, 2)), np.array([])
    intervals = np.array([[s, e] for s, e, _ in filtered])
    labels = np.array([l for _, _, l in filtered])
    return intervals, labels

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

def compute_metrics(ref_segments, est_segments, duration):
    n_frames = int(np.ceil(duration / TARGET_HOP_TIME))
    ref_ints, ref_labs = segments_to_arrays(ref_segments)
    est_ints, est_labs = segments_to_arrays(est_segments)

    ref_frames = np.full(n_frames, "?", dtype=object)
    for (s, e), l in zip(ref_ints, ref_labs):
        ref_frames[int(round(s / TARGET_HOP_TIME)):int(round(e / TARGET_HOP_TIME))] = l
    est_frames = np.full(n_frames, "?", dtype=object)
    for (s, e), l in zip(est_ints, est_labs):
        est_frames[int(round(s / TARGET_HOP_TIME)):int(round(e / TARGET_HOP_TIME))] = l

    ref_ints = filter_intervals(ref_ints)
    est_ints = filter_intervals(est_ints)

    import mir_eval

    # HR.5F
    hr5f = 0.0
    if len(ref_ints) > 0 and len(est_ints) > 0:
        _, _, hr5f = mir_eval.segment.detection(ref_ints, est_ints, window=0.5)

    # ACC
    mask = (ref_frames != "?") & (est_frames != "?")
    acc = float(np.mean(ref_frames[mask] == est_frames[mask])) if mask.sum() > 0 else 0.0

    # PWF
    pwf = 0.0
    if len(ref_labs) > 0 and len(est_labs) > 0:
        _, _, pwf = mir_eval.segment.pairwise(ref_ints, ref_labs, est_ints, est_labs)

    # Sf
    sf = 0.0
    if len(ref_labs) > 0 and len(est_labs) > 0:
        try:
            _, _, sf = mir_eval.segment.vmeasure(ref_ints, ref_labs, est_ints, est_labs)
        except Exception:
            sf = 0.0

    # CHR.5F
    chr5f = 0.0
    chr_ints_r = chorus_intervals(ref_segments)
    chr_ints_e = chorus_intervals(est_segments)
    if len(chr_ints_r) > 0 and len(chr_ints_e) > 0:
        _, _, chr5f = mir_eval.segment.detection(chr_ints_r, chr_ints_e, window=0.5)

    # CFI
    cfi = pairwise_f_binary(make_chorus_binary(ref_segments, n_frames), make_chorus_binary(est_segments, n_frames))

    return {"HR.5F": round(hr5f, 3), "ACC": round(acc, 3), "PWF": round(pwf, 3), "Sf": round(sf, 3), "CHR.5F": round(chr5f, 3), "CFI": round(cfi, 3)}

def ctl_loss_batch(logits, target_padded, target_lengths, blank_logprob=-1e10):
    B, T, C = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    S_max = target_padded.shape[1]
    if S_max == 0:
        return torch.tensor(0.0, device=logits.device)

    token_log_probs = log_probs.gather(2, target_padded.unsqueeze(1).expand(-1, T, -1))

    mask_S = torch.arange(S_max, device=logits.device).unsqueeze(0) >= target_lengths.unsqueeze(1)
    mask_S = mask_S.unsqueeze(1)
    token_log_probs = torch.where(mask_S.expand(-1, T, -1), torch.tensor(blank_logprob, device=logits.device), token_log_probs)

    log_alpha_list = [torch.full((B, S_max), blank_logprob, device=logits.device) for _ in range(T + 1)]
    log_alpha_list[0] = log_alpha_list[0].clone()
    log_alpha_list[0][:, 0] = 0.0

    for t in range(1, T + 1):
        prev = log_alpha_list[t - 1]
        trans = torch.full_like(prev, blank_logprob)
        trans[:, 1:] = prev[:, :-1]
        cur = token_log_probs[:, t - 1, :] + torch.logaddexp(prev, trans)
        mask_s_gt_t = torch.arange(S_max, device=logits.device).unsqueeze(0) >= t
        cur = torch.where(mask_s_gt_t | mask_S.squeeze(1), torch.tensor(blank_logprob, device=logits.device), cur)
        log_alpha_list[t] = cur

    batch_idx = torch.arange(B, device=logits.device)
    S_last = (target_lengths - 1).clamp(min=0, max=S_max - 1)
    log_likelihood = torch.stack([la[batch_idx, S_last] for la in log_alpha_list], dim=1)[:, -1]

    valid = (target_lengths > 0) & (target_lengths <= T)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)
    return -log_likelihood[valid].mean()
