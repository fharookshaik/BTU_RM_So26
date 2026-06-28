import os, ast
import numpy as np
import pandas as pd

HARMONIX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "harmonixset")
SEGMENT_DIR = os.path.join(HARMONIX_DIR, "dataset", "segments")
AUDIO_DIR = os.path.join(HARMONIX_DIR, "audio")
METADATA_FILE = os.path.join(HARMONIX_DIR, "dataset", "metadata.csv")

SAMPLE_RATE = 16000
HOP_LENGTH = 512
TIME_RES = HOP_LENGTH / SAMPLE_RATE  # 0.032, but paper uses ~0.192
FRAME_HOP = 512  # for STFT
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # ~31.25 Hz but paper uses ~5.2 Hz

# Paper: time resolution of ~5.2 per second (0.192s). Our STFT hop=512 at 16kHz gives ~31.25 Hz.
# We need to downsample to match paper's resolution. The paper uses a hop of 512 at 16kHz which
# gives 31.25 frames/sec, but their targets are at ~5.2 Hz (0.192s). This means they either
# average or downsample. Let's match what the paper says: "time resolution ~5.2 per second"
# which is about 1/0.192.
# To get exactly ~5.2 Hz, we can use stride=6 on the STFT frames (31.25/6 ≈ 5.2)
TARGET_HOP = 6  # downsample STFT frames by 6x → ~5.2 Hz
TARGET_FRAME_RATE = FRAME_RATE / TARGET_HOP  # ~5.208 Hz
TARGET_HOP_TIME = TARGET_HOP * TIME_RES  # ~0.192s
# Actual hop: 512 * 6 = 3072 samples at 16kHz = 0.192s ✓

FUNCTION_LABELS = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence"]
LABEL_TO_IDX = {l: i for i, l in enumerate(FUNCTION_LABELS)}
NUM_FUNCTIONS = len(FUNCTION_LABELS)  # 7
NUM_OUTPUTS = 1 + NUM_FUNCTIONS  # 8 (boundary + 7 functions)

SUBBSTRINGS = [
    ("silence", "silence"), ("pre-chorus", "verse"), ("prechorus", "verse"),
    ("refrain", "chorus"), ("chorus", "chorus"), ("theme", "chorus"),
    ("stutter", "chorus"), ("verse", "verse"), ("rap", "verse"),
    ("section", "verse"), ("slow", "verse"), ("build", "verse"),
    ("dialog", "verse"), ("intro", "intro"), ("raden", "intro"),
    ("opening", "intro"), ("bridge", "bridge"), ("trans", "bridge"),
    ("out", "outro"), ("coda", "outro"), ("ending", "outro"),
    ("break", "inst"), ("inst", "inst"), ("interlude", "inst"),
    ("improv", "inst"), ("solo", "inst")
]

def convert_label(label: str) -> str:
    if label == "end":
        return "end"
    for s1, s2 in SUBBSTRINGS:
        if s1 in label.lower():
            return s2
    return "inst"

def load_segments(filepath: str) -> list[tuple[float, float, str]]:
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

def get_duration_from_segments(filepath: str) -> float:
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

def get_songs_with_audio() -> list[str]:
    audio_files = {os.path.splitext(f)[0] for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")}
    seg_files = {os.path.splitext(f)[0] for f in os.listdir(SEGMENT_DIR) if f.endswith(".txt")}
    return sorted(audio_files & seg_files)

def generate_target_curves(
    segments: list[tuple[float, float, str]],
    duration: float,
    boundary_width: float = 0.6,
    hann_width: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = int(np.ceil(duration / TARGET_HOP_TIME))
    boundary = np.zeros(n_frames, dtype=np.float32)
    functions = np.zeros((n_frames, NUM_FUNCTIONS), dtype=np.float32)

    # Boundary curve: 0.6s window around each segment boundary
    boundary_frames = int(np.ceil(boundary_width / TARGET_HOP_TIME))
    onset_times = [seg[0] for seg in segments] + [duration]
    for t in onset_times:
        center = int(round(t / TARGET_HOP_TIME))
        start = max(0, center - boundary_frames // 2)
        end = min(n_frames, center + boundary_frames // 2)
        boundary[start:end] = 1.0

    # Hann smoothing for function curves
    hann_frames = int(round(hann_width / TARGET_HOP_TIME))
    hann_window = np.hanning(hann_frames * 2 + 1)
    ramp_up = hann_window[:hann_frames]
    ramp_down = hann_window[-hann_frames:]

    for start_t, end_t, label in segments:
        if label not in LABEL_TO_IDX:
            continue
        idx = LABEL_TO_IDX[label]
        onset_frame = int(round(start_t / TARGET_HOP_TIME))
        offset_frame = int(round(end_t / TARGET_HOP_TIME))

        # Ramp up (1s before onset)
        ru_start = max(0, onset_frame - hann_frames)
        ru_len = min(hann_frames, onset_frame - ru_start)
        for j in range(ru_len):
            if ru_start + j < n_frames:
                functions[ru_start + j, idx] = max(
                    functions[ru_start + j, idx], 1.0 - ramp_up[hann_frames - ru_len + j]
                )

        # Full activation
        for j in range(onset_frame, min(offset_frame, n_frames)):
            functions[j, idx] = 1.0

        # Ramp down (1s after offset)
        rd_start = offset_frame
        rd_len = min(hann_frames, n_frames - rd_start)
        for j in range(rd_len):
            functions[rd_start + j, idx] = max(
                functions[rd_start + j, idx], ramp_down[j] if j < len(ramp_down) else 0
            )

    return boundary, functions

def load_all_annotations() -> dict:
    song_ids = get_songs_with_audio()
    annotations = {}
    for sid in song_ids:
        seg_path = os.path.join(SEGMENT_DIR, f"{sid}.txt")
        segments = load_segments(seg_path)
        duration = get_duration_from_segments(seg_path)
        boundary, functions = generate_target_curves(segments, duration)
        annotations[sid] = {
            "segments": segments,
            "duration": duration,
            "boundary": boundary,
            "functions": functions,
        }
    return annotations

def get_section_tokens(segments: list[tuple[float, float, str]]) -> list[int]:
    return [LABEL_TO_IDX[lbl] for _, _, lbl in segments if lbl in LABEL_TO_IDX]
