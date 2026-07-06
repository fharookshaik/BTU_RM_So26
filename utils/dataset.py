"""PyTorch Dataset for HarmonixSet 24s chunks."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .target_generation import (
    build_targets, TARGET_FPS, NATIVE_FPS, DOWNSAMPLE_FACTOR,
    seconds_to_target_frames,
)

MELSPEC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "harmonixset", "melspecs")
)
SEGMENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "harmonixset", "dataset", "segments")
)

CHUNK_SECONDS = 24.0
HOP_SECONDS = 3.0
DEFAULT_CHUNK_FRAMES = int(round(CHUNK_SECONDS * TARGET_FPS))  # target-resolution frames


class HarmonixDataset(Dataset):
    """Load melspecs + segment annotations, enumerate 24s chunks with 3s hop.

    Returns (melspec_chunk, boundary_target, function_targets, token_seq)
    where:
        melspec_chunk: (1, 80, T_native) — native-resolution mel chunk
        boundary_target: (T_target, 1) — boundary activation
        function_targets: (T_target, 7) — function activation curves
        token_seq: list[int] — CTL token sequence for the chunk
    """

    def __init__(self, song_ids, augment=None, chunk_frames=None):
        super().__init__()
        self.song_ids = list(song_ids)
        self.augment = augment
        self.chunk_frames = chunk_frames or DEFAULT_CHUNK_FRAMES
        self.chunk_native_frames = int(round(CHUNK_SECONDS * NATIVE_FPS))
        self.chunk_native_frames = (self.chunk_native_frames // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR

        self.chunks = []
        for sid in self.song_ids:
            melspec_path = os.path.join(MELSPEC_DIR, f"{sid}-mel.npy")
            seg_path = os.path.join(SEGMENT_DIR, f"{sid}.txt")
            if not os.path.exists(melspec_path) or not os.path.exists(seg_path):
                continue
            melspec = np.load(melspec_path)
            total_native = melspec.shape[1]
            total_target = int(round(total_native / DOWNSAMPLE_FACTOR))

            boundaries, labels = self._load_segments(seg_path)
            b_curve, f_curves, token_seq = build_targets(boundaries, labels)

            target_len = min(len(b_curve), total_target)
            hop_frames = int(round(HOP_SECONDS * TARGET_FPS))
            for start in range(0, target_len - self.chunk_frames + 1, hop_frames):
                self.chunks.append((sid, start))

    @staticmethod
    def _load_segments(path):
        boundaries = []
        labels = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    boundaries.append(float(parts[0]))
                    labels.append(parts[1].strip())
        if not boundaries:
            return [0.0], ["inst"]
        boundaries = [0.0] + boundaries
        return boundaries, labels

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sid, target_offset = self.chunks[idx]
        melspec = np.load(os.path.join(MELSPEC_DIR, f"{sid}-mel.npy"))
        seg_data = self._load_segments(os.path.join(SEGMENT_DIR, f"{sid}.txt"))
        boundaries, labels = seg_data
        b_curve, f_curves, token_seq = build_targets(boundaries, labels)

        native_offset = target_offset * DOWNSAMPLE_FACTOR
        native_end = native_offset + self.chunk_native_frames
        total_native = melspec.shape[1]
        if native_end > total_native:
            native_offset = total_native - self.chunk_native_frames
            native_end = total_native

        chunk = melspec[:, native_offset:native_end]
        if chunk.shape[1] < self.chunk_native_frames:
            pad = self.chunk_native_frames - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad)))

        if self.augment is not None:
            chunk = self.augment(chunk)

        chunk_t = torch.from_numpy(chunk).unsqueeze(0).float()

        end_frame = min(target_offset + self.chunk_frames, len(b_curve))
        actual_frames = end_frame - target_offset
        if actual_frames < self.chunk_frames:
            target_offset = max(0, len(b_curve) - self.chunk_frames)
            end_frame = len(b_curve)
            actual_frames = self.chunk_frames

        b_target = b_curve[target_offset:end_frame]
        f_target = f_curves[target_offset:end_frame]
        if len(b_target) < self.chunk_frames:
            pad = self.chunk_frames - len(b_target)
            b_target = np.pad(b_target, (0, pad))
            f_target = np.pad(f_target, ((0, pad), (0, 0)))

        b_tensor = torch.from_numpy(b_target).float().unsqueeze(-1)
        f_tensor = torch.from_numpy(f_target).float()

        return chunk_t, b_tensor, f_tensor, token_seq
