import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_utils import (
    TARGET_HOP_TIME, TARGET_FRAME_RATE, NUM_OUTPUTS, NUM_FUNCTIONS,
    FUNCTION_LABELS, LABEL_TO_IDX, get_songs_with_audio, load_segments,
    generate_target_curves, get_duration_from_segments, get_section_tokens,
    HARMONIX_DIR, SEGMENT_DIR, AUDIO_DIR,
)
from src.features import load_features, FEATURES_DIR

CHUNK_DURATION = 12.0
CHUNK_HOP = 3.0
CHUNK_FRAMES = int(CHUNK_DURATION / TARGET_HOP_TIME)
CHUNK_HOP_FRAMES = int(CHUNK_HOP / TARGET_HOP_TIME)


class HarmonixChunkDataset(Dataset):
    def __init__(self, song_ids: list[str], annotations: dict, augment: bool = False):
        self.song_ids = song_ids
        self.annotations = annotations
        self.augment = augment
        self.feat_cache: dict[str, np.ndarray] = {}

        self.chunks = []
        for sid in song_ids:
            ann = annotations.get(sid)
            if ann is None:
                continue
            n_frames_ann = ann["boundary"].shape[0]
            try:
                feat = load_features(sid)
                n_frames_feat = feat.shape[1]
            except (FileNotFoundError, OSError):
                continue
            n_frames = min(n_frames_ann, n_frames_feat)
            if n_frames < CHUNK_FRAMES:
                continue
            self.feat_cache[sid] = feat
            for start in range(0, n_frames - CHUNK_FRAMES + 1, CHUNK_HOP_FRAMES):
                self.chunks.append((sid, start))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sid, start_frame = self.chunks[idx]
        ann = self.annotations[sid]
        end_frame = start_frame + CHUNK_FRAMES

        feat = self.feat_cache[sid]
        if end_frame > feat.shape[1]:
            feat_chunk = np.pad(feat, ((0, 0), (0, end_frame - feat.shape[1])), mode="constant")[:, start_frame:end_frame]
        else:
            feat_chunk = feat[:, start_frame:end_frame]

        T_ann = ann["boundary"].shape[0]
        if end_frame > T_ann:
            b = np.pad(ann["boundary"], (0, end_frame - T_ann), mode="constant")[start_frame:end_frame]
            f = np.pad(ann["functions"], ((0, end_frame - T_ann), (0, 0)), mode="constant")[start_frame:end_frame, :]
        else:
            b = ann["boundary"][start_frame:end_frame]
            f = ann["functions"][start_frame:end_frame, :]

        if len(b) < CHUNK_FRAMES:
            pad = CHUNK_FRAMES - len(b)
            b = np.pad(b, (0, pad), mode="constant")
            f = np.pad(f, ((0, pad), (0, 0)), mode="constant")

        section_tokens = get_section_tokens(ann["segments"])

        return (
            torch.from_numpy(feat_chunk).float(),
            torch.from_numpy(b).float(),
            torch.from_numpy(f).float(),
            torch.tensor(section_tokens, dtype=torch.long),
        )


def collate_fn(batch):
    feats, boundaries, funcs, tokens = zip(*batch)
    feats = torch.stack(feats, dim=0)
    boundaries = torch.stack(boundaries, dim=0)
    funcs = torch.stack(funcs, dim=0)
    return feats, boundaries, funcs, tokens


def get_dataloader(song_ids, annotations, batch_size=8, shuffle=True, augment=False):
    dataset = HarmonixChunkDataset(song_ids, annotations, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )
