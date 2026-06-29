import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from src.data_utils import (
    TARGET_HOP_TIME, NUM_FUNCTIONS,
    LABEL_TO_IDX,
    get_section_tokens,
)
from src.features import load_features

CHUNK_DURATION = 24.0
CHUNK_HOP = 3.0
CHUNK_FRAMES = int(CHUNK_DURATION / TARGET_HOP_TIME)
CHUNK_HOP_FRAMES = int(CHUNK_HOP / TARGET_HOP_TIME)


class SpecAugment:
    def __init__(self, freq_mask_param=8, time_mask_param=25, n_freq_masks=1, n_time_masks=1):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        x = spec.unsqueeze(0)
        for _ in range(self.n_freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.n_time_masks):
            x = self.time_mask(x)
        return x.squeeze(0)


class HarmonixChunkDataset(Dataset):
    def __init__(self, song_ids: list[str], annotations: dict, augment: bool = False):
        self.song_ids = song_ids
        self.annotations = annotations
        self.augment = augment
        self.augmenter = SpecAugment() if augment else None

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
            for start in range(0, n_frames - CHUNK_FRAMES + 1, CHUNK_HOP_FRAMES):
                self.chunks.append((sid, start))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sid, start_frame = self.chunks[idx]
        ann = self.annotations[sid]
        end_frame = start_frame + CHUNK_FRAMES

        feat = load_features(sid)
        if end_frame > feat.shape[1]:
            feat_chunk = np.pad(feat, ((0, 0), (0, end_frame - feat.shape[1])), mode="constant")[:, start_frame:end_frame]
        else:
            feat_chunk = feat[:, start_frame:end_frame]

        feat_tensor = torch.from_numpy(feat_chunk).float()
        if self.augment:
            feat_tensor = self.augmenter(feat_tensor)

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
            feat_tensor,
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


def get_dataloader(song_ids, annotations, batch_size=128, shuffle=True, augment=False):
    dataset = HarmonixChunkDataset(song_ids, annotations, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
