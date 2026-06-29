import os
import functools
import numpy as np
import torch
import torchaudio

from src.data_utils import (
    HARMONIX_DIR, AUDIO_DIR, SAMPLE_RATE, HOP_LENGTH, TARGET_HOP,
    TARGET_HOP_TIME, TARGET_FRAME_RATE, FRAME_RATE,
)

N_MELS = 96
WINDOW_LENGTH = 1024
FEATURES_DIR = os.path.join(HARMONIX_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)


def load_audio(filepath: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(filepath)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def compute_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=0,
        f_max=SAMPLE_RATE // 2,
        power=2.0,
        normalized=True,
    )
    spec = mel_spec(waveform)
    spec = torch.log(torch.clamp(spec, min=1e-10))
    return spec


def compute_harmonic_features(waveform: torch.Tensor) -> torch.Tensor:
    mel = compute_mel_spectrogram(waveform)
    mel = mel.squeeze(0)
    T_stft = mel.shape[1]
    T_target = T_stft // TARGET_HOP
    T_target_floor = T_stft // TARGET_HOP
    mel = mel[:, :T_target_floor * TARGET_HOP]
    mel = mel.view(N_MELS, T_target, TARGET_HOP).mean(dim=2)
    return mel


def precompute_features(song_ids: list[str], force: bool = False) -> dict[str, str]:
    paths = {}
    for sid in song_ids:
        out_path = os.path.join(FEATURES_DIR, f"{sid}.npy")
        if os.path.exists(out_path) and not force:
            paths[sid] = out_path
            continue
        audio_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
        if not os.path.exists(audio_path):
            continue
        waveform = load_audio(audio_path)
        feat = compute_harmonic_features(waveform)
        np.save(out_path, feat.numpy())
        paths[sid] = out_path
    return paths


@functools.lru_cache(maxsize=256)
def _load_npy(path: str) -> np.ndarray:
    return np.load(path)


def load_features(sid: str) -> np.ndarray:
    path = os.path.join(FEATURES_DIR, f"{sid}.npy")
    return _load_npy(path)
