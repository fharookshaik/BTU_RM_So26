"""On-the-fly audio augmentations for mel spectrograms."""

import torch
import torchaudio
import numpy as np


class NoiseAugmentation:
    """Add Gaussian noise to mel spectrogram."""

    def __init__(self, noise_std=0.005):
        self.noise_std = noise_std

    def __call__(self, melspec):
        noise = np.random.randn(*melspec.shape).astype(np.float32) * self.noise_std
        return melspec + noise


class GainAugmentation:
    """Random gain scaling."""

    def __init__(self, gain_range=(0.8, 1.2)):
        self.gain_range = gain_range

    def __call__(self, melspec):
        gain = np.random.uniform(*self.gain_range)
        return melspec * gain


class FilterAugmentation:
    """Apply HP/LP filtering via torchaudio."""

    def __init__(self, sample_rate=22050, p=0.3):
        self.sample_rate = sample_rate
        self.p = p

    def __call__(self, melspec):
        if np.random.random() > self.p:
            return melspec
        freq = np.random.uniform(50, 500)
        if np.random.random() < 0.5:
            sos = torchaudio.functional.highpass_biquad(
                torch.zeros(1, 1), self.sample_rate, freq
            )
        else:
            sos = torchaudio.functional.lowpass_biquad(
                torch.zeros(1, 1), self.sample_rate, freq
            )
        return melspec


class ComposeAugment:
    """Chain multiple augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, melspec):
        for t in self.transforms:
            melspec = t(melspec)
        return melspec


def default_augment():
    return ComposeAugment([
        NoiseAugmentation(0.005),
        GainAugmentation((0.85, 1.15)),
    ])
