import torch
import torch.nn as nn

from src.data_utils import NUM_OUTPUTS


class HarmonicCNN(nn.Module):
    """
    Harmonic-CNN instant model.
    Predicts 1 boundary + 7 function values for the center frame of a 24s chunk.

    Architecture (following Won et al. 2020 + Wang et al. 2022):
      - 7x Conv2D blocks with batch norm, ReLU
      - Frequency pooling (2,) on early layers to compress mel bands
      - Temporal dimension preserved through convs, then center-frame extracted
      - 2 dense layers → 8 outputs
    """
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            self._conv_block(1, 32, pool=(2, 1)),
            self._conv_block(32, 64, pool=(2, 1)),
            self._conv_block(64, 128, pool=(2, 1)),
            self._conv_block(128, 128, pool=(2, 1)),
            self._conv_block(128, 256, pool=(2, 1)),
            self._conv_block(256, 256, pool=(1, 1)),
            self._conv_block(256, 512, pool=(1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_OUTPUTS),
        )

    def _conv_block(self, in_ch, out_ch, pool):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x
