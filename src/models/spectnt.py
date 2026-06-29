import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_utils import NUM_OUTPUTS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(-2), :]


class SpectralEncoder(nn.Module):
    def __init__(self, d_model=96, nhead=4, dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        B, C, F, T = x.shape
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(B * T, F, C)
        x = self.norm_in(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.reshape(B, T, F, C)
        x = x.permute(0, 3, 2, 1)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, d_model=96, nhead=8, dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * F, T, C)
        x = self.norm_in(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.reshape(B, F, T, C)
        x = x.permute(0, 3, 1, 2)
        return x


class SpecTNTBlock(nn.Module):
    def __init__(self, d_model=96, nhead_spectral=4, nhead_temporal=8):
        super().__init__()
        self.spectral = SpectralEncoder(d_model, nhead_spectral)
        self.temporal = TemporalEncoder(d_model, nhead_temporal)
        self.norm_spec = nn.LayerNorm(d_model)
        self.norm_temp = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, F, T = x.shape
        spec_out = self.spectral(x)
        x = self.norm_spec((x + spec_out).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        temp_out = self.temporal(x)
        x = self.norm_temp((x + temp_out).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class ResNetFrontend(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=96):
        super().__init__()
        layers = []
        cfg = [(in_ch, base_ch, 1), (base_ch, base_ch*2, 2),
               (base_ch*2, base_ch*2, 1), (base_ch*2, base_ch*4, 2),
               (base_ch*4, base_ch*4, 1), (base_ch*4, base_ch*4, 1),
               (base_ch*4, out_ch, 1)]
        for c_in, c_out, stride_f in cfg:
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=(stride_f, 1), padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class SpecTNT(nn.Module):
    def __init__(
        self,
        d_model=96,
        n_blocks=5,
        nhead_spectral=4,
        nhead_temporal=8,
    ):
        super().__init__()
        self.frontend = ResNetFrontend(in_ch=1, out_ch=d_model)
        self.blocks = nn.ModuleList([
            SpecTNTBlock(d_model, nhead_spectral, nhead_temporal)
            for _ in range(n_blocks)
        ])
        self.output = nn.Linear(d_model, NUM_OUTPUTS)

    def forward(self, x):
        x = self.frontend(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x = self.output(x)
        return x
