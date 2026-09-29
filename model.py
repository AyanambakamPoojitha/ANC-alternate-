"""
Single-channel speech enhancement model.

Architecture: input spectrogram magnitude -> 2 GRU layers -> sigmoid mask
output, one value per frequency bin per frame. This is deliberately small
so it trains fast on limited data/time and still runs in real time on a
Jetson or laptop CPU.
"""

import torch
import torch.nn as nn


class MaskEstimator(nn.Module):
    def __init__(self, n_freq_bins: int = 257, hidden_size: int = 128, num_layers: int = 2):
        """
        n_freq_bins: number of frequency bins in the input spectrogram.
                     For a 512-point FFT this is 512//2 + 1 = 257.
        hidden_size: GRU hidden state size. Bigger = more capacity, slower.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_freq_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state=None):
        """
        x: (batch, time_frames, n_freq_bins) -- magnitude spectrogram
        hidden_state: optional GRU hidden state, for streaming/frame-by-frame
                      inference. Pass None for full-sequence training.
        Returns: mask of shape (batch, time_frames, n_freq_bins), values in [0, 1],
                 and the new hidden state (useful for real-time streaming).
        """
        gru_out, hidden_state = self.gru(x, hidden_state)
        mask = self.sigmoid(self.output_layer(gru_out))
        return mask, hidden_state
