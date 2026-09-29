"""
Builds (noisy, clean) training pairs by mixing clean speech clips with
noise clips at a random loudness ratio (SNR). Nothing here needs a
pre-recorded "noisy" dataset -- the mixing happens in code.

Folder layout expected:
    data/clean/*.wav   -> clean speech clips (any length, mono, any sample rate;
                           will be resampled to SAMPLE_RATE)
    data/noise/*.wav   -> noise clips (crowd, traffic, wind, gunfire, etc.)
"""

import glob
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000       # standard for speech models; keeps things small/fast
N_FFT = 512               # FFT size -> 257 frequency bins
HOP_LENGTH = 128           # 8ms hop at 16kHz; frame length 32ms with 50% overlap-ish
CLIP_SECONDS = 3.0        # length of each training example
SNR_RANGE_DB = (-5, 15)   # randomly mix noise louder or quieter than speech


def _load_random_segment(path: str, n_samples: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < n_samples:
        # pad short clips by repeating them
        repeats = int(np.ceil(n_samples / max(len(audio), 1)))
        audio = np.tile(audio, repeats)
    start = random.randint(0, max(len(audio) - n_samples, 0))
    return audio[start:start + n_samples]


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = np.mean(speech ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return speech + noise * scale


class NoisySpeechDataset(Dataset):
    def __init__(self, clean_dir: str, noise_dir: str, examples_per_epoch: int = 2000):
        self.clean_files = sorted(glob.glob(f"{clean_dir}/*.wav"))
        self.noise_files = sorted(glob.glob(f"{noise_dir}/*.wav"))
        if not self.clean_files or not self.noise_files:
            raise RuntimeError(
                "No .wav files found. Check clean_dir and noise_dir paths."
            )
        self.n_samples = int(CLIP_SECONDS * SAMPLE_RATE)
        self.examples_per_epoch = examples_per_epoch

    def __len__(self):
        return self.examples_per_epoch

    def _spectrogram(self, audio: np.ndarray) -> np.ndarray:
        spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        return np.abs(spec).T  # (time_frames, n_freq_bins)

    def __getitem__(self, idx):
        clean_path = random.choice(self.clean_files)
        noise_path = random.choice(self.noise_files)

        clean = _load_random_segment(clean_path, self.n_samples)
        noise = _load_random_segment(noise_path, self.n_samples)

        snr_db = random.uniform(*SNR_RANGE_DB)
        noisy = _mix_at_snr(clean, noise, snr_db)

        clean_mag = self._spectrogram(clean)
        noisy_mag = self._spectrogram(noisy)

        # Ideal ratio mask: what the model is trained to predict.
        # Clipped to [0, 1] since the network's sigmoid output can't exceed that.
        ideal_mask = np.clip(clean_mag / (noisy_mag + 1e-8), 0.0, 1.0)

        return (
            torch.tensor(noisy_mag, dtype=torch.float32),
            torch.tensor(ideal_mask, dtype=torch.float32),
        )
