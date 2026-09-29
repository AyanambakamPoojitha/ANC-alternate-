"""
Live demo script: captures audio from the GRENARO mic (via the USB audio
adapter), runs it through the trained mask-estimation model frame by
frame, and plays the cleaned audio out to the earphones in real time.

Before running, find your device indices with:
    python -c "import sounddevice as sd; print(sd.query_devices())"
and set INPUT_DEVICE / OUTPUT_DEVICE below (or pass them as arguments).

Usage:
    python realtime_infer.py --input_device 1 --output_device 2
"""

import argparse

import numpy as np
import sounddevice as sd
import torch

from model import MaskEstimator

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
FRAME_SAMPLES = HOP_LENGTH  # process this many new samples per callback


class StreamingDenoiser:
    """
    Wraps the model + a rolling audio buffer so we can process the
    incoming microphone stream frame by frame instead of all at once.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MaskEstimator(n_freq_bins=N_FFT // 2 + 1)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.hidden_state = None
        # rolling buffer holds the last N_FFT samples so each new hop
        # of audio can be turned into one overlapping analysis window
        self.buffer = np.zeros(N_FFT, dtype=np.float32)
        self.window = np.hanning(N_FFT).astype(np.float32)

    def process_frame(self, new_samples: np.ndarray) -> np.ndarray:
        # slide the buffer forward and drop in the new audio
        self.buffer = np.concatenate([self.buffer[len(new_samples):], new_samples])

        windowed = self.buffer * self.window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum).astype(np.float32)
        phase = np.angle(spectrum)

        with torch.no_grad():
            mag_tensor = torch.tensor(magnitude).view(1, 1, -1).to(self.device)
            mask, self.hidden_state = self.model(mag_tensor, self.hidden_state)
            mask = mask.view(-1).cpu().numpy()

        cleaned_magnitude = magnitude * mask
        cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = np.fft.irfft(cleaned_spectrum, n=N_FFT)

        # only the newest HOP_LENGTH samples of the reconstructed window
        # correspond to genuinely new audio; return just that slice
        return cleaned_audio[-len(new_samples):].astype(np.float32)


def run(args):
    denoiser = StreamingDenoiser(args.checkpoint_path, device=args.device)

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status)
        mono_in = indata[:, 0]
        cleaned = denoiser.process_frame(mono_in)
        outdata[:, 0] = cleaned

    print("Starting real-time noise suppression. Press Ctrl+C to stop.")
    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="float32",
        channels=1,
        device=(args.input_device, args.output_device),
        callback=callback,
    ):
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="mask_model.pt")
    parser.add_argument("--input_device", type=int, default=None,
                         help="Index of the USB audio adapter's input (mic).")
    parser.add_argument("--output_device", type=int, default=None,
                         help="Index of the USB audio adapter's output (earphones).")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")
    args = parser.parse_args()
    run(args)
