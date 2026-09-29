"""
Trains the single-channel mask estimator.

Usage:
    python train.py --clean_dir data/clean --noise_dir data/noise --epochs 15

For a demo-quality model (not research-grade), 10-20 epochs on a small
dataset (even ~30-60 minutes of speech + a handful of noise clips) is
usually enough to hear a clear, visible improvement in the video.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import NoisySpeechDataset
from model import MaskEstimator


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = NoisySpeechDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        examples_per_epoch=args.examples_per_epoch,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = MaskEstimator(n_freq_bins=257, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for noisy_mag, ideal_mask in loader:
            noisy_mag = noisy_mag.to(device)
            ideal_mask = ideal_mask.to(device)

            predicted_mask, _ = model(noisy_mag)
            loss = loss_fn(predicted_mask, ideal_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), args.checkpoint_path)

    print(f"Saved trained model to {args.checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="data/clean")
    parser.add_argument("--noise_dir", type=str, default="data/noise")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--examples_per_epoch", type=int, default=2000)
    parser.add_argument("--checkpoint_path", type=str, default="mask_model.pt")
    args = parser.parse_args()
    train(args)
