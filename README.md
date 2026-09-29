# Single-Mic AI Noise Suppression — Demo Build

This is the single-channel version of your noise cancellation model, built
for the SIH video demo using one GRENARO mic. It uses the same core idea
as your proposed architecture (spectrogram → neural network → mask →
multiply → inverse STFT) — the only real change is there's no second
reference-mic channel feeding the model.

## Files

- `model.py` — the GRU mask-estimation network (small and fast, good for
  a demo and for running on modest hardware).
- `dataset.py` — mixes clean speech + noise clips on the fly to create
  training pairs. You do not need pre-recorded noisy audio.
- `train.py` — trains the model and saves a checkpoint (`mask_model.pt`).
- `realtime_infer.py` — runs the trained model live: mic in → cleaned
  audio out, for the actual demo recording.

## 1. Install requirements

```bash
pip install torch librosa soundfile sounddevice numpy --break-system-packages
```

## 2. Get training data

You need two folders of `.wav` files:

```
data/clean/   -> speech clips (mono)
data/noise/   -> noise clips (mono)
```

Fast options for a demo (not a research dataset):
- **Clean speech:** record 30-60 minutes of yourself/teammates talking,
  or download a small subset of **LibriSpeech** (openslr.org/12).
- **Noise:** record ambient noise yourself (traffic, crowd, fan, etc.)
  with a phone, or grab clips from **ESC-50** (github.com/karolpiczak/ESC-50)
  or **MUSAN** (openslr.org/17). Match noise types to your actual use case
  where possible.

All files get resampled automatically to 16kHz mono, so don't worry about
matching sample rates yourself.

## 3. Train

```bash
python train.py --clean_dir data/clean --noise_dir data/noise --epochs 15
```

- On a laptop GPU: likely well under an hour for a demo-quality result.
- On CPU only: slower, but still workable overnight with a small dataset.
- Watch the printed loss go down each epoch — it won't hit zero, that's normal.
- If the demo audio still sounds too noisy, try more epochs, more noise
  variety, or a slightly larger `--hidden_size`.

## 4. Find your audio device indices

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Find the USB audio adapter in the list — note its input index (mic in)
and output index (earphones out). They may be the same device.

## 5. Run the live demo

```bash
python realtime_infer.py --input_device <IN_INDEX> --output_device <OUT_INDEX>
```

Speak into the GRENARO mic with background noise present — the output to
your earphones should have the noise audibly reduced. This is what you
record for the video.

## Notes for your presentation

- This single-mic model is the PoC / demo version. Say so plainly if asked.
- The proposed final architecture (dual-mic, in your idea submission)
  uses a reference mic as an extra hint to the network — this version
  has to recognize noise purely from spectral pattern, which is a well
  established approach (the same idea behind tools like RNNoise).
- If time is very short before the video, you can lower `--epochs` and
  use a smaller dataset just to get a visibly-working demo; perfect audio
  quality is not the goal here, showing the pipeline and concept working is.
