# Elephant-AID

A deployable acoustic monitoring application for **Raspberry Pi 5** that is trained on labelled audio from area of human-wildlife conflict zones. This uses machine-learning models to 
performs real-time classification of elephants (and human/noise) sounds from a USB or
infrasonic microphone. Confirmed detections can be pushed to the field via **SMS** and
to the cloud via **ThingsBoard**, both through a Waveshare **SIM7600G-H** cellular modem.

> This repository accompanies the manuscript *"<PAPER TITLE>"* (`<Authors>`, `<Journal>`, `<Year>`).
> Please see [Citation](#citation) below.

<!-- Optional badges — update or remove as appropriate.
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)
-->

---

## Contents

- [Overview](#overview)
- [Key features](#key-features)
- [Models](#models)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Repository layout](#repository-layout)
- [Configuration reference](#configuration-reference)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

The application provides a single Tkinter desktop interface with two workflows:

1. **Training** — build a labelled dataset by recording from a microphone or importing
   `.wav` / `.mp3` files, then train and deploy a classifier.
2. **Real-time monitoring** — stream live audio, classify it on a sliding window, apply
   temporal smoothing and a confidence threshold to produce *validated* detections, and
   optionally transmit alerts by SMS and telemetry to ThingsBoard.

All processing runs locally on the device; cellular features are entirely optional and
disabled by default.

## Key features

- Four interchangeable model back-ends (see [Models](#models)).
- Handcrafted spectral feature extraction (RMS, spectral centroid/bandwidth/rolloff,
  flatness, entropy, zero-crossing rate, and seven frequency-band energies/ratios).
- Configurable sample rate, window length, overlap, and band-pass filtering, supporting
  both audible USB microphones and low-frequency / infrasonic microphones.
- Temporal smoothing across recent predictions with a user-set confidence threshold to
  reduce false positives before an alert is raised.
- Optional **SMS alerts** (cyclic or event-triggered) via SIM7600G-H in AT text mode.
- Optional **ThingsBoard cloud telemetry** over cellular HTTP, sharing the same modem
  port, with one telemetry key per class for easy dashboarding.
- Serial-port auto-detection and diagnostics for the modem.

## Models

| Model | Feature representation | Classifier | Notes |
|-------|------------------------|------------|-------|
| Random Forest | Handcrafted spectral features | `RandomForestClassifier` | Fast, CPU-only, robust baseline |
| SVM-RBF | Handcrafted spectral features | `SVC` (RBF kernel) | Strong on small datasets |
| BirdNET Embeddings + Logistic Regression | BirdNET acoustic embeddings | `LogisticRegression` | Deep embeddings; internally uses 48 kHz, 3 s windows |
| Matt-Logic EfficientNet (retrainable) | Mel-spectrogram → 224×224 image | EfficientNetB0 → TFLite | Transfer learning; exported to TensorFlow Lite for inference |

Handcrafted models use the sample rate and band-pass settings from the UI. The BirdNET
model always resamples to 48 kHz and uses 3.0 s windows internally, regardless of the UI
setting. The Matt-Logic model converts each window to a mel-spectrogram image and fine-tunes
an ImageNet-pretrained EfficientNetB0 head, then exports a `.tflite` model for live inference.

## System requirements

### Hardware
- Raspberry Pi 5 (developed and tested on this platform).
- USB microphone (audible band) and/or an infrasonic/low-frequency microphone.
- *(Optional, for alerts)* Waveshare SIM7600G-H 4G/LTE HAT or USB dongle with an active
  SIM and data plan.

### Software
- Raspberry Pi OS (64-bit) or another Debian-based Linux. Windows/macOS work for the GUI
  and training but are untested for the cellular features.
- Python 3.9 or newer.
- System packages (Debian/Raspberry Pi OS):

```bash
sudo apt update
sudo apt install -y python3-tk ffmpeg libsndfile1 portaudio19-dev
```

`python3-tk` is required for the GUI, `ffmpeg` for MP3 import, `libsndfile1` for audio
I/O, and `portaudio19-dev` for `sounddevice`.

## Installation

```bash
# 1. Clone
git clone https://github.com/anand97aakash/Elephant-AID.git
cd animal-sound-detector

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes on the heavier optional dependencies:

- **BirdNET** model: `python -m pip install birdnet` (requires `libsndfile1`).
- **Matt-Logic** model: requires `tensorflow` for training. For inference on the Pi you
  may instead install the lighter `tflite-runtime`; the app automatically prefers
  `tf.lite.Interpreter` when full TensorFlow is present.

Each model back-end is loaded lazily, so you can run the app and use the Random Forest or
SVM models without installing BirdNET or TensorFlow.

## Quick start

```bash
source venv/bin/activate
python animal_sound_detector.py
```

Then: add labels → record/import a few samples per label → **Train + Deploy Model** →
switch to **Real-time Monitoring** → **Start Monitoring**.

## Usage

### Training tab
1. Select or add a **label** (e.g. *Elephant*, *Human*, *Noise*).
2. **Record Sample** from the selected microphone, or **Import WAV/MP3 Files** /
   **Import Folder to Label**.
3. Choose a **model** and click **Train + Deploy Model**. Validation accuracy and a
   per-class report are shown in the status log.
4. Trained models are saved per type and can be reloaded or deleted.

### Real-time monitoring tab
1. Select a trained model and **Load Selected Model**, then **Start Monitoring**.
2. Live waveform, spectrum, and band-energy plots update continuously.
3. The **valid confidence threshold** and temporal smoothing determine when a prediction
   is treated as a *confirmed* detection.

### SMS alerts (optional)
- Enable **SMS transmission**, choose **Cyclic** (summary every *X* seconds) or
  **Non-cyclic** (edge-triggered on each newly confirmed detection).
- Set the recipient number, select the SIM7600 AT port (use **Auto-detect AT port** if
  unsure), and send a **Test SMS** to verify.

### ThingsBoard upload (optional)
- Enable **ThingsBoard upload**, set the host, device access token, and APN.
- Telemetry is posted over cellular HTTP through the same modem port used for SMS; a
  shared lock prevents SMS and HTTP from contending for the port.

## Repository layout

```
animal-sound-detector/
├── animal_sound_detector.py   # Main application (GUI, training, monitoring, SMS, cloud)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # License (see note below)
├── CITATION.cff               # Machine-readable citation metadata
├── .gitignore
└── data/                      # Created at runtime (not committed)
    ├── dataset/               # Per-label audio samples
    ├── models/                # Trained model artifacts
    ├── tmp/                   # Scratch files
    └── labels.json            # Active label set
```

The `data/` directory and trained model artifacts are generated at runtime and are
excluded from version control by `.gitignore`.

## Configuration reference

| Setting | Default | Description |
|---------|---------|-------------|
| Sample rate | 48000 Hz | Capture/processing rate (forced to 48 kHz for BirdNET) |
| Window sec | 3.0 | Analysis window length |
| Overlap | 0.5 | Fractional window overlap |
| Low cut / High cut | 20 / 5000 Hz | Band-pass filter; try 1–500 Hz for infrasonic mics |
| Valid confidence threshold | 60% | Minimum confidence for a valid detection |
| SMS mode | Cyclic | `Cyclic every X seconds` or `Non-cyclic on confirmed detection` |
| SMS baud | 115200 | SIM7600 serial baud rate |
| ThingsBoard interval | 10 s | Minimum spacing between telemetry uploads |

## Reproducibility

- Set random seeds are used where applicable (`random_state=42` in the scikit-learn and
  train/test-split steps).
- To reproduce a published result, pin the exact package versions you used in
  `requirements.txt` (replace the version floors with `==` pins) and record the hardware,
  microphone model, and capture settings alongside your dataset.
- We recommend archiving the trained model artifact(s) and the label set (`labels.json`)
  used to generate reported figures.

> **Authors:** please deposit the dataset (or a representative subset) in a public
> repository and add its DOI here so the results can be independently reproduced.

## Troubleshooting

- **No input device / recording fails** — check `arecord -l`; confirm the USB mic is
  connected and `portaudio19-dev` is installed.
- **MP3 import fails** — install `ffmpeg`.
- **BirdNET encoder won't load** — ensure `birdnet` and `libsndfile1` are installed.
- **Matt-Logic live monitoring error** — install `tflite-runtime` or full `tensorflow`.
- **SMS "port busy"** — a serial terminal (minicom/picocom/screen) or ModemManager may
  own the port. Use **Diagnose selected port**, close the owner, or
  `sudo systemctl stop ModemManager`.

## Citation

If you use this software, please cite both the manuscript and the software (see
`CITATION.cff`). Example:

```bibtex
@article{<KEY>,
  title   = {Elephant-AID: Automated Acoustic Interactive Device (AID) to assist in human-elephant conflict mitigation.},
  author  = {H S Sathya Chandra Sagar, Vaishakh Rao, Akash Anand, Maia Persche, Virat Rajath Nayak, Zuzana Burivalova},
  journal = {Under Review},
  year    = {<Year>},
  doi     = {<DOI>}
}
```

## License

Released under the terms in the [`LICENSE`](LICENSE) file. A permissive **MIT** license is
provided as a common default for research software — **confirm this is appropriate for
your institution and journal**, and replace it (e.g. Apache-2.0, BSD-3-Clause, or GPL-3.0)
if required.

## Acknowledgements

- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) for the acoustic embedding encoder.
- The scikit-learn, TensorFlow, librosa, and SciPy communities.
- `<Add funding sources, institutions, and collaborators here.>`
