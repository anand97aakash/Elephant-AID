#!/usr/bin/env python3
"""
Animal Sound Detector Mic App v5 - RPi5 USB/Infrasonic Mic ML UI + SIM7600 SMS

Adds BirdNET embeddings + custom Elephant/Human/Noise classifier as a third model.

Models:
1) Random Forest on handcrafted audio features
2) SVM-RBF on handcrafted audio features
3) BirdNET Embeddings + Logistic Regression

Notes for BirdNET mode:
- BirdNET encoder is loaded lazily and only when needed.
- BirdNET mode internally uses 48 kHz and 3.0-second windows for training/inference.
- For live prediction, the latest audio window is written to a temporary WAV clip and encoded
  via BirdNET, then passed through the custom classifier.
"""

import os
import json
import time
import math
import queue
import shutil
import tempfile
import threading
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter, deque

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except Exception as e:
    serial = None
    list_ports = None
    SERIAL_IMPORT_ERROR = e
else:
    SERIAL_IMPORT_ERROR = None

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    SOUNDDEVICE_ERROR = e
else:
    SOUNDDEVICE_ERROR = None

try:
    from pydub import AudioSegment
except Exception as e:
    AudioSegment = None
    PYDUB_ERROR = e
else:
    PYDUB_ERROR = None

try:
    import librosa
except Exception as e:
    librosa = None
    LIBROSA_ERROR = e
else:
    LIBROSA_ERROR = None

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    TFLITE_IMPORT_ERROR = None
    TFLITE_RUNTIME_BACKEND = "tflite_runtime"
except Exception as e:
    TFLiteInterpreter = None
    TFLITE_IMPORT_ERROR = e
    TFLITE_RUNTIME_BACKEND = None

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    TF_IMPORT_ERROR = e
else:
    TF_IMPORT_ERROR = None

# TensorFlow 2.21 still provides tf.lite.Interpreter, but importing
# Interpreter directly from tensorflow.lite may fail on some builds.
if TFLiteInterpreter is None and tf is not None:
    try:
        TFLiteInterpreter = tf.lite.Interpreter
        TFLITE_IMPORT_ERROR = None
        TFLITE_RUNTIME_BACKEND = "tensorflow_lite"
    except Exception as e:
        TFLiteInterpreter = None
        TFLITE_IMPORT_ERROR = e
        TFLITE_RUNTIME_BACKEND = None

try:
    import birdnet as birdnet_lib
except Exception as e:
    birdnet_lib = None
    BIRDNET_IMPORT_ERROR = e
else:
    BIRDNET_IMPORT_ERROR = None

from scipy.io import wavfile
from scipy import signal

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


APP_NAME = "Animal Sound Detector Mic ML App v5 + Matt Logic + SIM7600 SMS"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = DATA_DIR / "models"
TMP_DIR = DATA_DIR / "tmp"
LABELS_FILE = DATA_DIR / "labels.json"
DEPLOYED_MODEL = MODEL_DIR / "deployed_model.joblib"
MATT_MODEL_DIR = PROJECT_DIR / "matt_model"
MATT_RETRAIN_TFLITE = MODEL_DIR / "matt_logic_retrain_model.tflite"
MATT_RETRAIN_LABELS = MODEL_DIR / "matt_logic_retrain_labels.txt"
MATT_IMAGE_SIZE = 224
MATT_RETRAIN_EPOCHS = 6
MATT_RETRAIN_BATCH_SIZE = 16

DEFAULT_LABELS = ["Elephant", "Dog", "Cow", "Bird", "Human", "Noise"]
SUPPORTED_AUDIO_EXTS = (".wav", ".mp3")

MODEL_HANDCRAFT_RF = "Random Forest"
MODEL_HANDCRAFT_SVM = "SVM-RBF"
MODEL_BIRDNET = "BirdNET Embeddings + Logistic Regression"
MODEL_MATT_RETRAIN = "Matt Logic EfficientNet Retrainable (TensorFlow)"
MODEL_OPTIONS = [MODEL_HANDCRAFT_RF, MODEL_HANDCRAFT_SVM, MODEL_BIRDNET, MODEL_MATT_RETRAIN]

LATEST_RF_MODEL = MODEL_DIR / "random_forest_latest.joblib"
LATEST_SVM_MODEL = MODEL_DIR / "svm_rbf_latest.joblib"
LATEST_BIRDNET_MODEL = MODEL_DIR / "birdnet_embeddings_latest.joblib"
LATEST_MATT_LOGIC_MODEL = MODEL_DIR / "matt_logic_latest.joblib"
MODEL_LATEST_PATHS = {
    MODEL_HANDCRAFT_RF: LATEST_RF_MODEL,
    MODEL_HANDCRAFT_SVM: LATEST_SVM_MODEL,
    MODEL_BIRDNET: LATEST_BIRDNET_MODEL,
    MODEL_MATT_RETRAIN: LATEST_MATT_LOGIC_MODEL,
}

SMS_MODE_CYCLIC = "Cyclic every X seconds"
SMS_MODE_EVENT = "Non-cyclic on confirmed detection"
SMS_MODES = [SMS_MODE_CYCLIC, SMS_MODE_EVENT]
DEFAULT_SMS_BAUD = 115200
DEFAULT_SMS_PORT_CANDIDATES = ("/dev/ttyUSB2", "/dev/ttyUSB3", "/dev/ttyUSB1", "/dev/ttyUSB0")
SMS_CTRL_Z = b"\x1A"

# ThingsBoard cloud upload settings for Waveshare SIM7600G-H.
# Cloud upload is disabled by default so existing prediction + SMS behavior is unchanged.
DEFAULT_TB_HOST = "thingsboard.cloud"
DEFAULT_TB_APN = "airtelgprs.com"
DEFAULT_TB_UPLOAD_INTERVAL_SEC = 10.0
DEFAULT_TB_HTTP_TIMEOUT_MS = 10000


# Realtime monitoring stability settings for Raspberry Pi 5.
PLOT_INTERVAL_SEC = 1.0
MONITOR_LOOP_MS = 120
MAX_AUDIO_QUEUE_BLOCKS = 30
MAX_BLOCKS_PER_UI_TICK = 12
MAX_PLOT_POINTS = 2500

BIRDNET_MONITOR_PREDICT_SEC = 3.0
BIRDNET_MONITOR_PLOT_SEC = 2.0

BIRDNET_TARGET_SR = 48000
BIRDNET_TARGET_WIN_SEC = 3.0
BIRDNET_BACKEND_CANDIDATES = ("tf", "pb")

FREQ_BANDS = [
    (0, 20, "0-20"),
    (20, 80, "20-80"),
    (80, 250, "80-250"),
    (250, 500, "250-500"),
    (500, 1000, "500-1k"),
    (1000, 3000, "1k-3k"),
    (3000, 8000, "3k-8k"),
]

FEATURE_NAMES = [
    "rms", "log_rms", "peak", "mean_abs", "std", "crest_factor", "zcr",
    "spec_centroid", "spec_bandwidth", "spec_rolloff85", "spec_flatness",
    "spec_entropy", "peak_freq", "dominant_power_ratio",
    "band_0_20", "band_20_80", "band_80_250", "band_250_500",
    "band_500_1000", "band_1000_3000", "band_3000_8000",
    "ratio_0_20", "ratio_20_80", "ratio_80_250", "ratio_250_500",
    "ratio_500_1000", "ratio_1000_3000", "ratio_3000_8000",
]


def safe_label(label: str) -> str:
    label = str(label).strip()
    cleaned = "".join(c if c.isalnum() or c in "_- " else "_" for c in label)
    cleaned = cleaned.replace(" ", "_").strip("_")
    return cleaned or "Unknown"


def now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    MATT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_FILE.exists():
        LABELS_FILE.write_text(json.dumps({"labels": DEFAULT_LABELS}, indent=2), encoding="utf-8")

    for label in load_labels():
        (DATASET_DIR / safe_label(label)).mkdir(parents=True, exist_ok=True)


def load_labels():
    try:
        data = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        labels = data.get("labels", DEFAULT_LABELS)
        labels = [str(x).strip() for x in labels if str(x).strip()]
        return labels if labels else DEFAULT_LABELS
    except Exception:
        return DEFAULT_LABELS


def save_labels(labels):
    labels = list(dict.fromkeys([str(x).strip() for x in labels if str(x).strip()]))
    LABELS_FILE.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
    for label in labels:
        (DATASET_DIR / safe_label(label)).mkdir(parents=True, exist_ok=True)


def to_float_audio(x):
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / max(1.0, float(np.iinfo(x.dtype).max))
    else:
        x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1.0, 1.0)


def write_wav(path: Path, sample_rate: int, audio_float):
    audio_float = np.asarray(audio_float, dtype=np.float32)
    audio_int16 = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
    wavfile.write(str(path), int(sample_rate), audio_int16)


def read_mp3_with_pydub(path: Path):
    if AudioSegment is None:
        raise RuntimeError(
            "MP3 support requires pydub + ffmpeg. Install with:\n"
            "sudo apt install -y ffmpeg\n"
            "source venv/bin/activate\n"
            "python -m pip install pydub"
        )
    audio = AudioSegment.from_file(str(path), format="mp3")
    audio = audio.set_channels(1)
    sr = int(audio.frame_rate)
    samples = np.array(audio.get_array_of_samples())

    if audio.sample_width == 1:
        scale = float(2 ** 7)
    elif audio.sample_width == 2:
        scale = float(2 ** 15)
    elif audio.sample_width == 3:
        scale = float(2 ** 23)
    else:
        scale = float(2 ** 31)

    x = samples.astype(np.float32) / scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return sr, np.clip(x, -1.0, 1.0)


def read_audio_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".wav":
        sr, data = wavfile.read(str(path))
        return int(sr), to_float_audio(data)
    if ext == ".mp3":
        return read_mp3_with_pydub(path)
    raise ValueError(f"Unsupported audio file: {path.name}")


def resample_if_needed(audio, old_sr, new_sr):
    old_sr = int(old_sr)
    new_sr = int(new_sr)
    if old_sr == new_sr:
        return audio.astype(np.float32)
    gcd = int(np.gcd(old_sr, new_sr))
    up = new_sr // gcd
    down = old_sr // gcd
    y = signal.resample_poly(audio, up, down)
    return y.astype(np.float32)


def apply_filter(audio, sr, low_cut, high_cut):
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < 32:
        return x

    x = x - float(np.mean(x))
    nyq = sr / 2.0
    low = max(0.0, float(low_cut))
    high = min(float(high_cut), nyq * 0.95)

    if high <= low:
        return x

    try:
        if low <= 0.0 and high >= nyq * 0.94:
            return x
        if low <= 0.0:
            sos = signal.butter(4, high / nyq, btype="lowpass", output="sos")
        elif high >= nyq * 0.94:
            sos = signal.butter(4, low / nyq, btype="highpass", output="sos")
        else:
            sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        try:
            return signal.sosfiltfilt(sos, x).astype(np.float32)
        except Exception:
            return signal.sosfilt(sos, x).astype(np.float32)
    except Exception:
        return x


def zero_crossing_rate(x):
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(x)) != 0))


def spectral_entropy(power):
    power = np.asarray(power, dtype=np.float64)
    total = float(np.sum(power)) + 1e-12
    p = power / total
    ent = -float(np.sum(p * np.log2(p + 1e-12)))
    return ent / float(np.log2(len(p) + 1e-12))


def band_power(freqs, power, f1, f2):
    mask = (freqs >= f1) & (freqs < f2)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def compute_spectrum(audio, sr):
    x = np.asarray(audio, dtype=np.float32)
    if len(x) == 0:
        return np.array([0.0]), np.array([0.0])
    n = len(x)
    win = np.hanning(n).astype(np.float32)
    spec = np.fft.rfft(x * win)
    mag = np.abs(spec).astype(np.float64)
    power = mag * mag
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return freqs, power


def extract_features(audio, sr, low_cut, high_cut):
    x = np.asarray(audio, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if len(x) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    xf = apply_filter(x, sr, low_cut, high_cut)

    rms = float(np.sqrt(np.mean(xf * xf) + 1e-12))
    peak = float(np.max(np.abs(xf)) + 1e-12)
    mean_abs = float(np.mean(np.abs(xf)))
    std = float(np.std(xf))
    crest = float(peak / (rms + 1e-12))
    zcr = zero_crossing_rate(xf)

    freqs, power = compute_spectrum(xf, sr)
    mag = np.sqrt(power)

    nyq = sr / 2.0
    low = max(0.0, float(low_cut))
    high = min(float(high_cut), nyq * 0.95)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)

    f = freqs[mask]
    p = power[mask]
    m = mag[mask]

    total_power = float(np.sum(p)) + 1e-12
    centroid = float(np.sum(f * p) / total_power)
    bandwidth = float(np.sqrt(np.sum(((f - centroid) ** 2) * p) / total_power))

    cumsum = np.cumsum(p)
    roll_idx = int(np.searchsorted(cumsum, 0.85 * total_power))
    roll_idx = min(max(roll_idx, 0), len(f) - 1)
    rolloff85 = float(f[roll_idx])

    flatness = float(np.exp(np.mean(np.log(m + 1e-12))) / (np.mean(m) + 1e-12))
    entropy = spectral_entropy(p)
    peak_idx = int(np.argmax(p)) if len(p) else 0
    peak_freq = float(f[peak_idx]) if len(f) else 0.0
    dom_ratio = float(np.max(p) / total_power) if len(p) else 0.0

    full_power = float(np.sum(power)) + 1e-12
    band_logs = []
    band_ratios = []
    for f1, f2, _name in FREQ_BANDS:
        bp = band_power(freqs, power, f1, min(f2, nyq))
        band_logs.append(float(np.log1p(bp)))
        band_ratios.append(float(bp / full_power))

    vals = [
        rms, np.log1p(rms), peak, mean_abs, std, crest, zcr,
        centroid, bandwidth, rolloff85, flatness, entropy, peak_freq, dom_ratio,
        *band_logs,
        *band_ratios,
    ]
    return np.asarray(vals, dtype=np.float32)


def split_windows(audio, sr, window_sec, overlap):
    win_len = max(128, int(sr * float(window_sec)))
    overlap = min(max(float(overlap), 0.0), 0.9)
    hop = max(1, int(win_len * (1.0 - overlap)))

    if len(audio) < win_len:
        out = np.zeros(win_len, dtype=np.float32)
        out[:len(audio)] = audio
        return [out]

    return [audio[start:start + win_len] for start in range(0, len(audio) - win_len + 1, hop)]


def parse_device_index(display_text):
    if not display_text or display_text.startswith("Default"):
        return None
    try:
        return int(display_text.split(":", 1)[0])
    except Exception:
        return None


def list_input_devices():
    out = ["Default input device"]
    if sd is None:
        return out
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) > 0:
                out.append(f"{idx}: {dev.get('name', f'Device {idx}')}")
    except Exception:
        pass
    return out


def list_serial_ports_for_sms():
    """Return likely modem AT serial ports for the SIM7600/SIM7600G-H USB dongle."""
    ports = []
    if list_ports is not None:
        try:
            for p in list_ports.comports():
                label = f"{p.device}"
                desc = (p.description or "").strip()
                if desc and desc.lower() != "n/a":
                    label = f"{p.device}  ({desc})"
                ports.append(label)
        except Exception:
            pass

    existing = {x.split()[0] for x in ports}
    for cand in DEFAULT_SMS_PORT_CANDIDATES:
        if Path(cand).exists() and cand not in existing:
            ports.append(cand)

    if not ports:
        ports = list(DEFAULT_SMS_PORT_CANDIDATES)
    return ports


def parse_serial_port(display_text):
    txt = str(display_text or "").strip()
    if not txt:
        return ""
    return txt.split()[0]


def audio_files_in_folder(folder: Path):
    files = []
    for ext in SUPPORTED_AUDIO_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
        files.extend(folder.rglob(f"*{ext.upper()}"))
    seen = set()
    result = []
    for f in files:
        try:
            key = f.resolve()
        except Exception:
            key = f
        if key not in seen and f.is_file():
            seen.add(key)
            result.append(f)
    return sorted(result)


def _coerce_numeric_array(obj):
    """Best-effort conversion of BirdNET encode output to numeric ndarray."""
    if obj is None:
        raise RuntimeError("BirdNET encode returned None.")

    # DataFrame-like
    if hasattr(obj, "select_dtypes"):
        try:
            num = obj.select_dtypes(include=[np.number])
            arr = num.to_numpy()
            if arr.size > 0:
                return np.asarray(arr, dtype=np.float32)
        except Exception:
            pass

    # If object has embeddings attribute
    for attr in ("embeddings", "embedding", "data", "values"):
        if hasattr(obj, attr):
            try:
                arr = np.asarray(getattr(obj, attr))
                if arr.size > 0:
                    return arr.astype(np.float32, copy=False)
            except Exception:
                pass

    if hasattr(obj, "to_numpy"):
        try:
            arr = obj.to_numpy()
            if np.asarray(arr).size > 0:
                return np.asarray(arr, dtype=np.float32)
        except Exception:
            pass

    arr = np.asarray(obj)
    if arr.size == 0:
        raise RuntimeError("BirdNET encode output had no numeric values.")

    # Object dtype fallback: keep only numeric-looking values
    if arr.dtype == object:
        flat = []
        for v in arr.ravel():
            try:
                flat.append(float(v))
            except Exception:
                continue
        if not flat:
            raise RuntimeError("BirdNET encode output could not be converted to numeric embedding values.")
        arr = np.asarray(flat, dtype=np.float32)

    return arr.astype(np.float32, copy=False)


class App:
    def __init__(self, root):
        ensure_dirs()

        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1340x880")
        self.root.minsize(1140, 760)

        self.labels = load_labels()

        self.sample_rate_var = tk.IntVar(value=48000)
        self.window_sec_var = tk.DoubleVar(value=3.0)
        self.overlap_var = tk.DoubleVar(value=0.5)
        self.low_cut_var = tk.DoubleVar(value=20.0)
        self.high_cut_var = tk.DoubleVar(value=5000.0)
        self.record_sec_var = tk.DoubleVar(value=10.0)
        self.model_type_var = tk.StringVar(value=MODEL_HANDCRAFT_RF)
        self.device_var = tk.StringVar(value="Default input device")
        self.label_var = tk.StringVar(value=self.labels[0] if self.labels else "Elephant")
        self.monitor_model_var = tk.StringVar(value="")

        # v5 live-validity and SIM7600 SMS settings.
        # Threshold accepts either 0.60 or 60 in the UI.
        self.conf_threshold_var = tk.DoubleVar(value=60.0)
        self.sms_enabled_var = tk.BooleanVar(value=False)
        self.sms_mode_var = tk.StringVar(value=SMS_MODE_CYCLIC)
        self.sms_interval_sec_var = tk.DoubleVar(value=60.0)
        self.sms_phone_var = tk.StringVar(value="")
        self.sms_port_var = tk.StringVar(value=list_serial_ports_for_sms()[0])
        self.sms_baud_var = tk.IntVar(value=DEFAULT_SMS_BAUD)
        self.sms_include_conf_var = tk.BooleanVar(value=True)

        # ThingsBoard telemetry upload settings. Uses the same SIM7600 AT port/baud
        # selected above for SMS, but it has its own master enable checkbox.
        self.tb_enabled_var = tk.BooleanVar(value=False)
        self.tb_host_var = tk.StringVar(value=DEFAULT_TB_HOST)
        self.tb_token_var = tk.StringVar(value="")
        self.tb_apn_var = tk.StringVar(value=DEFAULT_TB_APN)
        self.tb_upload_interval_sec_var = tk.DoubleVar(value=DEFAULT_TB_UPLOAD_INTERVAL_SEC)
        self.tb_upload_valid_only_var = tk.BooleanVar(value=False)

        self.status_q = queue.Queue()

        self.model_package = None
        self.pipeline = None
        self.birdnet_encoder = None
        self.birdnet_backend = None
        self.matt_retrain_interpreter = None
        self.matt_retrain_input_details = None
        self.matt_retrain_output_details = None
        self.matt_retrain_labels = None

        self.birdnet_live_wav = TMP_DIR / "birdnet_live.wav"

        self.monitor_q = queue.Queue(maxsize=MAX_AUDIO_QUEUE_BLOCKS)
        self.monitor_stream = None
        self.monitor_running = False
        self.monitor_audio = np.zeros(1, dtype=np.float32)
        self.last_plot = 0.0
        self.last_predict = 0.0
        self.prediction_history = deque(maxlen=5)

        self.sms_lock = threading.Lock()
        self.sms_send_running = False
        self.sms_thread = None
        self.sms_send_started_at = 0.0
        self.sms_last_error = ""
        self.last_cyclic_sms_time = 0.0
        self.cyclic_detection_stats = {}
        self.last_event_sms_signature = None

        self.tb_upload_running = False
        self.tb_thread = None
        self.tb_upload_started_at = 0.0
        self.tb_last_upload_time = 0.0
        self.tb_last_error = ""

        self.build_ui()
        self.refresh_dataset()
        self.load_model(show_popup=False)
        self.refresh_monitor_models()

        self.root.after(100, self.process_status)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------- UI ----------------
    def build_ui(self):
        self.build_settings_bar()
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.train_page = ttk.Frame(nb)
        self.monitor_page = ttk.Frame(nb)
        nb.add(self.train_page, text="Training")
        nb.add(self.monitor_page, text="Real-time Monitoring")

        self.build_training_page()
        self.build_monitoring_page()

    def build_settings_bar(self):
        box = ttk.LabelFrame(self.root, text="Mic / Processing Settings")
        box.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(box, text="Input device").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        self.device_combo = ttk.Combobox(
            box, textvariable=self.device_var, values=list_input_devices(), width=50, state="readonly"
        )
        self.device_combo.grid(row=0, column=1, padx=6, pady=4, sticky="w")
        ttk.Button(box, text="Refresh devices", command=self.refresh_devices).grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(box, text="Sample rate").grid(row=0, column=3, padx=6, pady=4)
        ttk.Entry(box, textvariable=self.sample_rate_var, width=10).grid(row=0, column=4, padx=6, pady=4)

        ttk.Label(box, text="Window sec").grid(row=0, column=5, padx=6, pady=4)
        ttk.Entry(box, textvariable=self.window_sec_var, width=8).grid(row=0, column=6, padx=6, pady=4)

        ttk.Label(box, text="Overlap").grid(row=0, column=7, padx=6, pady=4)
        ttk.Entry(box, textvariable=self.overlap_var, width=8).grid(row=0, column=8, padx=6, pady=4)

        ttk.Label(box, text="Low cut Hz").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(box, textvariable=self.low_cut_var, width=10).grid(row=1, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(box, text="High cut Hz").grid(row=1, column=2, padx=6, pady=4, sticky="w")
        ttk.Entry(box, textvariable=self.high_cut_var, width=10).grid(row=1, column=3, padx=6, pady=4, sticky="w")

        hint = (
            "USB mic: 20-5000 Hz | Infrasonic mic later: try 1-500 Hz or 1-1000 Hz | "
            "BirdNET mode internally uses 48 kHz + 3 sec windows"
        )
        ttk.Label(box, text=hint).grid(row=1, column=4, columnspan=5, padx=6, pady=4, sticky="w")

    def build_training_page(self):
        left = ttk.Frame(self.train_page)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        right = ttk.Frame(self.train_page)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        record_box = ttk.LabelFrame(left, text="1) Record / Import Samples")
        record_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(record_box, text="Label").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.label_combo = ttk.Combobox(
            record_box, textvariable=self.label_var, values=self.labels, width=24, state="readonly"
        )
        self.label_combo.grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(record_box, text="New label").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.new_label_var = tk.StringVar()
        ttk.Entry(record_box, textvariable=self.new_label_var, width=26).grid(row=1, column=1, padx=6, pady=6)
        ttk.Button(record_box, text="Add", command=self.add_label).grid(row=1, column=2, padx=6, pady=6)

        ttk.Label(record_box, text="Record sec").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(record_box, textvariable=self.record_sec_var, width=10).grid(row=2, column=1, padx=6, pady=6, sticky="w")

        ttk.Button(record_box, text="Record Sample", command=self.record_sample).grid(row=3, column=0, padx=6, pady=8, sticky="ew")
        ttk.Button(record_box, text="Import WAV/MP3 Files", command=self.import_audio_files).grid(row=3, column=1, padx=6, pady=8, sticky="ew")
        ttk.Button(record_box, text="Import Folder to Label", command=self.import_audio_folder).grid(row=4, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(record_box, text="Open Dataset Folder", command=self.open_dataset).grid(row=4, column=1, padx=6, pady=6, sticky="ew")

        train_box = ttk.LabelFrame(left, text="2) Train / Deploy Model")
        train_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(train_box, text="Model").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Combobox(
            train_box, textvariable=self.model_type_var, values=MODEL_OPTIONS, state="readonly", width=34
        ).grid(row=0, column=1, padx=6, pady=6)

        ttk.Button(train_box, text="Train + Deploy Model", command=self.train_model).grid(row=1, column=0, columnspan=2, padx=6, pady=8, sticky="ew")
        ttk.Button(train_box, text="Load Deployed Model", command=lambda: self.load_model(True)).grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="ew")
        ttk.Button(train_box, text="Delete Trained Model", command=self.delete_trained_model).grid(row=3, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        self.train_status_var = tk.StringVar(value="Training status: waiting")
        ttk.Label(train_box, textvariable=self.train_status_var, wraplength=360).grid(row=4, column=0, columnspan=2, padx=6, pady=6, sticky="w")

        data_box = ttk.LabelFrame(left, text="3) Label Dataset Management")
        data_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(data_box, text="Uses selected label above").grid(row=0, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        ttk.Button(data_box, text="Delete Selected Label Dataset", command=self.delete_selected_label_dataset).grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="ew")
        ttk.Label(data_box, text="After deletion, you can retrain immediately when prompted.", wraplength=340).grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="w")

        info = ttk.LabelFrame(left, text="Field Data Tip")
        info.pack(fill=tk.X)
        msg = (
            "Supported training files: WAV and MP3.\n\n"
            "Model options:\n"
            "- Random Forest / SVM-RBF = handcrafted spectral features\n"
            "- BirdNET Embeddings + Logistic Regression = deep embeddings + custom classifier\n"
            "- Matt Logic EfficientNet Retrainable (TensorFlow) = mel-spectrogram image pipeline + retrainable classifier\n\n"
            "BirdNET mode requires 'birdnet' to be installed in your venv.\n"
            "Matt Logic mode requires 'librosa', TFLite runtime, and TensorFlow."
        )
        ttk.Label(info, text=msg, wraplength=360).pack(padx=8, pady=8)

        dataset_box = ttk.LabelFrame(right, text="Dataset Summary")
        dataset_box.pack(fill=tk.BOTH, expand=True)

        columns = ("label", "files", "duration")
        self.dataset_tree = ttk.Treeview(dataset_box, columns=columns, show="headings", height=12)
        self.dataset_tree.heading("label", text="Label")
        self.dataset_tree.heading("files", text="Audio files")
        self.dataset_tree.heading("duration", text="Approx duration")
        self.dataset_tree.column("label", width=220)
        self.dataset_tree.column("files", width=120, anchor="center")
        self.dataset_tree.column("duration", width=160, anchor="center")
        self.dataset_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        ttk.Button(dataset_box, text="Refresh Dataset", command=self.refresh_dataset).pack(anchor="e", padx=6, pady=6)

        self.log_box = tk.Text(right, height=14, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self.log(f"Project dir: {PROJECT_DIR}")
        self.log(f"Dataset dir: {DATASET_DIR}")
        self.log(f"Model path: {DEPLOYED_MODEL}")

    def build_monitoring_page(self):
        top = ttk.Frame(self.monitor_page)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Monitor model").pack(side=tk.LEFT, padx=(4, 2))
        self.monitor_model_combo = ttk.Combobox(
            top,
            textvariable=self.monitor_model_var,
            values=self.available_monitor_model_names(),
            width=34,
            state="readonly",
        )
        self.monitor_model_combo.pack(side=tk.LEFT, padx=4)

        ttk.Button(top, text="Refresh trained models", command=self.refresh_monitor_models).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load Selected Model", command=self.load_selected_monitor_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load Last Trained", command=lambda: self.load_model(True)).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Start Monitoring", command=self.start_monitoring).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Stop Monitoring", command=self.stop_monitoring).pack(side=tk.LEFT, padx=4)

        self.model_status_var = tk.StringVar(value="Model: not loaded")
        ttk.Label(top, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=14)

        self.monitor_status_var = tk.StringVar(value="Monitoring: stopped")
        ttk.Label(top, textvariable=self.monitor_status_var).pack(side=tk.LEFT, padx=14)

        alert_box = ttk.LabelFrame(self.monitor_page, text="v5 Detection Validity + SIM7600 SMS Alerts")
        alert_box.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(alert_box, text="Valid confidence threshold (%)").grid(row=0, column=0, padx=6, pady=5, sticky="w")
        ttk.Entry(alert_box, textvariable=self.conf_threshold_var, width=8).grid(row=0, column=1, padx=6, pady=5, sticky="w")
        ttk.Label(alert_box, text="Prediction is considered valid only at/above this threshold.").grid(row=0, column=2, padx=6, pady=5, sticky="w")

        ttk.Checkbutton(alert_box, text="Enable SMS transmission", variable=self.sms_enabled_var).grid(row=1, column=0, padx=6, pady=5, sticky="w")
        ttk.Radiobutton(alert_box, text="Cyclic SMS", variable=self.sms_mode_var, value=SMS_MODE_CYCLIC).grid(row=1, column=1, padx=6, pady=5, sticky="w")
        ttk.Radiobutton(alert_box, text="Non-cyclic SMS", variable=self.sms_mode_var, value=SMS_MODE_EVENT).grid(row=1, column=2, padx=6, pady=5, sticky="w")

        ttk.Label(alert_box, text="Cyclic interval sec").grid(row=2, column=0, padx=6, pady=5, sticky="w")
        ttk.Entry(alert_box, textvariable=self.sms_interval_sec_var, width=8).grid(row=2, column=1, padx=6, pady=5, sticky="w")
        ttk.Checkbutton(alert_box, text="Include confidence in SMS", variable=self.sms_include_conf_var).grid(row=2, column=2, padx=6, pady=5, sticky="w")

        ttk.Label(alert_box, text="Recipient phone").grid(row=3, column=0, padx=6, pady=5, sticky="w")
        ttk.Entry(alert_box, textvariable=self.sms_phone_var, width=22).grid(row=3, column=1, padx=6, pady=5, sticky="w")
        ttk.Label(alert_box, text="SIM7600 AT port").grid(row=3, column=2, padx=6, pady=5, sticky="e")
        self.sms_port_combo = ttk.Combobox(alert_box, textvariable=self.sms_port_var, values=list_serial_ports_for_sms(), width=34)
        self.sms_port_combo.grid(row=3, column=3, padx=6, pady=5, sticky="w")
        ttk.Button(alert_box, text="Refresh ports", command=self.refresh_sms_ports).grid(row=3, column=4, padx=6, pady=5)

        ttk.Label(alert_box, text="Baud").grid(row=4, column=0, padx=6, pady=5, sticky="w")
        ttk.Entry(alert_box, textvariable=self.sms_baud_var, width=10).grid(row=4, column=1, padx=6, pady=5, sticky="w")
        ttk.Button(alert_box, text="Send Test SMS", command=self.send_test_sms).grid(row=4, column=2, padx=6, pady=5, sticky="w")
        ttk.Button(alert_box, text="Auto-detect AT port", command=self.auto_detect_sms_port).grid(row=4, column=3, padx=6, pady=5, sticky="w")
        ttk.Button(alert_box, text="Diagnose selected port", command=self.diagnose_selected_sms_port).grid(row=4, column=4, padx=6, pady=5, sticky="w")
        ttk.Button(alert_box, text="Reset SMS busy", command=self.reset_sms_busy).grid(row=5, column=0, padx=6, pady=5, sticky="w")
        self.sms_status_var = tk.StringVar(value="SMS: disabled")
        ttk.Label(alert_box, textvariable=self.sms_status_var, wraplength=780).grid(row=5, column=1, columnspan=4, padx=6, pady=5, sticky="w")

        tb_box = ttk.LabelFrame(self.monitor_page, text="ThingsBoard Cloud Upload through SIM7600G-H")
        tb_box.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Checkbutton(tb_box, text="Enable ThingsBoard upload", variable=self.tb_enabled_var).grid(row=0, column=0, padx=6, pady=5, sticky="w")
        ttk.Label(tb_box, text="Host").grid(row=0, column=1, padx=6, pady=5, sticky="e")
        ttk.Entry(tb_box, textvariable=self.tb_host_var, width=26).grid(row=0, column=2, padx=6, pady=5, sticky="w")
        ttk.Label(tb_box, text="Device token").grid(row=0, column=3, padx=6, pady=5, sticky="e")
        ttk.Entry(tb_box, textvariable=self.tb_token_var, width=34, show="*").grid(row=0, column=4, padx=6, pady=5, sticky="w")

        ttk.Label(tb_box, text="APN").grid(row=1, column=0, padx=6, pady=5, sticky="w")
        ttk.Entry(tb_box, textvariable=self.tb_apn_var, width=20).grid(row=1, column=1, padx=6, pady=5, sticky="w")
        ttk.Label(tb_box, text="Upload interval sec").grid(row=1, column=2, padx=6, pady=5, sticky="e")
        ttk.Entry(tb_box, textvariable=self.tb_upload_interval_sec_var, width=8).grid(row=1, column=3, padx=6, pady=5, sticky="w")
        ttk.Checkbutton(tb_box, text="Upload only valid detections", variable=self.tb_upload_valid_only_var).grid(row=1, column=4, padx=6, pady=5, sticky="w")

        ttk.Button(tb_box, text="Send Test Telemetry", command=self.send_test_thingsboard).grid(row=2, column=0, padx=6, pady=5, sticky="w")
        ttk.Label(tb_box, text="Uses the same SIM7600 AT port and baud selected in the SMS section above.").grid(row=2, column=1, columnspan=2, padx=6, pady=5, sticky="w")
        self.tb_status_var = tk.StringVar(value="ThingsBoard: disabled")
        ttk.Label(tb_box, textvariable=self.tb_status_var, wraplength=780).grid(row=2, column=3, columnspan=2, padx=6, pady=5, sticky="w")

        pred_box = ttk.LabelFrame(self.monitor_page, text="Live Prediction")
        pred_box.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.pred_var = tk.StringVar(value="Prediction: --")
        self.conf_var = tk.StringVar(value="Confidence: --")
        self.smooth_var = tk.StringVar(value="Smoothed decision: --")
        self.valid_status_var = tk.StringVar(value="Valid detection: --")
        self.prob_var = tk.StringVar(value="Probabilities: --")

        ttk.Label(pred_box, textvariable=self.pred_var, font=("Arial", 20, "bold")).pack(side=tk.LEFT, padx=12, pady=8)
        ttk.Label(pred_box, textvariable=self.conf_var, font=("Arial", 18)).pack(side=tk.LEFT, padx=12, pady=8)
        ttk.Label(pred_box, textvariable=self.smooth_var, font=("Arial", 16)).pack(side=tk.LEFT, padx=12, pady=8)
        ttk.Label(pred_box, textvariable=self.valid_status_var, font=("Arial", 14)).pack(side=tk.LEFT, padx=12, pady=8)
        ttk.Label(pred_box, textvariable=self.prob_var, wraplength=420).pack(side=tk.LEFT, padx=12, pady=8)

        plot_frame = ttk.Frame(self.monitor_page)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax_time = self.fig.add_subplot(311)
        self.ax_freq = self.fig.add_subplot(312)
        self.ax_hist = self.fig.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.clear_plots()

    # ---------------- Utils ----------------
    def log(self, msg):
        self.log_box.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see(tk.END)

    def post(self, target, msg):
        self.status_q.put((target, msg))

    def process_status(self):
        try:
            while True:
                target, msg = self.status_q.get_nowait()
                if target == "train":
                    self.train_status_var.set(msg)
                    self.log(msg)
                elif target == "monitor":
                    self.monitor_status_var.set(msg)
                elif target == "sms":
                    if hasattr(self, "sms_status_var"):
                        self.sms_status_var.set(msg)
                    try:
                        self.log(msg)
                    except Exception:
                        pass
                elif target == "tb":
                    if hasattr(self, "tb_status_var"):
                        self.tb_status_var.set(msg)
                    try:
                        self.log(msg)
                    except Exception:
                        pass
                elif target == "model":
                    self.model_status_var.set(msg)
                elif target == "refresh_dataset":
                    self.refresh_dataset()
                elif target == "refresh_monitor_models":
                    self.refresh_monitor_models()
                elif target == "error":
                    messagebox.showerror("Error", msg)
                elif target == "info":
                    messagebox.showinfo("Info", msg)
        except queue.Empty:
            pass
        self.root.after(100, self.process_status)

    def refresh_devices(self):
        devices = list_input_devices()
        self.device_combo["values"] = devices
        if self.device_var.get() not in devices:
            self.device_var.set(devices[0])

    def refresh_sms_ports(self):
        ports = list_serial_ports_for_sms()
        if hasattr(self, "sms_port_combo"):
            self.sms_port_combo["values"] = ports
        if self.sms_port_var.get() not in ports and ports:
            self.sms_port_var.set(ports[0])


    def selected_device(self):
        selected = self.device_var.get()
        dev = parse_device_index(selected)
        if dev is not None:
            return dev
        if sd is not None:
            try:
                devices = sd.query_devices()
                for idx, d in enumerate(devices):
                    if int(d.get("max_input_channels", 0)) > 0:
                        return idx
            except Exception:
                pass
        return None

    def get_cfg(self):
        sr = int(self.sample_rate_var.get())
        win = float(self.window_sec_var.get())
        overlap = float(self.overlap_var.get())
        low = float(self.low_cut_var.get())
        high = float(self.high_cut_var.get())

        if sr < 1000:
            raise ValueError("Sample rate must be at least 1000 Hz")
        if win < 0.5:
            raise ValueError("Window sec must be at least 0.5")
        if not (0 <= overlap < 0.95):
            raise ValueError("Overlap must be between 0 and 0.95")
        if high <= low:
            raise ValueError("High cut must be greater than low cut")

        return {"sample_rate": sr, "window_sec": win, "overlap": overlap, "low_cut": low, "high_cut": high}

    def get_effective_cfg(self, model_type, cfg):
        eff = dict(cfg)
        if model_type == MODEL_BIRDNET:
            eff["sample_rate"] = BIRDNET_TARGET_SR
            eff["window_sec"] = BIRDNET_TARGET_WIN_SEC
            eff["birdnet_forced"] = True
        else:
            eff["birdnet_forced"] = False
        return eff

    def is_birdnet_model_type(self, model_type):
        return model_type == MODEL_BIRDNET

    def is_matt_retrain_model_type(self, model_type):
        return model_type == MODEL_MATT_RETRAIN

    def add_label(self):
        label = self.new_label_var.get().strip()
        if not label:
            return
        if label not in self.labels:
            self.labels.append(label)
            save_labels(self.labels)
        self.label_combo["values"] = self.labels
        self.label_var.set(label)
        self.new_label_var.set("")
        self.refresh_dataset()
        self.log(f"Added/selected label: {label}")

    def refresh_dataset(self):
        self.labels = load_labels()
        if hasattr(self, "label_combo"):
            self.label_combo["values"] = self.labels
            if self.label_var.get() not in self.labels and self.labels:
                self.label_var.set(self.labels[0])

        if not hasattr(self, "dataset_tree"):
            return

        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)

        for label in self.labels:
            folder = DATASET_DIR / safe_label(label)
            folder.mkdir(parents=True, exist_ok=True)
            audio_files = audio_files_in_folder(folder)
            total = 0.0
            for audio_path in audio_files:
                try:
                    sr, x = read_audio_file(audio_path)
                    total += len(x) / float(sr)
                except Exception:
                    pass
            self.dataset_tree.insert("", tk.END, values=(label, len(audio_files), self.format_duration(total)))

    @staticmethod
    def format_duration(sec):
        sec = int(sec)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"


    def get_latest_model_path(self, model_name):
        return MODEL_LATEST_PATHS.get(model_name)

    def available_monitor_model_names(self):
        names = []
        for model_name in MODEL_OPTIONS:
            path = self.get_latest_model_path(model_name)
            if path is not None and Path(path).exists():
                names.append(model_name)
        return names

    def refresh_monitor_models(self):
        names = self.available_monitor_model_names()
        if hasattr(self, "monitor_model_combo"):
            self.monitor_model_combo["values"] = names

        current = self.monitor_model_var.get().strip()
        if current not in names:
            if self.model_package and self.model_package.get("model_type") in names:
                self.monitor_model_var.set(self.model_package.get("model_type"))
            elif names:
                self.monitor_model_var.set(names[0])
            else:
                self.monitor_model_var.set("")

    def save_latest_model_package(self, package):
        model_name = package.get("model_type")
        latest_path = self.get_latest_model_path(model_name)
        if latest_path is None:
            raise RuntimeError(f"No latest-model path configured for {model_name}")
        joblib.dump(package, latest_path)
        joblib.dump(package, DEPLOYED_MODEL)
        self.post("refresh_monitor_models", "")
        return latest_path

    def load_selected_monitor_model(self, show_popup=True):
        model_name = self.monitor_model_var.get().strip()
        if not model_name:
            if show_popup:
                messagebox.showwarning("No model selected", "Select a trained model for monitoring first.")
            return False
        return self.load_model(show_popup=show_popup, model_name=model_name)


    # ---------------- Matt Logic integration ----------------
    def ensure_tf_ready(self):
        if tf is None:
            raise RuntimeError(
                "Matt Logic model requires TensorFlow.\n\n"
                "Inside your venv run:\n"
                "python -m pip install tensorflow"
            )
        if librosa is None:
            raise RuntimeError(
                "Matt Logic model requires librosa.\n\n"
                "Inside your venv run:\n"
                "python -m pip install librosa"
            )
        if TFLiteInterpreter is None:
            raise RuntimeError(
                "Matt Logic live monitoring requires a TFLite interpreter.\n\n"
                "This app can use either tflite-runtime or TensorFlow Lite.\n"
                "You already verified TensorFlow Lite works from the terminal, so this error usually means the app imported the wrong fallback path.\n"
                "Use the patched file that prefers tf.lite.Interpreter when TensorFlow is installed.\n"
            )

    def _resize_2d_array(self, arr, out_h, out_w):
        arr = np.asarray(arr, dtype=np.float32)
        in_h, in_w = arr.shape
        if in_h == out_h and in_w == out_w:
            return arr

        x_old = np.linspace(0.0, 1.0, in_w)
        x_new = np.linspace(0.0, 1.0, out_w)
        temp = np.vstack([np.interp(x_new, x_old, row) for row in arr])

        y_old = np.linspace(0.0, 1.0, in_h)
        y_new = np.linspace(0.0, 1.0, out_h)
        out = np.vstack([np.interp(y_new, y_old, temp[:, col]) for col in range(temp.shape[1])]).T
        return out.astype(np.float32)

    def matt_audio_to_image(self, audio, sr, add_batch_dim=True):
        if librosa is None:
            raise RuntimeError("librosa is required for Matt Logic spectrogram preprocessing.")

        x = np.asarray(audio, dtype=np.float32).ravel()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if len(x) == 0:
            x = np.zeros(1024, dtype=np.float32)

        spec = librosa.feature.melspectrogram(y=x, sr=int(sr), n_mels=MATT_IMAGE_SIZE)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_resized = self._resize_2d_array(spec_db, MATT_IMAGE_SIZE, MATT_IMAGE_SIZE)

        spec_resized = np.nan_to_num(spec_resized, nan=0.0, posinf=0.0, neginf=0.0)
        spec_shifted = spec_resized - float(np.nanmin(spec_resized))
        max_val = float(np.nanmax(spec_shifted))
        if max_val > 0:
            spec_scaled = spec_shifted * (255.0 / max_val)
        else:
            spec_scaled = np.zeros_like(spec_shifted, dtype=np.float32)

        img = np.clip(spec_scaled, 0, 255).astype(np.uint8)
        img3 = np.stack([img, img, img], axis=-1)
        return np.expand_dims(img3, 0) if add_batch_dim else img3

    def ensure_matt_retrain_model_ready(self):
        if TFLiteInterpreter is None:
            raise RuntimeError(
                "Matt Logic live monitoring requires TFLite interpreter.\n\n"
                "Inside your venv run:\n"
                "python -m pip install tflite-runtime"
            )
        if not self.model_package or "matt_logic" not in self.model_package:
            raise RuntimeError("No deployed Matt Logic model is loaded.")

        meta = self.model_package["matt_logic"]
        model_path = Path(meta["model_path"])
        labels_path = Path(meta["labels_path"])

        if not model_path.exists():
            raise RuntimeError(f"Matt Logic TFLite model not found:\n{model_path}")

        if self.matt_retrain_labels is None:
            if labels_path.exists():
                self.matt_retrain_labels = [line.strip() for line in labels_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            else:
                self.matt_retrain_labels = list(self.model_package.get("labels", []))

        if self.matt_retrain_interpreter is None:
            self.matt_retrain_interpreter = TFLiteInterpreter(model_path=str(model_path))
            self.matt_retrain_interpreter.allocate_tensors()
            self.matt_retrain_input_details = self.matt_retrain_interpreter.get_input_details()
            self.matt_retrain_output_details = self.matt_retrain_interpreter.get_output_details()

        return self.matt_retrain_interpreter

    def matt_retrain_predict_audio_window(self, audio, sr):
        interpreter = self.ensure_matt_retrain_model_ready()
        image = self.matt_audio_to_image(audio, sr, add_batch_dim=True)

        input_detail = self.matt_retrain_input_details[0]
        output_detail = self.matt_retrain_output_details[0]
        input_index = int(input_detail["index"])
        output_index = int(output_detail["index"])

        tensor = image.astype(np.float32) if input_detail["dtype"] == np.float32 else image.astype(input_detail["dtype"])
        interpreter.set_tensor(input_index, tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        scores = np.squeeze(output).astype(np.float32)

        quant = output_detail.get("quantization", (0.0, 0))
        if quant and len(quant) == 2:
            scale, zero_point = quant
            if scale not in (None, 0):
                scores = (scores - float(zero_point)) * float(scale)

        scores = np.ravel(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0))
        if not (np.all(scores >= 0) and 0.8 <= float(np.sum(scores)) <= 1.2):
            ex = np.exp(scores - np.max(scores))
            probs = ex / np.sum(ex)
        else:
            s = float(np.sum(scores))
            probs = scores / s if s > 0 else np.ones(len(self.matt_retrain_labels), dtype=np.float32) / max(1, len(self.matt_retrain_labels))

        return list(self.matt_retrain_labels), probs.astype(np.float32)

    def _train_matt_retrain_model(self, cfg):
        self.ensure_tf_ready()
        self.post("train", "Matt Logic model selected. Building mel-spectrogram image dataset...")

        X = []
        y = []
        file_counter = Counter()
        total_files = sum(len(audio_files_in_folder(DATASET_DIR / safe_label(label))) for label in load_labels())
        seen_files = 0

        for label in load_labels():
            folder = DATASET_DIR / safe_label(label)
            audio_paths = audio_files_in_folder(folder)
            for audio_path in audio_paths:
                seen_files += 1
                try:
                    sr, audio = read_audio_file(audio_path)
                    audio = resample_if_needed(audio, sr, cfg["sample_rate"])
                    sr = int(cfg["sample_rate"])
                    windows = split_windows(audio, sr, cfg["window_sec"], cfg["overlap"])

                    for w in windows:
                        img = self.matt_audio_to_image(w, sr, add_batch_dim=False).astype(np.float32) / 255.0
                        X.append(img)
                        y.append(label)

                    file_counter[label] += 1
                    if seen_files % 5 == 0 or seen_files == total_files:
                        self.post("train", f"Matt Logic dataset progress: {seen_files}/{total_files} audio files")
                except Exception as e:
                    self.post("train", f"Skipped {audio_path.name}: {e}")

        if not X:
            raise RuntimeError("No WAV/MP3 training data found for Matt Logic model.")
        if len(set(y)) < 2:
            raise RuntimeError("Need at least 2 labels with samples for Matt Logic model.")

        class_names = sorted(set(y))
        label_to_idx = {lab: i for i, lab in enumerate(class_names)}
        y_idx = np.asarray([label_to_idx[val] for val in y], dtype=np.int32)
        X = np.stack(X).astype(np.float32)

        counts = Counter(y)
        self.post("train", f"Matt Logic dataset: {len(y)} windows | Classes used: {dict(counts)}")

        min_class_count = min(counts.values())
        if len(y_idx) >= 10 and min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=0.25, random_state=42, stratify=y_idx
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y_idx, y_idx
            self.post("train", "Not enough data for validation split. Training/evaluating on all available data.")

        try:
            base = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=(MATT_IMAGE_SIZE, MATT_IMAGE_SIZE, 3),
                pooling="avg",
            )
            self.post("train", "Loaded EfficientNetB0 ImageNet weights.")
        except Exception as e:
            self.post("train", f"Could not load ImageNet weights ({e}). Falling back to random init.")
            base = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(MATT_IMAGE_SIZE, MATT_IMAGE_SIZE, 3),
                pooling="avg",
            )

        base.trainable = False
        inputs = tf.keras.Input(shape=(MATT_IMAGE_SIZE, MATT_IMAGE_SIZE, 3), name="matt_logic_input")
        x = base(inputs, training=False)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="class_probs")(x)
        model = tf.keras.Model(inputs, outputs, name="matt_logic_retrainable")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.post("train", f"Training Matt Logic model for {MATT_RETRAIN_EPOCHS} epochs...")
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=MATT_RETRAIN_EPOCHS,
            batch_size=MATT_RETRAIN_BATCH_SIZE,
            verbose=0,
        )

        pred_probs = model.predict(X_test, verbose=0)
        pred_idx = np.argmax(pred_probs, axis=1)
        y_true_names = [class_names[i] for i in y_test]
        y_pred_names = [class_names[i] for i in pred_idx]

        acc = accuracy_score(y_true_names, y_pred_names)
        report = classification_report(
            y_true_names, y_pred_names, labels=class_names, target_names=class_names, zero_division=0
        )
        cm = confusion_matrix(y_true_names, y_pred_names, labels=class_names).tolist()

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        MATT_RETRAIN_TFLITE.write_bytes(tflite_model)
        MATT_RETRAIN_LABELS.write_text("\n".join(class_names), encoding="utf-8")

        self.matt_retrain_interpreter = None
        self.matt_retrain_input_details = None
        self.matt_retrain_output_details = None
        self.matt_retrain_labels = None

        package = {
            "pipeline": None,
            "feature_backend": "matt_logic_tflite",
            "feature_names": None,
            "embedding_dim": None,
            "labels": class_names,
            "config": cfg,
            "ui_config": cfg,
            "model_type": MODEL_MATT_RETRAIN,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "file_counter": dict(file_counter),
            "window_counter": dict(counts),
            "metrics": {
                "accuracy": float(acc),
                "report": report,
                "confusion_matrix": cm,
                "classes": class_names,
            },
            "matt_logic": {
                "enabled": True,
                "runtime_backend": TFLITE_RUNTIME_BACKEND or "tensorflow_lite",
                "model_path": str(MATT_RETRAIN_TFLITE),
                "labels_path": str(MATT_RETRAIN_LABELS),
                "input_size": [MATT_IMAGE_SIZE, MATT_IMAGE_SIZE, 3],
                "epochs": MATT_RETRAIN_EPOCHS,
                "batch_size": MATT_RETRAIN_BATCH_SIZE,
                "logic": "Mel spectrogram -> power_to_db -> resize 224x224 -> EfficientNetB0 -> export TFLite",
            },
        }

        latest_path = self.save_latest_model_package(package)
        self.model_package = package
        self.pipeline = None
        self.post("train", f"Validation accuracy: {acc:.3f}")
        self.post("train", "Report:\n" + report)
        self.post("model", self.model_status(package))
        self.post("train", f"Matt Logic model trained and deployed:\n{latest_path}")

    # ---------------- BirdNET integration ----------------
    def ensure_birdnet_ready(self):
        if birdnet_lib is None:
            raise RuntimeError(
                "birdnet is not installed.\n\n"
                "Inside your venv run:\n"
                "sudo apt install -y libsndfile1\n"
                "python -m pip install birdnet"
            )
        if self.birdnet_encoder is not None:
            return self.birdnet_encoder

        errors = []
        for backend in BIRDNET_BACKEND_CANDIDATES:
            try:
                encoder = birdnet_lib.load("acoustic", "2.4", backend)
                self.birdnet_encoder = encoder
                self.birdnet_backend = backend
                return encoder
            except Exception as e:
                errors.append(f"{backend}: {e}")

        raise RuntimeError(
            "Could not load BirdNET acoustic encoder.\nTried backends:\n- "
            + "\n- ".join(errors)
        )

    def birdnet_encode_audio_window(self, audio, sr, cfg):
        self.ensure_birdnet_ready()
        x = np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = apply_filter(x, sr, cfg["low_cut"], cfg["high_cut"])
        x = resample_if_needed(x, sr, BIRDNET_TARGET_SR)

        target_len = int(BIRDNET_TARGET_SR * BIRDNET_TARGET_WIN_SEC)
        if len(x) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(x)] = x
            x = padded
        elif len(x) > target_len:
            x = x[:target_len]

        tmp_path = self.birdnet_live_wav
        write_wav(tmp_path, BIRDNET_TARGET_SR, x)
        result = self.birdnet_encoder.encode(str(tmp_path))

        arr = _coerce_numeric_array(result)
        if arr.ndim == 1:
            vec = arr
        elif arr.ndim == 2:
            vec = np.mean(arr, axis=0)
        else:
            vec = np.ravel(arr)

        vec = np.asarray(vec, dtype=np.float32).ravel()
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        if vec.size < 8:
            raise RuntimeError(
                "BirdNET encode returned too few numeric values to be a usable embedding."
            )

        return vec

    # ---------------- Dataset actions ----------------
    def record_sample(self):
        if sd is None:
            messagebox.showerror("sounddevice error", str(SOUNDDEVICE_ERROR))
            return
        try:
            cfg = self.get_cfg()
            duration = float(self.record_sec_var.get())
            if duration <= 0:
                raise ValueError("Record duration must be positive")
        except Exception as e:
            messagebox.showerror("Invalid setting", str(e))
            return

        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("Label missing", "Select or add a label first.")
            return

        threading.Thread(target=self._record_thread, args=(label, cfg, duration), daemon=True).start()

    def _record_thread(self, label, cfg, duration):
        try:
            sr = int(cfg["sample_rate"])
            samples = int(sr * duration)
            device_id = self.selected_device()
            if device_id is None:
                raise RuntimeError("No input microphone found. Check arecord -l and USB mic connection.")
            self.post("train", f"Recording {duration:.1f}s for label '{label}' using device {device_id}...")
            audio = sd.rec(samples, samplerate=sr, channels=1, dtype="float32", device=device_id, blocking=True)
            audio = to_float_audio(audio)
            folder = DATASET_DIR / safe_label(label)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{safe_label(label)}_{now_string()}.wav"
            write_wav(path, sr, audio)
            self.post("train", f"Saved: {path.name}")
            self.post("refresh_dataset", "")
        except Exception as e:
            self.post("error", "Recording failed:\n\n" + str(e) + "\n\n" + traceback.format_exc())

    def import_audio_files(self):
        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("Label missing", "Select or add a label first.")
            return
        files = filedialog.askopenfilenames(
            title="Select WAV/MP3 audio files",
            filetypes=[
                ("Audio files", "*.wav *.WAV *.mp3 *.MP3"),
                ("WAV files", "*.wav *.WAV"),
                ("MP3 files", "*.mp3 *.MP3"),
                ("All files", "*.*"),
            ],
        )
        if files:
            self.copy_audio_files_to_label([Path(f) for f in files], label, source_name="files")

    def import_audio_folder(self):
        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("Label missing", "Select or add a label first.")
            return
        folder = filedialog.askdirectory(title="Select folder containing WAV/MP3 files")
        if not folder:
            return
        files = audio_files_in_folder(Path(folder))
        if not files:
            messagebox.showwarning("No audio", "No WAV/MP3 files found in selected folder.")
            return
        self.copy_audio_files_to_label(files, label, source_name=f"folder {folder}")

    def copy_audio_files_to_label(self, files, label, source_name="files"):
        folder = DATASET_DIR / safe_label(label)
        folder.mkdir(parents=True, exist_ok=True)
        count = 0
        skipped = 0
        for src in files:
            try:
                if src.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
                    skipped += 1
                    continue
                dst_name = f"{safe_label(label)}_imported_{now_string()}_{count:04d}{src.suffix.lower()}"
                dst = folder / dst_name
                shutil.copy2(src, dst)
                count += 1
            except Exception as e:
                skipped += 1
                self.log(f"Skipped {src}: {e}")
        self.refresh_dataset()
        self.log(f"Imported {count} audio file(s) from {source_name} to label '{label}'. Skipped: {skipped}")

    def open_dataset(self):
        path = str(DATASET_DIR)
        try:
            if os.name == "nt":
                os.startfile(path)
            elif os.uname().sysname == "Darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception:
            messagebox.showinfo("Dataset folder", path)

    def delete_trained_model(self):
        model_name = self.model_type_var.get().strip()
        latest_path = self.get_latest_model_path(model_name)
        if latest_path is None:
            messagebox.showinfo("No model", "Selected model type does not have a stored latest artifact.")
            return
        if not latest_path.exists():
            messagebox.showinfo("No model", f"No saved latest model exists for '{model_name}'.")
            return

        ok = messagebox.askyesno("Delete trained model", f"Delete latest saved model for:\n\n{model_name}\n\n{latest_path}")
        if not ok:
            return

        try:
            latest_path.unlink(missing_ok=True)
            if model_name == MODEL_MATT_RETRAIN:
                MATT_RETRAIN_TFLITE.unlink(missing_ok=True)
                MATT_RETRAIN_LABELS.unlink(missing_ok=True)

            if DEPLOYED_MODEL.exists():
                try:
                    pkg = joblib.load(DEPLOYED_MODEL)
                    if pkg.get("model_type") == model_name:
                        DEPLOYED_MODEL.unlink(missing_ok=True)
                except Exception:
                    pass

            if self.model_package and self.model_package.get("model_type") == model_name:
                self.pipeline = None
                self.model_package = None
                self.matt_retrain_interpreter = None
                self.matt_retrain_input_details = None
                self.matt_retrain_output_details = None
                self.matt_retrain_labels = None
                self.model_status_var.set("Model: deleted")

            self.refresh_monitor_models()
            self.log(f"Deleted latest saved model for '{model_name}'.")
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))

    def delete_selected_label_dataset(self):
        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("Label missing", "Select a label first.")
            return
        folder = DATASET_DIR / safe_label(label)
        files = audio_files_in_folder(folder)
        if not files:
            messagebox.showinfo("No dataset", f"No audio files found for label '{label}'.")
            return
        ok = messagebox.askyesno(
            "Delete label dataset",
            f"Delete all {len(files)} audio file(s) for label '{label}'?\n\nThis will not delete other labels.",
        )
        if not ok:
            return
        deleted = 0
        for f in files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                self.log(f"Could not delete {f.name}: {e}")
        self.refresh_dataset()
        self.log(f"Deleted {deleted} audio file(s) from label '{label}'.")
        if messagebox.askyesno("Retrain now?", "Retrain model now using latest remaining dataset?"):
            self.train_model()

    # ---------------- Training ----------------
    def train_model(self):
        try:
            cfg = self.get_cfg()
        except Exception as e:
            messagebox.showerror("Invalid setting", str(e))
            return
        threading.Thread(target=self._train_thread, args=(cfg,), daemon=True).start()

    def _train_thread(self, cfg):
        try:
            selected_model = self.model_type_var.get()
            if self.is_matt_retrain_model_type(selected_model):
                self._train_matt_retrain_model(cfg)
                return

            eff_cfg = self.get_effective_cfg(selected_model, cfg)
            is_birdnet = self.is_birdnet_model_type(selected_model)

            if is_birdnet:
                self.post("train", "BirdNET model selected. Preparing BirdNET encoder...")
                self.ensure_birdnet_ready()
                self.post(
                    "train",
                    f"BirdNET backend ready ({self.birdnet_backend}). "
                    f"Using {BIRDNET_TARGET_SR} Hz and {BIRDNET_TARGET_WIN_SEC:.1f}s windows."
                )
                self.post("train", "Extracting BirdNET embeddings from WAV/MP3 dataset...")
            else:
                self.post("train", "Extracting handcrafted features from WAV/MP3 dataset...")

            X = []
            y = []
            file_counter = Counter()

            for label in load_labels():
                folder = DATASET_DIR / safe_label(label)
                audio_paths = audio_files_in_folder(folder)
                for audio_path in audio_paths:
                    try:
                        sr, audio = read_audio_file(audio_path)
                        audio = resample_if_needed(audio, sr, eff_cfg["sample_rate"])
                        sr = int(eff_cfg["sample_rate"])
                        windows = split_windows(audio, sr, eff_cfg["window_sec"], eff_cfg["overlap"])

                        for w in windows:
                            if is_birdnet:
                                feat = self.birdnet_encode_audio_window(w, sr, eff_cfg)
                            else:
                                feat = extract_features(w, sr, eff_cfg["low_cut"], eff_cfg["high_cut"])
                            X.append(feat)
                            y.append(label)

                        file_counter[label] += 1
                    except Exception as e:
                        self.post("train", f"Skipped {audio_path.name}: {e}")

            if not X:
                raise RuntimeError("No WAV/MP3 training data found.")
            if len(set(y)) < 2:
                raise RuntimeError("Need at least 2 labels with samples after exclusions/deletions.")

            X = np.vstack(X).astype(np.float32)
            y = np.asarray(y)
            counts = Counter(y)
            self.post("train", f"Windows: {len(y)} | Classes used: {dict(counts)} | Feature dim: {X.shape[1]}")

            if selected_model == MODEL_HANDCRAFT_SVM:
                clf = SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                )
                pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
                feature_backend = "handcrafted"
            elif selected_model == MODEL_HANDCRAFT_RF:
                clf = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                )
                pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
                feature_backend = "handcrafted"
            else:
                clf = LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                )
                pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
                feature_backend = "birdnet_embeddings"

            metrics = {}
            min_class_count = min(counts.values())

            if len(y) >= 10 and min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_).tolist()
                metrics = {
                    "accuracy": float(acc),
                    "report": report,
                    "confusion_matrix": cm,
                    "classes": list(pipeline.classes_),
                }
                self.post("train", f"Validation accuracy: {acc:.3f}")
                self.post("train", "Report:\n" + report)
                pipeline.fit(X, y)
            else:
                self.post("train", "Not enough data for validation split. Training on all available data.")
                pipeline.fit(X, y)

            package = {
                "pipeline": pipeline,
                "feature_backend": feature_backend,
                "feature_names": FEATURE_NAMES if feature_backend == "handcrafted" else None,
                "embedding_dim": int(X.shape[1]) if feature_backend == "birdnet_embeddings" else None,
                "labels": list(pipeline.classes_),
                "config": eff_cfg,
                "ui_config": cfg,
                "model_type": selected_model,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "file_counter": dict(file_counter),
                "window_counter": dict(counts),
                "metrics": metrics,
                "birdnet": {
                    "enabled": feature_backend == "birdnet_embeddings",
                    "backend": self.birdnet_backend if feature_backend == "birdnet_embeddings" else None,
                    "target_sr": BIRDNET_TARGET_SR if feature_backend == "birdnet_embeddings" else None,
                    "target_window_sec": BIRDNET_TARGET_WIN_SEC if feature_backend == "birdnet_embeddings" else None,
                },
            }

            latest_path = self.save_latest_model_package(package)
            self.model_package = package
            self.pipeline = pipeline
            self.post("model", self.model_status(package))
            self.post("train", f"Model trained and deployed:\n{latest_path}")
        except Exception as e:
            self.post("error", "Training failed:\n\n" + str(e) + "\n\n" + traceback.format_exc())
            self.post("train", f"Training failed: {e}")

    # ---------------- Model loading/status ----------------
    def model_status(self, package):
        cfg = package.get("config", {})
        labels = ", ".join(package.get("labels", []))
        backend = package.get("feature_backend", "unknown")
        extra = ""
        if backend == "birdnet_embeddings":
            bird = package.get("birdnet", {})
            extra = f" | BirdNET: {bird.get('backend', '--')} | Emb dim: {package.get('embedding_dim', '--')}"
        elif backend == "matt_logic_tflite":
            matt = package.get("matt_logic", {})
            extra = f" | Matt Logic: {matt.get('runtime_backend', '--')} | Input: {matt.get('input_size', '--')}"
        return (
            f"Model: {package.get('model_type', '--')} | "
            f"Features: {backend} | Classes: {labels} | "
            f"SR: {cfg.get('sample_rate', '--')} Hz | "
            f"Band: {cfg.get('low_cut', '--')}-{cfg.get('high_cut', '--')} Hz"
            f"{extra}"
        )

    def load_model(self, show_popup=True, model_name=None, model_path=None):
        try:
            target_path = None
            if model_path is not None:
                target_path = Path(model_path)
            elif model_name is not None:
                target_path = self.get_latest_model_path(model_name)
            else:
                target_path = DEPLOYED_MODEL

            if target_path is None or not Path(target_path).exists():
                if hasattr(self, "model_status_var"):
                    self.model_status_var.set("Model: not loaded")
                if show_popup:
                    messagebox.showwarning("No model", "Requested model was not found. Train it first.")
                return False

            package = joblib.load(target_path)
            self.model_package = package
            self.pipeline = package.get("pipeline")
            self.matt_retrain_interpreter = None
            self.matt_retrain_input_details = None
            self.matt_retrain_output_details = None
            self.matt_retrain_labels = None

            if hasattr(self, "model_status_var"):
                self.model_status_var.set(self.model_status(package))
            if hasattr(self, "monitor_model_var") and package.get("model_type"):
                self.monitor_model_var.set(package.get("model_type"))
            self.refresh_monitor_models()

            if show_popup:
                messagebox.showinfo("Model loaded", f"Loaded model:\n\n{package.get('model_type', '--')}")
            return True
        except Exception as e:
            if show_popup:
                messagebox.showerror("Model load failed", str(e))
            if hasattr(self, "model_status_var"):
                self.model_status_var.set("Model: load failed")
            return False

    # ---------------- Monitoring plots ----------------
    def clear_plots(self):
        self.ax_time.clear()
        self.ax_time.set_title("Time Domain Waveform")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.grid(True, alpha=0.3)

        self.ax_freq.clear()
        self.ax_freq.set_title("Frequency Domain")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        self.ax_freq.grid(True, alpha=0.3)

        self.ax_hist.clear()
        self.ax_hist.set_title("Frequency Domain Histogram / Band Energy")
        self.ax_hist.set_xlabel("Frequency band (Hz)")
        self.ax_hist.set_ylabel("Log energy")
        self.ax_hist.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    # ---------------- Monitoring ----------------
    def start_monitoring(self):
        if sd is None:
            messagebox.showerror("sounddevice error", str(SOUNDDEVICE_ERROR))
            return

        selected_monitor_model = self.monitor_model_var.get().strip()
        current_model_name = self.model_package.get("model_type") if self.model_package else ""

        if selected_monitor_model and selected_monitor_model != current_model_name:
            if not self.load_selected_monitor_model(show_popup=False):
                messagebox.showwarning("No model", "Select and load a trained monitoring model first.")
                return
        elif self.pipeline is None and (not self.model_package or self.model_package.get("feature_backend") != "matt_logic_tflite"):
            if not self.load_model(show_popup=False):
                messagebox.showwarning("No model", "Train/load a model first.")
                return

        if self.monitor_running:
            return
        try:
            cfg = self.model_package.get("config", self.get_cfg()) if self.model_package else self.get_cfg()
            if self.model_package and self.model_package.get("feature_backend") == "birdnet_embeddings":
                self.ensure_birdnet_ready()
            if self.model_package and self.model_package.get("feature_backend") == "matt_logic_tflite":
                self.ensure_matt_retrain_model_ready()

            sr = int(cfg["sample_rate"])
            block = max(256, int(sr * 0.1))
            device_id = self.selected_device()
            if device_id is None:
                raise RuntimeError("No input microphone found.")
            keep = int(sr * max(float(cfg["window_sec"]) * 2.5, 5.0))
            self.monitor_audio = np.zeros(keep, dtype=np.float32)
            self.prediction_history.clear()
            self.last_plot = 0.0
            self.last_predict = 0.0
            self.last_cyclic_sms_time = time.time()
            self.cyclic_detection_stats.clear()
            self.last_event_sms_signature = None
            self.tb_last_upload_time = 0.0

            try:
                while True:
                    self.monitor_q.get_nowait()
            except queue.Empty:
                pass

            self.monitor_stream = sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocksize=block,
                device=device_id,
                callback=self.audio_callback,
            )
            self.monitor_stream.start()
            self.monitor_running = True
            self.monitor_status_var.set(f"Monitoring: running on device {device_id}")
            self.root.after(MONITOR_LOOP_MS, self.process_monitor_audio)
        except Exception as e:
            self.monitor_running = False
            messagebox.showerror("Monitoring failed", str(e))

    def stop_monitoring(self):
        self.monitor_running = False
        try:
            if self.monitor_stream is not None:
                self.monitor_stream.stop()
                self.monitor_stream.close()
        except Exception:
            pass
        self.monitor_stream = None
        self.monitor_status_var.set("Monitoring: stopped")

    def audio_callback(self, indata, frames, time_info, status):
        try:
            block = to_float_audio(indata).copy()
            try:
                self.monitor_q.put_nowait(block)
            except queue.Full:
                try:
                    self.monitor_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.monitor_q.put_nowait(block)
                except queue.Full:
                    pass
        except Exception:
            pass

    def process_monitor_audio(self):
        if not self.monitor_running:
            return

        blocks = deque(maxlen=MAX_BLOCKS_PER_UI_TICK)
        drained = 0
        try:
            while drained < MAX_AUDIO_QUEUE_BLOCKS:
                blocks.append(self.monitor_q.get_nowait())
                drained += 1
        except queue.Empty:
            pass

        if blocks:
            new = np.concatenate(list(blocks)).astype(np.float32)
            cfg = self.model_package.get("config", self.get_cfg()) if self.model_package else self.get_cfg()
            sr = int(cfg["sample_rate"])
            keep = int(sr * max(float(cfg["window_sec"]) * 2.5, 5.0))

            self.monitor_audio = np.concatenate([self.monitor_audio, new])
            if len(self.monitor_audio) > keep:
                self.monitor_audio = self.monitor_audio[-keep:]

            now = time.time()

            feature_backend = self.model_package.get("feature_backend", "handcrafted") if self.model_package else "handcrafted"

            plot_sec = BIRDNET_MONITOR_PLOT_SEC if feature_backend == "birdnet_embeddings" else PLOT_INTERVAL_SEC
            if now - self.last_plot >= plot_sec:
                self.update_plots(cfg)
                self.last_plot = now

            if feature_backend == "birdnet_embeddings":
                hop_sec = BIRDNET_MONITOR_PREDICT_SEC
            else:
                hop_sec = max(1.0, float(cfg["window_sec"]) * (1.0 - float(cfg["overlap"])))

            if now - self.last_predict >= hop_sec:
                self.predict_latest(cfg)
                self.last_predict = now

        self.root.after(MONITOR_LOOP_MS, self.process_monitor_audio)

    def latest_window(self, cfg):
        sr = int(cfg["sample_rate"])
        win_len = int(sr * float(cfg["window_sec"]))
        if len(self.monitor_audio) >= win_len:
            return self.monitor_audio[-win_len:]
        out = np.zeros(win_len, dtype=np.float32)
        out[-len(self.monitor_audio):] = self.monitor_audio
        return out

    def update_plots(self, cfg):
        sr = int(cfg["sample_rate"])
        low = float(cfg["low_cut"])
        high = float(cfg["high_cut"])
        x = self.latest_window(cfg)

        xf = apply_filter(x, sr, low, high)

        t = np.arange(len(xf)) / float(sr)
        step_t = max(1, len(xf) // MAX_PLOT_POINTS)
        self.ax_time.clear()
        self.ax_time.plot(t[::step_t], xf[::step_t], linewidth=1.0)
        self.ax_time.set_title("Time Domain Waveform")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.set_ylim(-1.05, 1.05)
        self.ax_time.grid(True, alpha=0.3)

        freqs, power = compute_spectrum(xf, sr)
        mag = np.sqrt(power)
        max_f = min(high, sr / 2.0)
        mask = freqs <= max_f
        f_plot = freqs[mask]
        m_plot = mag[mask]
        step_f = max(1, len(f_plot) // MAX_PLOT_POINTS)

        self.ax_freq.clear()
        self.ax_freq.plot(f_plot[::step_f], m_plot[::step_f], linewidth=1.0)
        self.ax_freq.set_title("Frequency Domain")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        self.ax_freq.grid(True, alpha=0.3)

        band_names = []
        band_values = []
        nyq = sr / 2.0
        for f1, f2, name in FREQ_BANDS:
            if f1 > max_f:
                continue
            bp = band_power(freqs, power, f1, min(f2, nyq, max_f))
            band_names.append(name)
            band_values.append(float(np.log1p(bp)))

        self.ax_hist.clear()
        self.ax_hist.bar(band_names, band_values)
        self.ax_hist.set_title("Frequency Domain Histogram / Band Energy")
        self.ax_hist.set_xlabel("Frequency band (Hz)")
        self.ax_hist.set_ylabel("Log energy")
        self.ax_hist.grid(True, axis="y", alpha=0.3)
        self.ax_hist.tick_params(axis="x", rotation=20)

        self.canvas.draw_idle()

    def predict_latest(self, cfg):
        feature_backend = self.model_package.get("feature_backend", "handcrafted") if self.model_package else "handcrafted"
        if self.pipeline is None and feature_backend != "matt_logic_tflite":
            return

        x = self.latest_window(cfg)

        try:
            if feature_backend == "birdnet_embeddings":
                feat = self.birdnet_encode_audio_window(x, int(cfg["sample_rate"]), cfg).reshape(1, -1)
                probs = self.pipeline.predict_proba(feat)[0]
                classes = list(self.pipeline.classes_)
            elif feature_backend == "matt_logic_tflite":
                classes, probs = self.matt_retrain_predict_audio_window(x, int(cfg["sample_rate"]))
            else:
                feat = extract_features(x, int(cfg["sample_rate"]), cfg["low_cut"], cfg["high_cut"]).reshape(1, -1)
                probs = self.pipeline.predict_proba(feat)[0]
                classes = list(self.pipeline.classes_)
            idx = int(np.argmax(probs))

            pred = str(classes[idx])
            conf = float(probs[idx])

            pairs = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
            prob_txt = " | ".join([f"{c}: {p * 100:.1f}%" for c, p in pairs])

            self.prediction_history.append((pred, conf))
            smooth = self.smooth_decision()

            threshold = self.get_conf_threshold()
            confirmed_label = self.confirmed_label_from_smooth(smooth)
            is_valid = bool(confirmed_label) and conf >= threshold

            self.pred_var.set(f"Prediction: {pred}")
            self.conf_var.set(f"Confidence: {conf * 100:.1f}%")
            self.smooth_var.set(f"Smoothed decision: {smooth}")
            if is_valid:
                self.valid_status_var.set(f"Valid detection: {confirmed_label}")
            else:
                self.valid_status_var.set(f"Valid detection: none (threshold {threshold * 100:.0f}%)")
            self.prob_var.set(f"Probabilities: {prob_txt}")

            self.handle_sms_logic(confirmed_label, conf, is_valid)
            self.handle_thingsboard_logic(pred, conf, pairs, smooth, confirmed_label, is_valid)

        except Exception as e:
            self.monitor_status_var.set(f"Prediction error: {e}")

    def get_conf_threshold(self):
        try:
            value = float(self.conf_threshold_var.get())
        except Exception:
            value = 60.0
        # Accept both 0.60 and 60.
        if value > 1.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    def smooth_decision(self):
        if not self.prediction_history:
            return "--"
        threshold = self.get_conf_threshold()
        confident = [p for p, c in self.prediction_history if c >= threshold]
        if not confident:
            return "Uncertain"
        counts = Counter(confident)
        label, count = counts.most_common(1)[0]
        if len(self.prediction_history) >= 5 and count >= 3:
            return f"{label} confirmed"
        if count >= 2:
            return f"Possible {label}"
        return "Collecting evidence"

    @staticmethod
    def confirmed_label_from_smooth(smooth):
        text = str(smooth or "")
        suffix = " confirmed"
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()
        return None

    # ---------------- SIM7600 SMS alerting ----------------
    def get_sms_interval_sec(self):
        try:
            val = float(self.sms_interval_sec_var.get())
        except Exception:
            val = 60.0
        return max(5.0, val)

    def sms_is_enabled(self):
        return bool(self.sms_enabled_var.get())

    def selected_sms_port(self):
        return parse_serial_port(self.sms_port_var.get())

    def sms_worker_is_alive(self):
        return bool(self.sms_thread is not None and self.sms_thread.is_alive())

    def reset_sms_busy(self):
        # This only clears the app-side guard. It cannot kill an OS-level process
        # such as minicom/picocom/ModemManager holding the modem port.
        if self.sms_worker_is_alive():
            age = time.time() - float(self.sms_send_started_at or 0.0)
            self.post("sms", f"SMS worker still active for {age:.1f}s. Wait for timeout or close the process using the port.")
            return
        self.sms_send_running = False
        self.sms_thread = None
        self.sms_send_started_at = 0.0
        self.post("sms", "SMS busy flag reset.")

    def port_owner_text(self, port):
        lines = []
        for cmd in (["fuser", "-v", port], ["lsof", port]):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                output = (result.stdout or "") + (result.stderr or "")
                output = output.strip()
                if output:
                    lines.append("$ " + " ".join(cmd) + "\n" + output)
            except Exception as e:
                lines.append("$ " + " ".join(cmd) + f"\nCould not run: {e}")
        return "\n\n".join(lines).strip()

    def diagnose_selected_sms_port(self):
        port = self.selected_sms_port()
        if not port:
            messagebox.showwarning("No port selected", "Select a SIM7600 AT port first.")
            return
        owner = self.port_owner_text(port)
        if not owner:
            owner = (
                f"No process owner was reported for {port}.\n"
                "If SMS still says port busy, close any serial terminal and stop ModemManager."
            )
        self.post("sms", f"Port diagnosis for {port}:\n{owner[:700]}")
        messagebox.showinfo("SIM7600 port diagnosis", owner)

    def probe_at_port(self, port, baud):
        try:
            with serial.Serial(port=port, baudrate=int(baud), timeout=0.4, write_timeout=2, exclusive=False) as ser:
                time.sleep(0.2)
                ser.reset_input_buffer()
                ser.write(b"AT\r")
                ser.flush()
                resp = self.serial_read_until(ser, expected=["OK", "ERROR"], timeout=1.5)
                return "OK" in resp, resp.strip()
        except Exception as e:
            return False, str(e)

    def auto_detect_sms_port(self):
        if serial is None:
            messagebox.showerror("pyserial missing", f"pyserial not installed: {SERIAL_IMPORT_ERROR}")
            return
        try:
            baud = int(self.sms_baud_var.get())
        except Exception:
            baud = DEFAULT_SMS_BAUD

        ports = []
        # Prefer displayed/refreshed ports but also check ttyUSB candidates.
        for display in list(self.sms_port_combo["values"] if hasattr(self, "sms_port_combo") else []):
            port = parse_serial_port(display)
            if port and port not in ports:
                ports.append(port)
        for cand in DEFAULT_SMS_PORT_CANDIDATES:
            if Path(cand).exists() and cand not in ports:
                ports.append(cand)

        results = []
        for port in ports:
            ok, resp = self.probe_at_port(port, baud)
            results.append(f"{port}: {'AT OK' if ok else 'no AT'} | {resp[:120]}")
            if ok:
                self.sms_port_var.set(port)
                self.post("sms", f"Auto-detected SIM7600 AT port: {port}")
                messagebox.showinfo("SIM7600 AT port found", f"AT response found on {port}\n\n" + "\n".join(results))
                return

        msg = "No AT-responsive port found.\n\n" + "\n".join(results)
        self.post("sms", msg[:700])
        messagebox.showwarning("No AT port found", msg)

    def handle_sms_logic(self, confirmed_label, confidence, is_valid):
        # The enable checkbox is the master SMS gate. Nothing is sent unless it is checked.
        if not self.sms_is_enabled():
            self.cyclic_detection_stats.clear()
            self.last_event_sms_signature = None
            if hasattr(self, "sms_status_var"):
                self.sms_status_var.set("SMS: disabled")
            return

        mode = self.sms_mode_var.get()
        now = time.time()

        if mode == SMS_MODE_CYCLIC:
            if is_valid and confirmed_label:
                previous = self.cyclic_detection_stats.get(confirmed_label, 0.0)
                self.cyclic_detection_stats[confirmed_label] = max(previous, float(confidence))

            if self.last_cyclic_sms_time <= 0:
                self.last_cyclic_sms_time = now

            interval = self.get_sms_interval_sec()
            if now - self.last_cyclic_sms_time >= interval:
                message = self.build_cyclic_sms_message()
                self.send_sms_async(message, reason="cyclic")
                self.cyclic_detection_stats.clear()
                self.last_cyclic_sms_time = now
                self.last_event_sms_signature = None
        else:
            if is_valid and confirmed_label:
                # Edge-triggered event SMS: avoid sending the same label repeatedly while it remains confirmed.
                signature = confirmed_label
                if signature != self.last_event_sms_signature:
                    message = self.build_event_sms_message(confirmed_label, confidence)
                    self.send_sms_async(message, reason="confirmed detection")
                    self.last_event_sms_signature = signature
            else:
                self.last_event_sms_signature = None

    def build_cyclic_sms_message(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.cyclic_detection_stats:
            return f"No detection @ {ts}"

        items = []
        for label, conf in sorted(self.cyclic_detection_stats.items(), key=lambda item: item[0]):
            if self.sms_include_conf_var.get():
                items.append(f"{label} {conf * 100:.0f}%")
            else:
                items.append(str(label))
        return f"Detected animal sounds: {', '.join(items)} @ {ts}"

    def build_event_sms_message(self, label, confidence):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.sms_include_conf_var.get():
            return f"Detected animal sound: {label} ({confidence * 100:.1f}%) @ {ts}"
        return f"Detected animal sound: {label} @ {ts}"

    def send_test_sms(self):
        if not self.sms_is_enabled():
            messagebox.showwarning("SMS disabled", "Enable SMS transmission before sending a test SMS.")
            return
        self.send_sms_async("Animal sound detector test SMS", reason="test")

    def send_sms_async(self, message, reason="alert"):
        phone = self.sms_phone_var.get().strip()
        port = self.selected_sms_port()
        try:
            baud = int(self.sms_baud_var.get())
        except Exception:
            baud = DEFAULT_SMS_BAUD

        if serial is None:
            self.post("sms", f"SMS error: pyserial not installed ({SERIAL_IMPORT_ERROR}). Run: python -m pip install pyserial")
            return
        if not phone:
            self.post("sms", "SMS error: recipient phone number is empty")
            return
        if not port:
            self.post("sms", "SMS error: SIM7600 AT port is empty")
            return

        # If an old thread already exited, recover automatically.
        if self.sms_send_running and not self.sms_worker_is_alive():
            self.sms_send_running = False
            self.sms_thread = None

        if self.sms_send_running:
            age = time.time() - float(self.sms_send_started_at or 0.0)
            self.post("sms", f"SMS busy: previous send still active for {age:.1f}s. Wait or click Diagnose selected port.")
            return

        self.sms_send_running = True
        self.sms_send_started_at = time.time()
        self.sms_thread = threading.Thread(
            target=self._sms_thread,
            args=(port, baud, phone, str(message), reason),
            daemon=True,
        )
        self.sms_thread.start()

    def _sms_thread(self, port, baud, phone, message, reason):
        try:
            with self.sms_lock:
                self.post("sms", f"SMS sending ({reason}) via {port}...")
                self.send_sms_via_sim7600(port, baud, phone, message)
                self.post("sms", f"SMS sent ({reason}): {message[:120]}")
                self.sms_last_error = ""
        except Exception as e:
            self.sms_last_error = str(e)
            self.post("sms", f"SMS failed ({reason}): {e}")
        finally:
            self.sms_send_running = False
            self.sms_send_started_at = 0.0

    def serial_read_until(self, ser, expected=None, timeout=5.0):
        expected = expected or []
        if isinstance(expected, str):
            expected = [expected]
        end = time.time() + float(timeout)
        buf = ""
        while time.time() < end:
            try:
                chunk = ser.read(ser.in_waiting or 1)
            except Exception:
                chunk = b""
            if chunk:
                try:
                    buf += chunk.decode("utf-8", errors="ignore")
                except Exception:
                    buf += str(chunk)
                if any(token in buf for token in expected):
                    return buf
            else:
                time.sleep(0.05)
        return buf

    def send_at(self, ser, cmd, expected="OK", timeout=3.0):
        ser.reset_input_buffer()
        ser.write((cmd + "\r").encode("ascii", errors="ignore"))
        ser.flush()
        response = self.serial_read_until(ser, expected=[expected, "ERROR"], timeout=timeout)
        if expected and expected not in response:
            raise RuntimeError(f"AT command failed: {cmd} -> {response.strip() or 'timeout'}")
        if "ERROR" in response and expected not in response:
            raise RuntimeError(f"AT command error: {cmd} -> {response.strip()}")
        return response

    def send_sms_via_sim7600(self, port, baud, phone, message):
        # SIM7600 SMS text-mode sequence: AT, text mode, AT+CMGS, message body, Ctrl+Z.
        # Open/close per send so the app recovers if the USB dongle is unplugged/replugged.
        try:
            ser_ctx = serial.Serial(port=port, baudrate=int(baud), timeout=0.5, write_timeout=5, exclusive=False)
        except Exception as e:
            owner = self.port_owner_text(port)
            hint = (
                f"Could not open {port}: {e}. "
                "Close minicom/picocom/screen/serial monitor or stop ModemManager if it owns the port."
            )
            if owner:
                hint += f"\nPort owner info:\n{owner[:500]}"
            raise RuntimeError(hint)

        with ser_ctx as ser:
            time.sleep(0.2)
            self.send_at(ser, "AT", expected="OK", timeout=2.0)
            self.send_at(ser, "ATE0", expected="OK", timeout=2.0)
            self.send_at(ser, "AT+CMGF=1", expected="OK", timeout=2.0)
            self.send_at(ser, 'AT+CSCS="GSM"', expected="OK", timeout=2.0)

            ser.reset_input_buffer()
            ser.write((f'AT+CMGS="{phone}"\r').encode("ascii", errors="ignore"))
            ser.flush()
            prompt = self.serial_read_until(ser, expected=[">", "ERROR"], timeout=5.0)
            if ">" not in prompt:
                raise RuntimeError(f"No SMS prompt from modem: {prompt.strip() or 'timeout'}")

            clean_message = str(message).replace("\r", " ").replace("\n", " ")[:300]
            ser.write(clean_message.encode("ascii", errors="ignore") + SMS_CTRL_Z)
            ser.flush()
            response = self.serial_read_until(ser, expected=["+CMGS:", "OK", "ERROR"], timeout=18.0)
            if "+CMGS:" not in response and "OK" not in response:
                raise RuntimeError(f"SMS send not confirmed: {response.strip() or 'timeout'}")
            if "ERROR" in response and "+CMGS:" not in response:
                raise RuntimeError(f"SMS send error: {response.strip()}")
            return response

    # ---------------- ThingsBoard cloud upload through SIM7600 ----------------
    def thingsboard_is_enabled(self):
        return bool(self.tb_enabled_var.get())

    def get_tb_upload_interval_sec(self):
        try:
            val = float(self.tb_upload_interval_sec_var.get())
        except Exception:
            val = DEFAULT_TB_UPLOAD_INTERVAL_SEC
        return max(1.0, val)

    def tb_worker_is_alive(self):
        return bool(self.tb_thread is not None and self.tb_thread.is_alive())

    def safe_tb_key(self, text):
        cleaned = "".join(c if c.isalnum() else "_" for c in str(text).strip())
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return cleaned or "Unknown"

    def build_thingsboard_payload(self, pred, confidence, pairs, smooth, confirmed_label, is_valid):
        threshold = self.get_conf_threshold()
        payload = {
            "animal_prediction_label": str(pred),
            "animal_prediction_confidence": round(float(confidence), 4),
            "animal_prediction_confidence_pct": round(float(confidence) * 100.0, 1),
            "animal_prediction_valid": bool(is_valid),
            "animal_confirmed_label": str(confirmed_label or ""),
            "animal_smooth_decision": str(smooth),
            "animal_threshold_pct": round(float(threshold) * 100.0, 1),
            "animal_model_type": str(self.model_package.get("model_type", "") if self.model_package else ""),
            "animal_uploaded_from": "RPi5_SIM7600G_H",
            "animal_client_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add one numeric ThingsBoard key per class for easy dashboard widgets.
        for cls, prob in pairs:
            key = "animal_prob_" + self.safe_tb_key(cls)
            payload[key] = round(float(prob) * 100.0, 1)

        return payload

    def handle_thingsboard_logic(self, pred, confidence, pairs, smooth, confirmed_label, is_valid):
        # Independent master gate. Existing prediction and SMS behavior continues unchanged.
        if not self.thingsboard_is_enabled():
            if hasattr(self, "tb_status_var"):
                self.tb_status_var.set("ThingsBoard: disabled")
            return

        if bool(self.tb_upload_valid_only_var.get()) and not is_valid:
            if hasattr(self, "tb_status_var"):
                self.tb_status_var.set("ThingsBoard: waiting for valid detection")
            return

        now = time.time()
        interval = self.get_tb_upload_interval_sec()
        if self.tb_last_upload_time > 0 and now - self.tb_last_upload_time < interval:
            return

        payload = self.build_thingsboard_payload(pred, confidence, pairs, smooth, confirmed_label, is_valid)
        self.send_thingsboard_async(payload, reason="prediction")
        self.tb_last_upload_time = now

    def send_test_thingsboard(self):
        if not self.thingsboard_is_enabled():
            messagebox.showwarning("ThingsBoard disabled", "Enable ThingsBoard upload before sending test telemetry.")
            return
        payload = {
            "animal_tb_test": 1,
            "animal_tb_test_message": "Animal sound detector test telemetry",
            "animal_uploaded_from": "RPi5_SIM7600G_H",
            "animal_client_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.send_thingsboard_async(payload, reason="test")

    def send_thingsboard_async(self, payload, reason="telemetry"):
        port = self.selected_sms_port()
        try:
            baud = int(self.sms_baud_var.get())
        except Exception:
            baud = DEFAULT_SMS_BAUD

        host = self.tb_host_var.get().strip().strip("/")
        token = self.tb_token_var.get().strip()
        apn = self.tb_apn_var.get().strip()

        if serial is None:
            self.post("tb", f"ThingsBoard error: pyserial not installed ({SERIAL_IMPORT_ERROR}). Run: python -m pip install pyserial")
            return
        if not port:
            self.post("tb", "ThingsBoard error: SIM7600 AT port is empty")
            return
        if not host:
            self.post("tb", "ThingsBoard error: host is empty")
            return
        if not token:
            self.post("tb", "ThingsBoard error: device access token is empty")
            return
        if not apn:
            self.post("tb", "ThingsBoard error: APN is empty")
            return

        # If an old thread already exited, recover automatically.
        if self.tb_upload_running and not self.tb_worker_is_alive():
            self.tb_upload_running = False
            self.tb_thread = None

        if self.tb_upload_running:
            age = time.time() - float(self.tb_upload_started_at or 0.0)
            if hasattr(self, "tb_status_var"):
                self.tb_status_var.set(f"ThingsBoard busy: previous upload active for {age:.1f}s")
            return

        self.tb_upload_running = True
        self.tb_upload_started_at = time.time()
        self.tb_thread = threading.Thread(
            target=self._thingsboard_thread,
            args=(port, baud, apn, host, token, dict(payload), reason),
            daemon=True,
        )
        self.tb_thread.start()

    def _thingsboard_thread(self, port, baud, apn, host, token, payload, reason):
        try:
            # Share the same modem lock as SMS so SMS and HTTP never fight for the SIM7600 port.
            with self.sms_lock:
                self.post("tb", f"ThingsBoard uploading ({reason}) via {port}...")
                status_code = self.send_thingsboard_via_sim7600(port, baud, apn, host, token, payload)
                self.post("tb", f"ThingsBoard upload OK ({reason}), HTTP {status_code}")
                self.tb_last_error = ""
        except Exception as e:
            self.tb_last_error = str(e)
            self.post("tb", f"ThingsBoard upload failed ({reason}): {e}")
        finally:
            self.tb_upload_running = False
            self.tb_upload_started_at = 0.0

    def send_http_action_post(self, ser, timeout=35.0):
        ser.reset_input_buffer()
        ser.write(b"AT+HTTPACTION=1\r")
        ser.flush()
        first = self.serial_read_until(ser, expected=["OK", "ERROR"], timeout=5.0)
        if "ERROR" in first and "OK" not in first:
            raise RuntimeError(f"HTTPACTION start failed: {first.strip()}")

        action = self.serial_read_until(ser, expected=["+HTTPACTION:", "ERROR"], timeout=timeout)
        combined = (first or "") + (action or "")
        if "ERROR" in action and "+HTTPACTION:" not in action:
            raise RuntimeError(f"HTTPACTION failed: {combined.strip() or 'timeout'}")

        status_code = None
        for line in combined.replace("\r", "\n").split("\n"):
            line = line.strip()
            if "+HTTPACTION:" not in line:
                continue
            try:
                values = line.split(":", 1)[1].strip().split(",")
                # Format: +HTTPACTION: <method>,<status_code>,<data_len>
                if len(values) >= 2:
                    status_code = int(values[1].strip())
                    break
            except Exception:
                pass

        if status_code is None:
            raise RuntimeError(f"No HTTP status from modem: {combined.strip() or 'timeout'}")
        if status_code < 200 or status_code >= 300:
            raise RuntimeError(f"ThingsBoard HTTP status {status_code}: {combined.strip()}")
        return status_code

    def send_thingsboard_via_sim7600(self, port, baud, apn, host, token, payload):
        url = f"http://{host}/api/v1/{token}/telemetry"
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        payload_bytes = payload_json.encode("ascii", errors="ignore")

        try:
            ser_ctx = serial.Serial(port=port, baudrate=int(baud), timeout=0.5, write_timeout=8, exclusive=False)
        except Exception as e:
            owner = self.port_owner_text(port)
            hint = (
                f"Could not open {port}: {e}. "
                "Close minicom/picocom/screen/serial monitor or stop ModemManager if it owns the port."
            )
            if owner:
                hint += f"\nPort owner info:\n{owner[:500]}"
            raise RuntimeError(hint)

        with ser_ctx as ser:
            time.sleep(0.2)
            self.send_at(ser, "AT", expected="OK", timeout=2.0)
            self.send_at(ser, "ATE0", expected="OK", timeout=2.0)

            # Network/PDP setup. Repeated commands are safe and help after modem reconnects.
            try:
                self.send_at(ser, "AT+CMEE=2", expected="OK", timeout=2.0)
            except Exception:
                pass
            self.send_at(ser, f'AT+CGDCONT=1,"IP","{apn}"', expected="OK", timeout=4.0)
            self.send_at(ser, "AT+CGATT=1", expected="OK", timeout=10.0)
            self.send_at(ser, "AT+CGACT=1,1", expected="OK", timeout=12.0)

            # Clean any previous HTTP session. Ignore failure because HTTP may not be initialized yet.
            try:
                self.send_at(ser, "AT+HTTPTERM", expected="OK", timeout=3.0)
            except Exception:
                pass

            try:
                self.send_at(ser, "AT+HTTPINIT", expected="OK", timeout=5.0)
                self.send_at(ser, 'AT+HTTPPARA="CID",1', expected="OK", timeout=3.0)
                self.send_at(ser, f'AT+HTTPPARA="URL","{url}"', expected="OK", timeout=5.0)
                self.send_at(ser, 'AT+HTTPPARA="CONTENT","application/json"', expected="OK", timeout=3.0)

                ser.reset_input_buffer()
                ser.write(f"AT+HTTPDATA={len(payload_bytes)},{DEFAULT_TB_HTTP_TIMEOUT_MS}\r".encode("ascii"))
                ser.flush()
                prompt = self.serial_read_until(ser, expected=["DOWNLOAD", "ERROR"], timeout=6.0)
                if "DOWNLOAD" not in prompt:
                    raise RuntimeError(f"No HTTPDATA DOWNLOAD prompt: {prompt.strip() or 'timeout'}")

                ser.write(payload_bytes)
                ser.flush()
                data_resp = self.serial_read_until(ser, expected=["OK", "ERROR"], timeout=12.0)
                if "OK" not in data_resp:
                    raise RuntimeError(f"HTTPDATA not accepted: {data_resp.strip() or 'timeout'}")

                return self.send_http_action_post(ser, timeout=35.0)
            finally:
                try:
                    self.send_at(ser, "AT+HTTPTERM", expected="OK", timeout=4.0)
                except Exception:
                    pass

    def close(self):
        self.stop_monitoring()
        self.root.destroy()


def main():
    ensure_dirs()
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
