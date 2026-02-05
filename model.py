from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


DEFAULT_SR = 16000
FEATURE_VERSION = 2


@dataclass
class ModelBundle:
    cough_model: RandomForestClassifier
    cough_labels: LabelEncoder


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_audio_bytes(audio_bytes: bytes, target_sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        y = f.read(dtype="float32")
        sr = f.samplerate
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    if len(y) < sr // 4:
        y = np.pad(y, (0, sr // 4 - len(y)))

    y = y / (np.max(np.abs(y)) + 1e-6)

    n_fft = 1024
    hop = 256

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64)
    log_mel = librosa.power_to_db(mel)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]

    def stats(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x.mean(axis=1), x.std(axis=1)])

    features = np.concatenate(
        [
            stats(mfcc),
            stats(mfcc_delta),
            stats(mfcc_delta2),
            stats(log_mel),
            stats(chroma),
            stats(contrast),
            [zcr.mean(), zcr.std(), rms.mean(), rms.std(), centroid.mean(), rolloff.mean()],
        ]
    )
    return features.astype(np.float32)


def _speaker_signature(speaker_idx: int, length: int, sr: int) -> np.ndarray:
    base_freq = 110 + speaker_idx * 18
    t = np.linspace(0, length / sr, num=length, endpoint=False)
    signal = (
        0.55 * np.sin(2 * np.pi * base_freq * t)
        + 0.25 * np.sin(2 * np.pi * (base_freq * 2.05) * t)
        + 0.20 * np.sin(2 * np.pi * (base_freq * 3.1) * t)
    )
    return signal


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def _highpass(x: np.ndarray, win: int) -> np.ndarray:
    return x - _moving_average(x, win)


def _lowpass(x: np.ndarray, win: int) -> np.ndarray:
    return _moving_average(x, win)


def _cough_burst(length: int, sr: int, cough_type: str) -> np.ndarray:
    y = np.zeros(length, dtype=np.float32)
    if cough_type == "dry":
        bursts = np.random.randint(2, 4)
        for _ in range(bursts):
            max_start = max(1, length - sr // 8)
            start = np.random.randint(0, max_start)
            burst_len = np.random.randint(sr // 40, sr // 10)
            burst_len = min(burst_len, length - start)
            envelope = np.hanning(burst_len)
            noise = _highpass(np.random.randn(burst_len).astype(np.float32), win=8)
            y[start : start + burst_len] += envelope * noise
    elif cough_type == "wet":
        bursts = np.random.randint(2, 5)
        for _ in range(bursts):
            max_start = max(1, length - sr // 6)
            start = np.random.randint(0, max_start)
            burst_len = np.random.randint(sr // 20, sr // 5)
            burst_len = min(burst_len, length - start)
            envelope = np.exp(-np.linspace(0, 3.0, burst_len)).astype(np.float32)
            noise = _lowpass(np.random.randn(burst_len).astype(np.float32), win=18)
            y[start : start + burst_len] += envelope * noise
    else:
        bursts = np.random.randint(1, 3)
        for _ in range(bursts):
            max_start = max(1, length - sr // 6)
            start = np.random.randint(0, max_start)
            burst_len = np.random.randint(sr // 16, sr // 6)
            burst_len = min(burst_len, length - start)
            t = np.linspace(0, burst_len / sr, num=burst_len, endpoint=False)
            tone = 0.2 * np.sin(2 * np.pi * 400 * t) * np.sin(2 * np.pi * 6 * t)
            noise = _highpass(np.random.randn(burst_len).astype(np.float32), win=12)
            envelope = np.hanning(burst_len)
            y[start : start + burst_len] += envelope * (0.7 * noise + tone)
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y


def synthesize_sample(
    speaker_idx: int,
    cough: bool,
    duration: float,
    sr: int,
    cough_type: str = "dry",
) -> np.ndarray:
    length = int(duration * sr)
    speaker = _speaker_signature(speaker_idx, length, sr)
    noise = 0.05 * np.random.randn(length).astype(np.float32)
    if cough:
        burst = _cough_burst(length, sr, cough_type=cough_type)
        y = 0.4 * speaker + 0.6 * burst + noise
    else:
        y = 0.8 * speaker + noise
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y.astype(np.float32)


def generate_synthetic_dataset(
    output_dir: Path,
    speakers: int = 5,
    samples_per_class: int = 50,
    duration: float = 1.3,
    sr: int = DEFAULT_SR,
) -> None:
    _ensure_dir(output_dir)
    metadata = []
    cough_types = ["dry", "wet", "wheeze"]
    for speaker_idx in range(1, speakers + 1):
        for cough in [True, False]:
            label = "cough" if cough else "non_cough"
            for sample_idx in range(samples_per_class):
                cough_type = np.random.choice(cough_types) if cough else "none"
                y = synthesize_sample(speaker_idx, cough, duration, sr, cough_type=cough_type)
                fname = f"speaker{speaker_idx}_{label}_{cough_type}_{sample_idx:03d}.wav"
                path = output_dir / fname
                sf.write(path, y, sr)
                metadata.append(
                    {
                        "file": str(path),
                        "speaker": f"speaker{speaker_idx}",
                        "label": label,
                        "cough_type": cough_type,
                    }
                )

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _collect_wavs(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.wav") if p.is_file()])


def build_models(data_root: Path, models_dir: Path) -> ModelBundle:
    _ensure_dir(models_dir)
    generated_dir = data_root / "generated"
    wavs = []
    if generated_dir.exists():
        wavs.extend(_collect_wavs(generated_dir))

    if not wavs:
        generate_synthetic_dataset(generated_dir)
        wavs = _collect_wavs(generated_dir)

    features = []
    cough_labels = []

    for wav_path in wavs:
        name = wav_path.stem
        parts = name.split("_")
        label = None
        for token in parts:
            if token in {"cough", "non_cough"}:
                label = token
                break
        if label is None:
            continue
        y, sr = librosa.load(wav_path, sr=DEFAULT_SR, mono=True)
        feat = extract_features(y, sr)
        features.append(feat)
        cough_labels.append(label)

    X = np.vstack(features)
    cough_enc = LabelEncoder()
    y_cough = cough_enc.fit_transform(cough_labels)

    cough_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            C=5.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        ),
    )

    cough_model.fit(X, y_cough)

    joblib.dump(
        {
            "cough_model": cough_model,
            "cough_labels": cough_enc,
            "feature_version": FEATURE_VERSION,
        },
        models_dir / "bundle.joblib",
    )

    return ModelBundle(cough_model, cough_enc)


def load_models(models_dir: Path) -> ModelBundle | None:
    bundle_path = models_dir / "bundle.joblib"
    if not bundle_path.exists():
        return None
    payload = joblib.load(bundle_path)
    if payload.get("feature_version") != FEATURE_VERSION:
        return None
    return ModelBundle(
        payload["cough_model"],
        payload["cough_labels"],
    )
