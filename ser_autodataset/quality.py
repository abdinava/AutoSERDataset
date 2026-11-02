#clip cleaning and quality eval

from typing import Dict, List, Tuple

import numpy as np
import webrtcvad
from pydub import AudioSegment

from .config import DatasetConfig


def vad_speech_ratio(audio_16k_mono: AudioSegment, aggressiveness: int = 2, frame_ms: int = 20) -> float:

    vad = webrtcvad.Vad(aggressiveness)

    if (audio_16k_mono.frame_rate != 16000 or audio_16k_mono.channels != 1
            or audio_16k_mono.sample_width != 2):
        audio_16k_mono = audio_16k_mono.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    pcm = audio_16k_mono.raw_data
    sample_rate = audio_16k_mono.frame_rate
    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0) * 2)
    if bytes_per_frame <= 0:
        return 0.0

    total = voiced = 0
    for off in range(0, len(pcm) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm[off:off + bytes_per_frame]
        total += 1
        if vad.is_speech(frame, sample_rate):
            voiced += 1

    return (voiced / total) if total else 0.0


def estimate_clipping_ratio(clip: AudioSegment) -> float:

    samples = np.array(clip.get_array_of_samples()).astype(np.float32)
    if samples.size == 0:
        return 0.0
    max_possible = float(2 ** (8 * clip.sample_width - 1) - 1)
    abs_samples = np.abs(samples) / max_possible
    return float(np.mean(abs_samples >= 0.98))


def clean_clip_audio(clip: AudioSegment, cfg: DatasetConfig) -> AudioSegment:

    clip = clip.set_frame_rate(cfg.sample_rate).set_channels(cfg.channels).set_sample_width(cfg.sample_width_bytes)

    peak_dbfs = clip.max_dBFS if clip.max_dBFS != float("-inf") else -120.0
    gain_needed = cfg.target_peak_dbfs - peak_dbfs
    if gain_needed > 0:
        clip = clip.apply_gain(min(gain_needed, cfg.max_gain_db))

    return clip


def measure_clip_quality(clip: AudioSegment, cfg: DatasetConfig) -> Dict[str, float]:
    return {
        "duration_s": len(clip) / 1000.0,
        "dbfs": float(clip.dBFS if clip.dBFS != float("-inf") else -120.0),
        "peak_dbfs": float(clip.max_dBFS if clip.max_dBFS != float("-inf") else -120.0),
        "vad_ratio": float(vad_speech_ratio(clip)),
        "clipping_ratio": float(estimate_clipping_ratio(clip)),
    }


def evaluate_clip_for_dataset(clip: AudioSegment, cfg: DatasetConfig) -> Tuple[bool, Dict[str, float], List[str]]:
    metrics = measure_clip_quality(clip, cfg)
    reasons = []

    if metrics["duration_s"] < cfg.min_len_s:
        reasons.append("too_short_after_cleaning")
    if metrics["duration_s"] > cfg.max_len_s:
        reasons.append("too_long_after_cleaning")
    if metrics["dbfs"] < cfg.min_dbfs:
        reasons.append("too_quiet")
    if metrics["peak_dbfs"] < cfg.min_peak_dbfs:
        reasons.append("peak_too_low")
    if metrics["vad_ratio"] < cfg.min_vad_ratio:
        reasons.append("not_enough_speech")
    if metrics["clipping_ratio"] > cfg.max_clipping_ratio:
        reasons.append("too_much_clipping")

    return (len(reasons) == 0), metrics, reasons
