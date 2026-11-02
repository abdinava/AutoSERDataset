#Audio extraction and loading.

#ffmpeg handles extraction videoaudio formats

import os
import subprocess

from pydub import AudioSegment

from .config import DatasetConfig


def extract_audio_16k_mono(in_path: str, out_path: str, cfg: DatasetConfig) -> str:

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    audio_filters = "highpass=f=60,lowpass=f=7600"
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-vn",
        "-ac", "1",
        "-ar", str(cfg.sample_rate),
        "-af", audio_filters,
        "-f", "wav",
        out_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {in_path}:\n{result.stderr.decode(errors='ignore')[-1000:]}"
        )
    return out_path


def load_audiosegment(path_wav_16k: str, cfg: DatasetConfig) -> AudioSegment:
    a = AudioSegment.from_wav(path_wav_16k)
    if a.frame_rate != cfg.sample_rate:
        a = a.set_frame_rate(cfg.sample_rate)
    if a.channels != cfg.channels:
        a = a.set_channels(cfg.channels)
    if a.sample_width != cfg.sample_width_bytes:
        a = a.set_sample_width(cfg.sample_width_bytes)
    return a
