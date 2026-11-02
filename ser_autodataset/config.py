#Dataset Config
#sets the settings for the rest of the program

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:

    # audio standardization
    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2

    # clip length target
    target_len_s: float = 6.0
    min_len_s: float = 2.5
    max_len_s: float = 9.0

    # natural utterance boundary controls
    max_join_gap_s: float = 0.75
    pause_boundary_s: float = 0.35
    edge_seek_ms: int = 900
    edge_padding_ms: int = 70
    silence_thresh_db: float = -40.0
    split_search_radius_ms: int = 2500
    split_min_silence_ms: int = 250

    # clean audio quality gates
    min_vad_ratio: float = 0.35
    min_dbfs: float = -35.0
    min_peak_dbfs: float = -30.0
    target_peak_dbfs: float = -3.0
    max_gain_db: float = 10.0
    max_clipping_ratio: float = 0.01

    # ASR model
    whisper_model: str = "small"
    whisper_language: Optional[str] = "en"

    # emotion model (emotion2vec+)
    emotion_model_id: str = "iic/emotion2vec_plus_base"
