
#speech transcription with faster-whisper

from typing import TYPE_CHECKING, Dict, List, Tuple

from .config import DatasetConfig

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

_WHISPER_CACHE: Dict[Tuple[str, str, str], "WhisperModel"] = {}


def get_whisper_model(cfg: DatasetConfig) -> "WhisperModel":
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    key = (cfg.whisper_model, device, compute_type)

    if key not in _WHISPER_CACHE:
        _WHISPER_CACHE[key] = WhisperModel(
            cfg.whisper_model, device=device, compute_type=compute_type
        )
    return _WHISPER_CACHE[key]


# NOTE:
# Returns Whisper segments as plain dicts: {"start": float, "end": float, "text": str}
def transcribe_segments(wav_16k_path: str, cfg: DatasetConfig) -> List[dict]:

    model = get_whisper_model(cfg)
    segments_iter, _info = model.transcribe(
        wav_16k_path,
        language=cfg.whisper_language,
        word_timestamps=True,
        beam_size=5,
    )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments_iter
    ]
