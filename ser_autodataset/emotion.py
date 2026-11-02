"""Emotion labeling with emotion2vec+ (notebook Step 7).

funasr is imported lazily so it only loads when labeling actually runs.
"""

from typing import Any, Dict

from .config import DatasetConfig

EMOTION_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
]


def normalize_emotion_label(label: str) -> str:
    label = str(label).strip().split("/")[-1]
    return "unknown" if label == "<unk>" else label


_EMOTION_MODEL_CACHE: Dict[str, Any] = {}


def get_emotion_model(cfg: DatasetConfig):
    if cfg.emotion_model_id not in _EMOTION_MODEL_CACHE:
        from funasr import AutoModel
        _EMOTION_MODEL_CACHE[cfg.emotion_model_id] = AutoModel(model=cfg.emotion_model_id, hub="hf")
    return _EMOTION_MODEL_CACHE[cfg.emotion_model_id]


def classify_audio_emotion(wav_path: str, emotion_model, output_dir: str = "emotion_outputs") -> dict:
    res = emotion_model.generate(
        wav_path, output_dir=output_dir, granularity="utterance", extract_embedding=False
    )
    if isinstance(res, list):
        res = res[0] if res else {}

    labels = [normalize_emotion_label(x) for x in (res.get("labels") or [])]
    scores = [float(x) for x in (res.get("scores") or [])]
    label_to_score = {label: 0.0 for label in EMOTION_LABELS}
    for label, score in zip(labels, scores):
        label_to_score[label] = score

    if label_to_score:
        top_emotion = max(label_to_score, key=label_to_score.get)
        top_score = float(label_to_score[top_emotion])
    else:
        top_emotion, top_score = "unknown", 0.0

    result = {"predicted_emotion": top_emotion, "emotion_confidence": top_score}
    for emotion_name in EMOTION_LABELS:
        result[f"score_{emotion_name}"] = float(label_to_score.get(emotion_name, 0.0))
    return result
