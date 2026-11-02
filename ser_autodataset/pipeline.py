#Defines the pipeline.
#the main pipeline w emotion labeling

#Input: audio/video file
#Output: labeled clips

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from .audio_io import extract_audio_16k_mono, load_audiosegment
from .chunking import build_sentence_aware_chunks
from .config import DatasetConfig
from .emotion import EMOTION_LABELS, classify_audio_emotion, get_emotion_model
from .quality import clean_clip_audio, evaluate_clip_for_dataset
from .segmentation import split_long_clip_on_pauses, trim_edges_to_nonsilence
from .transcription import transcribe_segments


def process_media_file(
    input_path: str,
    out_dir: str,
    cfg: DatasetConfig,
) -> str:

    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(out_dir, "_tmp_16k.wav")
    extract_audio_16k_mono(input_path, wav_path, cfg)

    audio = load_audiosegment(wav_path, cfg)
    segments = transcribe_segments(wav_path, cfg)
    chunks = build_sentence_aware_chunks(segments, cfg)

    exported = 0
    accepted_meta, rejected_meta = [], []

    for ch in tqdm(chunks, desc=f"Exporting {Path(input_path).name}"):
        start_ms, end_ms = trim_edges_to_nonsilence(audio, int(ch.start_s * 1000), int(ch.end_s * 1000), cfg)
        if end_ms <= start_ms:
            rejected_meta.append({
                "start_s": ch.start_s, "end_s": ch.end_s, "text": ch.text,
                "boundary_reason": ch.boundary_reason, "rejection_reasons": ["invalid_trimmed_bounds"],
            })
            continue

        base_clip = audio[start_ms:end_ms]
        subclips = split_long_clip_on_pauses(base_clip, cfg)

        for sub_idx, (subclip, rel_start_ms, rel_end_ms) in enumerate(subclips):
            abs_start_ms = start_ms + rel_start_ms
            abs_end_ms = start_ms + rel_end_ms

            cleaned_clip = clean_clip_audio(subclip, cfg)
            keep, metrics, reasons = evaluate_clip_for_dataset(cleaned_clip, cfg)

            row = {
                "source_file": os.path.basename(input_path),
                "start_s": abs_start_ms / 1000.0,
                "end_s": abs_end_ms / 1000.0,
                "text": ch.text,
                "boundary_reason": ch.boundary_reason,
                "subclip_index": sub_idx,
                **metrics,
            }

            if not keep:
                rejected_meta.append({**row, "rejection_reasons": reasons})
                continue

            fname = f"chunk_{exported:05d}_{abs_start_ms/1000:.2f}s_{abs_end_ms/1000:.2f}s.wav"
            out_path = os.path.join(out_dir, fname)
            cleaned_clip.export(out_path, format="wav")

            accepted_meta.append({"file": fname, "chunk_path": out_path, **row})
            exported += 1

    with open(os.path.join(out_dir, "chunks_meta.json"), "w", encoding="utf-8") as f:
        json.dump(accepted_meta, f, ensure_ascii=False, indent=2)
    pd.DataFrame(accepted_meta).to_csv(os.path.join(out_dir, "chunks_meta.csv"), index=False)

    with open(os.path.join(out_dir, "rejected_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(rejected_meta, f, ensure_ascii=False, indent=2)

    summary = {
        "input_file": input_path,
        "segments_found": len(segments),
        "chunk_candidates": len(chunks),
        "accepted_chunks": len(accepted_meta),
        "rejected_chunks": len(rejected_meta),
        "config": asdict(cfg),
    }
    with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        os.remove(wav_path)
    except OSError:
        pass

    return out_dir


def create_emotion_labeled_dataset(chunks_dir: str, cfg: DatasetConfig) -> Tuple[str, str]:

    with open(os.path.join(chunks_dir, "chunks_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    emotion_model = get_emotion_model(cfg)
    rows = []

    package_dir = os.path.join(chunks_dir, "emotion_labeled_dataset")
    os.makedirs(package_dir, exist_ok=True)
    for emotion_name in EMOTION_LABELS:
        os.makedirs(os.path.join(package_dir, emotion_name), exist_ok=True)

    for item in tqdm(meta, desc="Emotion labeling"):
        prediction = classify_audio_emotion(
            wav_path=item["chunk_path"],
            emotion_model=emotion_model,
            output_dir=os.path.join(chunks_dir, "emotion_outputs"),
        )

        emotion_dir = os.path.join(package_dir, prediction["predicted_emotion"])
        labeled_audio_path = os.path.join(emotion_dir, item["file"])
        shutil.copy2(item["chunk_path"], labeled_audio_path)

        rows.append({
            **item,
            "labeled_audio_path": labeled_audio_path,
            "relative_labeled_audio_path": os.path.relpath(labeled_audio_path, package_dir),
            "model_id": cfg.emotion_model_id,
            **prediction,
        })

    with open(os.path.join(package_dir, "emotion_labeled_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(package_dir, "emotion_labeled_dataset.csv"), index=False)

    zip_path = shutil.make_archive(os.path.join(chunks_dir, "emotion_labeled_dataset"), "zip", package_dir)
    return zip_path, os.path.join(package_dir, "emotion_labeled_dataset.json")
