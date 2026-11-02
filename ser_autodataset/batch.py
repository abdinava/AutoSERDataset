#Batch processing of a ZIP of media files

import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from .config import DatasetConfig
from .pipeline import create_emotion_labeled_dataset, process_media_file

MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def safe_name(name: str, max_len: int = 80) -> str:

    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return (stem or "media")[:max_len]


def safe_extract_zip(zip_path: str, extract_dir: str) -> str:

    extract_root = Path(extract_dir).resolve()
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            member_path = Path(member.filename)
            if member.is_dir():
                continue
            if "__MACOSX" in member_path.parts or member_path.name.startswith("._"):
                continue

            target_path = (extract_root / member_path).resolve()
            try:
                target_path.relative_to(extract_root)
            except ValueError:
                print("Skipping unsafe ZIP entry:", member.filename)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    return str(extract_root)


def find_media_files(root_dir: str) -> List[str]:

    root = Path(root_dir)
    found = [
        str(p) for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
        and "__MACOSX" not in p.parts and not p.name.startswith("._")
    ]
    return sorted(found)


def combine_chunk_dirs(chunk_dirs: List[str], combined_dir: str) -> Tuple[str, List[dict]]:

    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)
    os.makedirs(combined_dir, exist_ok=True)

    combined_meta, combined_rejected = [], []

    for source_idx, chunks_dir in enumerate(chunk_dirs):
        meta_path = os.path.join(chunks_dir, "chunks_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for item in meta:
                old_chunk_path = item.get("chunk_path")
                if not old_chunk_path or not os.path.exists(old_chunk_path):
                    continue
                source_file = item.get("source_file") or Path(old_chunk_path).stem
                new_file = f"{source_idx:04d}_{safe_name(source_file)}_{item.get('file') or Path(old_chunk_path).name}"
                new_chunk_path = os.path.join(combined_dir, new_file)
                shutil.copy2(old_chunk_path, new_chunk_path)

                updated = dict(item)
                updated["file"] = new_file
                updated["chunk_path"] = new_chunk_path
                updated["source_chunks_dir"] = chunks_dir
                combined_meta.append(updated)

        rejected_path = os.path.join(chunks_dir, "rejected_chunks.json")
        if os.path.exists(rejected_path):
            with open(rejected_path, "r", encoding="utf-8") as f:
                rejected = json.load(f)
            for item in rejected:
                combined_rejected.append({**item, "source_chunks_dir": chunks_dir})

    with open(os.path.join(combined_dir, "chunks_meta.json"), "w", encoding="utf-8") as f:
        json.dump(combined_meta, f, ensure_ascii=False, indent=2)
    pd.DataFrame(combined_meta).to_csv(os.path.join(combined_dir, "chunks_meta.csv"), index=False)
    with open(os.path.join(combined_dir, "rejected_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(combined_rejected, f, ensure_ascii=False, indent=2)

    summary = {
        "source_chunk_dirs": chunk_dirs,
        "accepted_chunks": len(combined_meta),
        "rejected_chunks": len(combined_rejected),
    }
    with open(os.path.join(combined_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return combined_dir, combined_meta


def process_zip_of_media(
    zip_path: str,
    cfg: DatasetConfig,
    extract_dir: str = "input_media",
    per_file_chunks_root: str = "chunks_by_file",
    combined_chunks_dir: str = "combined_chunks_out",
) -> Tuple[str, str]:

    for d in (extract_dir, per_file_chunks_root):
        if os.path.exists(d):
            shutil.rmtree(d)

    safe_extract_zip(zip_path, extract_dir)
    media_paths = find_media_files(extract_dir)
    if not media_paths:
        raise RuntimeError(f"No video/audio files found inside ZIP: {zip_path}")
    print(f"Found {len(media_paths)} media file(s).")

    chunk_dirs = []
    for idx, media_path in enumerate(tqdm(media_paths, desc="Processing files")):
        out_dir = os.path.join(per_file_chunks_root, f"{idx:04d}_{safe_name(media_path)}")
        process_media_file(input_path=media_path, out_dir=out_dir, cfg=cfg)
        chunk_dirs.append(out_dir)

    combined_dir, combined_meta = combine_chunk_dirs(chunk_dirs, combined_chunks_dir)
    if not combined_meta:
        raise RuntimeError(
            "The ZIP was processed, but no chunks passed the filters. "
            "Try lowering min_vad_ratio, min_dbfs, or min_len_s in DatasetConfig."
        )
    print(f"Combined accepted chunks: {len(combined_meta)}")

    return create_emotion_labeled_dataset(chunks_dir=combined_dir, cfg=cfg)
