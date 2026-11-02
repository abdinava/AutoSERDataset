#Command-line entry point
#EX:
#     python -m ser_autodataset interview.mp4
#     python -m ser_autodataset clips.zip                      # batch mode (auto-detected)
#     python -m ser_autodataset talk.mp3 --whisper-model medium --target-len 5
#     python -m ser_autodataset talk.mp3 --config my_config.json


import argparse
import json
import os
import sys
from dataclasses import fields

from .config import DatasetConfig


def build_config(args: argparse.Namespace) -> DatasetConfig:
    overrides = {}

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            overrides.update(json.load(f))
        valid = {f.name for f in fields(DatasetConfig)}
        unknown = set(overrides) - valid
        if unknown:
            raise SystemExit(f"Unknown config keys in {args.config}: {sorted(unknown)}")

    # Explicit CLI flags win over the JSON file.
    cli_map = {
        "target_len_s": args.target_len,
        "min_len_s": args.min_len,
        "max_len_s": args.max_len,
        "whisper_model": args.whisper_model,
        "emotion_model_id": args.emotion_model,
    }
    overrides.update({k: v for k, v in cli_map.items() if v is not None})
    if args.language is not None:
        overrides["whisper_language"] = None if args.language.lower() == "auto" else args.language

    return DatasetConfig(**overrides)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ser_autodataset",
        description="Build an emotion-labeled SER dataset from a video/audio file or a ZIP of them.",
    )
    p.add_argument("input", help="Path to a video/audio file, or a .zip of media files (batch mode).")
    p.add_argument("--out-dir", default="chunks_out",
                   help="Output dir for single-file mode (default: chunks_out).")
    p.add_argument("--batch", action="store_true",
                   help="Force batch (ZIP) mode. Auto-enabled for .zip inputs.")
    p.add_argument("--extract-dir", default="input_media", help="Batch: where the ZIP is extracted.")
    p.add_argument("--per-file-dir", default="chunks_by_file", help="Batch: per-file chunk output root.")
    p.add_argument("--combined-dir", default="combined_chunks_out", help="Batch: combined chunk output dir.")

    p.add_argument("--config", help="JSON file with DatasetConfig field overrides.")
    p.add_argument("--target-len", type=float, help="Target clip length in seconds (default 6.0).")
    p.add_argument("--min-len", type=float, help="Minimum clip length in seconds (default 2.5).")
    p.add_argument("--max-len", type=float, help="Maximum clip length in seconds (default 9.0).")
    p.add_argument("--whisper-model", help='faster-whisper model size/name (default "small").')
    p.add_argument("--language", help='Whisper language code, or "auto" to auto-detect (default "en").')
    p.add_argument("--emotion-model", help='emotion2vec+ model id (default "iic/emotion2vec_plus_base").')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    cfg = build_config(args)
    print("Using:", args.input)

    # Heavy imports (torch, whisper, funasr) happen only once we actually run.
    if args.batch or args.input.lower().endswith(".zip"):
        from .batch import process_zip_of_media
        zip_path, dataset_json_path = process_zip_of_media(
            zip_path=args.input,
            cfg=cfg,
            extract_dir=args.extract_dir,
            per_file_chunks_root=args.per_file_dir,
            combined_chunks_dir=args.combined_dir,
        )
    else:
        from .pipeline import create_emotion_labeled_dataset, process_media_file
        chunks_dir = process_media_file(input_path=args.input, out_dir=args.out_dir, cfg=cfg)
        print("Exported chunks in:", chunks_dir)
        zip_path, dataset_json_path = create_emotion_labeled_dataset(chunks_dir=chunks_dir, cfg=cfg)

    print("Emotion-labeled ZIP:", zip_path)
    print("Emotion-labeled JSON manifest:", dataset_json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
