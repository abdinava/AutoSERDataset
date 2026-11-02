#edge trimming and pause-based splitting of chunks.
#one of the last additions mainly fro clean up and post processing

from typing import List, Optional, Tuple

from pydub import AudioSegment
from pydub.silence import detect_nonsilent, detect_silence

from .config import DatasetConfig


def trim_edges_to_nonsilence(
    audio: AudioSegment,
    start_ms: int,
    end_ms: int,
    cfg: DatasetConfig,
) -> Tuple[int, int]:

    start_ms = max(0, start_ms)
    end_ms = min(len(audio), end_ms)
    if end_ms <= start_ms:
        return start_ms, end_ms

    s_win_start = start_ms
    s_win_end = min(end_ms, start_ms + cfg.edge_seek_ms)
    start_win = audio[s_win_start:s_win_end]
    ns = detect_nonsilent(start_win, min_silence_len=120, silence_thresh=cfg.silence_thresh_db)
    if ns:
        start_ms = s_win_start + ns[0][0]

    e_win_start = max(start_ms, end_ms - cfg.edge_seek_ms)
    e_win_end = end_ms
    end_win = audio[e_win_start:e_win_end]
    ns2 = detect_nonsilent(end_win, min_silence_len=120, silence_thresh=cfg.silence_thresh_db)
    if ns2:
        end_ms = e_win_start + ns2[-1][1]

    start_ms = max(0, start_ms - cfg.edge_padding_ms)
    end_ms = min(len(audio), end_ms + cfg.edge_padding_ms)

    return start_ms, end_ms


def find_best_pause_split_ms(clip: AudioSegment, cfg: DatasetConfig) -> Optional[int]:

    if len(clip) <= int(cfg.max_len_s * 1000):
        return None

    silence_ranges = detect_silence(
        clip, min_silence_len=cfg.split_min_silence_ms, silence_thresh=cfg.silence_thresh_db
    )
    if not silence_ranges:
        return None

    target_ms = int(cfg.target_len_s * 1000)
    min_keep_ms = int(cfg.min_len_s * 1000)
    max_keep_ms = len(clip) - min_keep_ms

    candidates = [
        int((s + e) / 2) for s, e in silence_ranges
        if min_keep_ms <= int((s + e) / 2) <= max_keep_ms
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda x: abs(x - target_ms))


def split_long_clip_on_pauses(
    clip: AudioSegment, cfg: DatasetConfig
) -> List[Tuple[AudioSegment, int, int]]:

    results: List[Tuple[AudioSegment, int, int]] = []

    def _split(current_clip: AudioSegment, offset_ms: int):
        if len(current_clip) <= int(cfg.max_len_s * 1000):
            results.append((current_clip, offset_ms, offset_ms + len(current_clip)))
            return

        split_ms = find_best_pause_split_ms(current_clip, cfg)
        if split_ms is None:
            split_ms = int(cfg.target_len_s * 1000)

        left = current_clip[:split_ms]
        right = current_clip[split_ms:]

        if len(left) < int(cfg.min_len_s * 1000) or len(right) < int(cfg.min_len_s * 1000):
            results.append((current_clip, offset_ms, offset_ms + len(current_clip)))
            return

        _split(left, offset_ms)
        _split(right, offset_ms + split_ms)

    _split(clip, 0)
    return results
