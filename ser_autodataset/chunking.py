# Group transcript segments into nat good chunks

import re
from dataclasses import dataclass
from typing import List, Tuple

from .config import DatasetConfig

_SENT_END_RE = re.compile(r"[.!?。！？]\s*$")
_SOFT_BOUNDARY_RE = re.compile(r"[,;:،，؛…]\s*$")


def is_sentence_end(text: str) -> bool:
    return bool(_SENT_END_RE.search((text or "").strip()))


def is_soft_boundary(text: str) -> bool:
    return bool(_SOFT_BOUNDARY_RE.search((text or "").strip()))


@dataclass
class Chunk:
    start_s: float
    end_s: float
    text: str
    boundary_reason: str = "unknown"


def build_sentence_aware_chunks(segments: List[dict], cfg: DatasetConfig) -> List[Chunk]:

    chunks: List[Chunk] = []
    if not segments:
        return chunks

    i = 0
    n = len(segments)

    while i < n:
        seg = segments[i]
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        text_parts = [(seg.get("text") or "").strip()]
        last_end_s = end_s

        candidates: List[Tuple[float, int, str]] = []

        if is_sentence_end(text_parts[-1]):
            candidates.append((last_end_s, i, "sentence_end"))
        elif is_soft_boundary(text_parts[-1]):
            candidates.append((last_end_s, i, "soft_punctuation"))

        j = i + 1
        while j < n:
            next_seg = segments[j]
            ns = float(next_seg["start"])
            ne = float(next_seg["end"])
            nt = (next_seg.get("text") or "").strip()

            gap_s = max(0.0, ns - last_end_s)
            prospective_dur = ne - start_s

            if gap_s > cfg.max_join_gap_s:
                candidates.append((last_end_s, j - 1, "long_pause_break"))
                break

            if prospective_dur > cfg.max_len_s:
                break

            text_parts.append(nt)
            last_end_s = ne

            if is_sentence_end(nt):
                candidates.append((last_end_s, j, "sentence_end"))
            elif is_soft_boundary(nt):
                candidates.append((last_end_s, j, "soft_punctuation"))

            if gap_s >= cfg.pause_boundary_s:
                candidates.append((float(next_seg["start"]), j - 1, "pause_boundary"))

            if prospective_dur >= cfg.target_len_s and candidates:
                break

            j += 1

        best_end_s = last_end_s
        best_j = j - 1 if j > i else i
        boundary_reason = "fallback"

        valid_candidates = [
            (t, idx, reason)
            for (t, idx, reason) in candidates
            if cfg.min_len_s <= (t - start_s) <= cfg.max_len_s
        ]

        if valid_candidates:
            best_end_s, best_j, boundary_reason = min(
                valid_candidates,
                key=lambda x: abs((x[0] - start_s) - cfg.target_len_s)
            )
        else:
            total_dur = last_end_s - start_s
            if total_dur < cfg.min_len_s and j < n:
                k = j
                while k < n:
                    ne = float(segments[k]["end"])
                    if (ne - start_s) > cfg.max_len_s:
                        break
                    text_parts.append((segments[k].get("text") or "").strip())
                    last_end_s = ne
                    if is_sentence_end((segments[k].get("text") or "").strip()):
                        boundary_reason = "extended_to_sentence_end"
                    if (last_end_s - start_s) >= cfg.min_len_s:
                        break
                    k += 1
                best_end_s = last_end_s
                best_j = k - 1 if k > i else i

        chunk_text = " ".join([t for t in text_parts if t]).strip()
        chunks.append(Chunk(start_s=start_s, end_s=best_end_s, text=chunk_text, boundary_reason=boundary_reason))
        i = best_j + 1

    return chunks
