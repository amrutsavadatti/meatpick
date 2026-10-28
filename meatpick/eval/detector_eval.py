"""Run a detector over a saved truth set (dev or test) and report recall, precision, IoU, time."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meatpick.eval.matching import MatchResult, frame_stems, load_truth_frame, match_masks
from meatpick.types import Frame


@dataclass
class DetectorReport:
    n_frames: int = 0
    total: MatchResult = field(default_factory=MatchResult)
    isolated: MatchResult = field(default_factory=MatchResult)
    touching: MatchResult = field(default_factory=MatchResult)
    times_ms: list[float] = field(default_factory=list)
    misses: list[tuple[str, int]] = field(default_factory=list)  # (frame, piece id)
    false_positives: list[tuple[str, int]] = field(default_factory=list)  # (frame, mask area px)

    @staticmethod
    def _rate(num: int, den: int) -> float:
        return num / den if den else float("nan")

    def summary(self) -> dict:
        t = self.total
        return {
            "frames": self.n_frames,
            "pieces": t.tp + t.fn,
            "recall": round(self._rate(t.tp, t.tp + t.fn), 4),
            "precision": round(self._rate(t.tp, t.tp + t.fp + 0), 4),
            "recall_isolated": round(
                self._rate(self.isolated.tp, self.isolated.tp + self.isolated.fn), 4
            ),
            "recall_touching": round(
                self._rate(self.touching.tp, self.touching.tp + self.touching.fn), 4
            ),
            "pieces_touching": self.touching.tp + self.touching.fn,
            "distinct_missed_pieces": len({(s.split("_")[0], pid) for s, pid in self.misses}),
            "false_positives": t.fp,
            "partial_pieces_detected": t.partial_ok,
            "mean_iou": round(float(np.mean(t.ious)), 4) if t.ious else float("nan"),
            "ms_per_frame_mean": round(float(np.mean(self.times_ms)), 2),
            "ms_per_frame_p95": round(float(np.percentile(self.times_ms, 95)), 2),
        }


def evaluate_detector(detector, root: str | Path, limit: int | None = None) -> DetectorReport:
    rep = DetectorReport()
    stems = frame_stems(root)[: limit or None]
    for stem in stems:
        tf = load_truth_frame(root, stem)
        frame = Frame(tf.image, tf.record["encoder"], tf.record["t_s"])
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        rep.times_ms.append(1000 * (time.perf_counter() - t0))
        masks = [d.mask for d in dets]
        pieces = tf.record["pieces"]
        m = match_masks(masks, tf.instance_map, pieces)
        rep.total += m
        touching_ids = {p["id"] for p in pieces if p["touching"]}
        for p in pieces:
            if p["visibility"] < 0.5:
                continue
            hit = p["id"] not in m.missed_ids
            bucket = rep.touching if p["id"] in touching_ids else rep.isolated
            bucket.tp += hit
            bucket.fn += not hit
        rep.misses += [(stem, pid) for pid in m.missed_ids]
        rep.false_positives += [(stem, int(masks[i].sum())) for i in m.fp_indices]
        rep.n_frames += 1
    return rep
