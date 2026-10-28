"""Match detections to ground truth by mask IoU. Shared by the detector tests and the evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TruthFrame:
    stem: str
    image: np.ndarray  # BGR
    instance_map: np.ndarray  # int32, piece id per pixel, -1 for belt
    record: dict  # the saved truth json


def load_truth_frame(root: str | Path, stem: str) -> TruthFrame:
    root = Path(root)
    image = cv2.imread(str(root / "images" / f"{stem}.jpg"))
    inst = cv2.imread(str(root / "inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
    record = json.loads((root / "truth" / f"{stem}.json").read_text())
    return TruthFrame(stem, image, inst, record)


def frame_stems(root: str | Path) -> list[str]:
    index = json.loads((Path(root) / "index.json").read_text())
    return [r["frame"] for r in index]


@dataclass
class MatchResult:
    tp: int = 0
    fn: int = 0
    fp: int = 0  # detections matching no piece at all
    partial_ok: int = 0  # detections of pieces too hidden to count (valid, not penalised)
    ious: list[float] = field(default_factory=list)
    matched: list[tuple[int, int]] = field(default_factory=list)  # (detection index, piece id)
    missed_ids: list[int] = field(default_factory=list)
    fp_indices: list[int] = field(default_factory=list)

    def __iadd__(self, other: MatchResult) -> MatchResult:
        self.tp += other.tp
        self.fn += other.fn
        self.fp += other.fp
        self.partial_ok += other.partial_ok
        self.ious += other.ious
        return self


def match_masks(
    masks: list[np.ndarray],
    inst: np.ndarray,
    pieces: list[dict],
    min_visibility: float = 0.5,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy one-to-one matching by IoU against the visible region of each truth piece.

    Pieces below ``min_visibility`` cannot be fairly demanded, so missing them is not a miss and
    detecting them is not a false positive.
    """
    res = MatchResult()
    gt = {p["id"]: inst == p["id"] for p in pieces}
    counted = [p["id"] for p in pieces if p["visibility"] >= min_visibility]
    iou = np.zeros((len(masks), len(pieces)))
    ids = [p["id"] for p in pieces]
    for i, m in enumerate(masks):
        for j, pid in enumerate(ids):
            union = np.count_nonzero(m | gt[pid])
            iou[i, j] = np.count_nonzero(m & gt[pid]) / union if union else 0.0
    used_det, used_gt = set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < iou_threshold:
            break
        if i in used_det or ids[j] in used_gt or ids[j] not in counted:
            continue
        used_det.add(i)
        used_gt.add(ids[j])
        res.tp += 1
        res.ious.append(float(iou[i, j]))
        res.matched.append((int(i), ids[j]))
    res.missed_ids = [pid for pid in counted if pid not in used_gt]
    res.fn = len(res.missed_ids)
    for i in range(len(masks)):
        if i in used_det:
            continue
        if iou.shape[1] and iou[i].max() >= iou_threshold:
            res.partial_ok += 1
        else:
            res.fp += 1
            res.fp_indices.append(i)
    return res
