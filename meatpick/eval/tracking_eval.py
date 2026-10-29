"""Tracking quality: ID switches, duplicate tracks and coverage, on saved sets and long runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from meatpick.eval.matching import load_truth_frame
from meatpick.eval.pose_eval import angle_error_deg
from meatpick.perception.tracker import Tracker, track_position_at
from meatpick.types import Frame

ACTIVE = ("confirmed", "assigned")


class TrackingScorer:
    """Links each visible truth piece to the confirmed track nearest to it, frame by frame."""

    def __init__(self, gate_mm: float = 25.0):
        self.gate = gate_mm
        self.last_track: dict = {}  # piece key -> last track id linked to it
        self.pieces: set = set()
        self.piece_frames = self.covered = self.switches = self.dup_frames = 0
        self.dup_pieces: set = set()
        self.switch_pieces: set = set()
        self.events: list[dict] = []  # first few switches and duplicates, with context
        self.frame_index = 0

    def frame(self, seq, truth_pieces: list[dict], tracks) -> dict:
        """truth_pieces: dicts with id, x_mm, y_mm, visibility. Returns piece key -> track id."""
        links = {}
        self.frame_index += 1
        active = [t for t in tracks if t.status in ACTIVE]
        for p in truth_pieces:
            if p["visibility"] < 0.5:
                continue
            key = (seq, p["id"])
            self.pieces.add(key)
            self.piece_frames += 1
            near = sorted(
                (np.hypot(t.x_mm - p["x_mm"], t.y_mm - p["y_mm"]), t.id)
                for t in active
                if np.hypot(t.x_mm - p["x_mm"], t.y_mm - p["y_mm"]) <= self.gate
            )
            if len(near) >= 2:
                self.dup_frames += 1
                if key not in self.dup_pieces and len(self.events) < 20:
                    self.events.append(self._event("duplicate", key, p, [t for _, t in near]))
                self.dup_pieces.add(key)
            if near:
                self.covered += 1
                tid = near[0][1]
                if key in self.last_track and self.last_track[key] != tid:
                    self.switches += 1
                    self.switch_pieces.add(key)
                    if len(self.events) < 20:
                        self.events.append(
                            self._event("switch", key, p, [self.last_track[key], tid])
                        )
                self.last_track[key] = tid
                links[key] = tid
        return links

    def _event(self, kind: str, key, piece: dict, tracks: list[int]) -> dict:
        return {
            "kind": kind, "piece": key[1], "frame": self.frame_index, "tracks": tracks,
            "visibility": round(piece["visibility"], 2), "touching": piece.get("touching"),
        }  # fmt: skip

    def summary(self) -> dict:
        n = len(self.pieces)
        return {
            "pieces": n,
            "piece_frames": self.piece_frames,
            "coverage_confirmed": round(self.covered / max(self.piece_frames, 1), 4),
            "id_switches": self.switches,
            "id_switches_per_500_pieces": round(500 * self.switches / max(n, 1), 2),
            "pieces_with_duplicate_tracks": len(self.dup_pieces),
            "duplicate_track_frames": self.dup_frames,
            "events": self.events,
        }


def evaluate_tracking_saved(perceiver, root: str | Path, cfg: dict) -> dict:
    """Run perception + tracker over each saved sequence (fresh tracker per sequence)."""
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    by_seq = defaultdict(list)
    for r in index:
        by_seq[r["sequence"]].append(r["frame"])
    scorer = TrackingScorer()
    for seq, stems in sorted(by_seq.items()):
        tracker = Tracker(cfg, perceiver.cal)
        for stem in stems:
            tf = load_truth_frame(root, stem)
            frame = Frame(tf.image, tf.record["encoder"], tf.record["t_s"])
            tracks = tracker.update(frame, perceiver.perceive(frame))
            scorer.frame(seq, tf.record["pieces"], tracks)
    return scorer.summary()


def evaluate_tracking_long(cfg: dict, cal, perceiver, camera, world, n_pieces: int = 500) -> dict:
    """In-memory long run. Also checks the coasted track against the true piece position while
    the piece is inside the robot's reach zone, which is where the arm will use it."""
    scorer = TrackingScorer()
    tracker = Tracker(cfg, cal)
    mm = cfg["belt"]["mm_per_encoder_count"]
    z0, z1 = cfg["robot"]["reach_zone_x_mm"]
    link: dict = {}  # world piece id -> track id (last linked while in view)
    err_xy, err_ang, fat_ok, entered, entered_tracked = [], [], [], set(), set()
    min_vis: dict[int, float] = {}  # lowest visibility while the piece is mid-view
    per_piece_err: dict[int, list[float]] = defaultdict(list)
    while True:
        world.step()
        got = camera.maybe_capture(world)
        if got is None:
            if world.n_spawned >= n_pieces and not world.pieces:
                break
            continue
        frame, truth = got
        tracks = tracker.update(frame, perceiver.perceive(frame))
        pieces = [{"id": p.id, "x_mm": p.x_mm, "y_mm": p.y_mm, "visibility": p.visibility}
                  for p in truth.pieces]  # fmt: skip
        link.update({k[1]: v for k, v in scorer.frame(0, pieces, tracks).items()})
        for p in truth.pieces:
            if 60 < p.x_mm < 300:
                min_vis[p.id] = min(min_vis.get(p.id, 1.0), p.visibility)
        by_id = {t.id: t for t in tracks}
        for wp in world.pieces:
            if wp.state != "on_belt" or not (z0 <= wp.x_mm <= z1):
                continue
            entered.add(wp.id)
            tr = by_id.get(link.get(wp.id, -1))
            if tr is None or tr.status not in ACTIVE:
                continue
            entered_tracked.add(wp.id)
            px, py = track_position_at(tr, world.belt.encoder, mm)
            err_xy.append((px - wp.x_mm, py - wp.y_mm))
            per_piece_err[wp.id].append(float(np.hypot(px - wp.x_mm, py - wp.y_mm)))
            err_ang.append(angle_error_deg(tr.angle_deg, wp.theta_deg))
            fat_ok.append(tr.fat_side == _belt_fat_side(wp))
        if world.n_spawned >= n_pieces and not world.pieces:
            break
    out = scorer.summary()
    e = np.array(err_xy)
    out.update(
        pieces_reaching_reach_zone=len(entered),
        reach_zone_tracked_fraction=round(len(entered_tracked) / max(len(entered), 1), 4),
        reach_zone_samples=len(e),
        reach_zone_position_error_mm_median=round(float(np.median(np.hypot(*e.T))), 3),
        reach_zone_position_error_mm_p95=round(float(np.percentile(np.hypot(*e.T), 95)), 3),
        reach_zone_position_error_mm_max=round(float(np.hypot(*e.T).max()), 3),
        reach_zone_angle_error_deg_median=round(float(np.median(err_ang)), 3),
        reach_zone_angle_error_deg_p95=round(float(np.percentile(err_ang, 95)), 3),
        reach_zone_fat_side_accuracy=round(float(np.mean(fat_ok)), 4),
    )
    out["reach_zone_by_visibility"] = _grouped(per_piece_err, min_vis, len(entered))
    return out


def _grouped(per_piece_err: dict, min_vis: dict, n_entered: int) -> dict:
    """Error vs the TRUE centre, split by whether another piece ever covered part of the piece.
    A covered piece's visible part has a different centroid from its true centre, so no 2D
    detector can match the truth there; the uncovered group is the fair test of the tracker."""
    groups = {"uncovered (visibility >= 0.95 mid-view)": [], "partly covered": []}
    for pid, errs in per_piece_err.items():
        key = (
            "uncovered (visibility >= 0.95 mid-view)"
            if min_vis.get(pid, 1.0) >= 0.95
            else "partly covered"
        )
        groups[key].append(float(np.median(errs)))
    out = {}
    for name, e in groups.items():
        e = np.array(e)
        out[name] = {
            "pieces": len(e),
            "median_mm": round(float(np.median(e)), 3) if len(e) else None,
            "within_2mm": round(float(np.mean(e < 2)), 4) if len(e) else None,
            "within_5mm": round(float(np.mean(e < 5)), 4) if len(e) else None,
            "over_20mm": round(float(np.mean(e > 20)), 4) if len(e) else None,
        }
    out["pieces_with_no_track"] = n_entered - len(per_piece_err)
    return out


def _belt_fat_side(wp) -> int:
    from meatpick.sim import geometry

    return geometry.fat_side_image_to_belt(wp.sprite.angle_deg, wp.sprite.fat_side)
