"""Tracker: stable IDs and smoothed poses for pieces moving with the belt.

Positions are in the belt frame (x along travel from the camera's first row, y across) and are
stored at a reference encoder count. Between frames a piece moves exactly as far as the belt, so
the position at another encoder count is ``x + (encoder - ref_encoder) * mm_per_count``: tracks
are predicted by dead reckoning and keep coasting after they leave the camera view, until they are
picked or pass the end of the reach zone.

A detection cut by the frame edge has a biased centroid. It can extend a track but never
overwrites its pose; the pose comes from detections that are fully in view.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from meatpick.perception.detector import touches_border as _touches_border
from meatpick.perception.detector import touches_bottom
from meatpick.types import Detection, Frame, Track

NO_MATCH = 1e9
LIVE = ("tentative", "confirmed", "assigned")


def track_position_at(track: Track, encoder: int, mm_per_count: float) -> tuple[float, float]:
    """Where the track's piece is at another encoder count (it moves only along belt x)."""
    return track.x_mm + (encoder - track.ref_encoder) * mm_per_count, track.y_mm


def _angle_median(angles: list[float]) -> float:
    """Median of angles that repeat every 180 degrees, measured around the first one."""
    ref = angles[0]
    offsets = [((a - ref + 90.0) % 180.0) - 90.0 for a in angles]
    return float((ref + np.median(offsets)) % 180.0)


@dataclass
class _State:
    track: Track
    angles: list[float] = field(default_factory=list)
    fat_votes: list[int] = field(default_factory=list)
    full_hits: int = 0
    half_extent_x: float = 0.0  # half the piece's extent along belt x, from its last clean view
    misses: int = 0  # consecutive frames unseen while in the interior of the view


class Tracker:
    def __init__(self, cfg: dict, calibration=None):
        p = cfg["perception"]
        self.t = p["tracker"]
        self.gate = p["track_gate_mm"]
        self.confirm_hits = p["track_confirm_hits"]
        self.mm_per_count = cfg["belt"]["mm_per_encoder_count"]
        if calibration is not None:  # the tilted camera's real view, from calibration
            self.x_top, self.x_bot = calibration.view_span_x_mm()
        else:
            self.x_top = 0.0
            self.x_bot = cfg["camera"]["height_px"] / (
                cfg["camera"]["width_px"] / cfg["camera"]["fov_mm"][0]
            )
        self.reach_end = cfg["robot"]["reach_zone_x_mm"][1]
        self._states: dict[int, _State] = {}
        self._next_id = 0
        self._last_encoder: int | None = None
        self._last_t: float | None = None
        self.speed_mm_s = 0.0  # belt speed estimate
        self.finished: list[Track] = []  # picked or missed, drained by the caller

    # --- queries --------------------------------------------------------------------------
    @property
    def tracks(self) -> list[Track]:
        """Snapshots: later updates never change a Track a caller is holding."""
        return [replace(s.track) for s in self._states.values()]

    def get(self, track_id: int) -> Track | None:
        s = self._states.get(track_id)
        return replace(s.track) if s else None

    def drain_finished(self) -> list[Track]:
        out, self.finished = self.finished, []
        return out

    # --- status changes from the scheduler ------------------------------------------------
    def mark_assigned(self, track_id: int) -> None:
        if track_id in self._states and self._states[track_id].track.status == "confirmed":
            self._states[track_id].track.status = "assigned"

    def mark_picked(self, track_id: int) -> None:
        s = self._states.pop(track_id, None)
        if s is not None:
            s.track.status = "picked"
            self.finished.append(replace(s.track))

    # --- main step ------------------------------------------------------------------------
    def update(self, frame: Frame, detections: list[Detection]) -> list[Track]:
        enc, now = frame.encoder, frame.t_s
        self._advance(enc, now)
        margin = self.t["fov_margin_mm"]
        in_view = [
            s
            for s in self._states.values()
            if self.x_top - margin <= s.track.x_mm <= self.x_bot + margin
        ]
        borders = [_touches_border(d.mask) for d in detections]

        matches = self._associate(in_view, detections, borders)
        matched_states = set()
        for si, di in matches:
            self._observe(in_view[si], detections[di], borders[di], enc)
            matched_states.add(si)
        matched_dets = {di for _, di in matches}

        for si, s in enumerate(in_view):
            if si in matched_states:
                continue
            interior = self.x_top + margin <= s.track.x_mm <= self.x_bot - margin
            if s.track.status == "tentative":
                del self._states[s.track.id]  # a real piece shows up in consecutive frames
            elif interior:
                s.misses += 1
                if s.misses >= self.t["interior_miss_limit"]:
                    del self._states[s.track.id]  # lost inside the view: do not keep a ghost

        live = [s.track for s in self._states.values()]
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            near = any(
                np.hypot(det.x_mm - tr.x_mm, det.y_mm - tr.y_mm) < self.t["spawn_suppress_mm"]
                for tr in live
            )
            exiting = touches_bottom(det.mask)  # pieces only ever enter at the top
            if not near and not exiting:  # else: a fragment, or a piece already tracked
                self._spawn(det, borders[di], enc)

        self._retire(now)
        return self.tracks

    # --- internals ------------------------------------------------------------------------
    def _advance(self, enc: int, now: float) -> None:
        """Dead-reckon every track to this frame's encoder count; update the belt speed."""
        if self._last_encoder is not None and now > self._last_t and enc >= self._last_encoder:
            inst = (enc - self._last_encoder) * self.mm_per_count / (now - self._last_t)
            a = self.t["speed_ema"]
            self.speed_mm_s = (
                inst if self.speed_mm_s == 0.0 else a * inst + (1 - a) * self.speed_mm_s
            )
        self._last_encoder, self._last_t = enc, now
        for s in self._states.values():
            s.track.x_mm += (enc - s.track.ref_encoder) * self.mm_per_count
            s.track.ref_encoder = enc

    def _expected_x(self, s: _State) -> float:
        """Where the detector's centroid should be: a piece cut by the frame edge shows only its
        visible part, whose centroid lies up to half the hidden length away from the true centre."""
        x = s.track.x_mm
        if s.full_hits == 0 or s.half_extent_x <= 0:
            return x
        lo, hi = max(x - s.half_extent_x, self.x_top), min(x + s.half_extent_x, self.x_bot)
        return 0.5 * (lo + hi) if hi > lo else x

    def _associate(self, states, dets, borders) -> list[tuple[int, int]]:
        if not states or not dets:
            return []
        cost = np.full((len(states), len(dets)), NO_MATCH)
        for i, s in enumerate(states):
            ex = self._expected_x(s)
            for j, d in enumerate(dets):
                dist = float(np.hypot(ex - d.x_mm, s.track.y_mm - d.y_mm))
                gate = self.gate + (self.t["border_gate_extra_mm"] if borders[j] else 0.0)
                if dist <= gate:
                    cost[i, j] = dist
        rows, cols = linear_sum_assignment(cost)
        return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if cost[i, j] < NO_MATCH]

    def _observe(self, s: _State, det: Detection, partial: bool, enc: int) -> None:
        tr = s.track
        tr.hits += 1
        tr.last_obs_encoder = enc
        tr.touching = det.touching
        s.misses = 0
        tr.confidence = 0.5 * tr.confidence + 0.5 * det.confidence
        if not partial:
            self._blend_pose(s, det)
        elif s.full_hits == 0 and not touches_bottom(det.mask):
            # Still entering: the visible part's centroid creeps at half the belt speed, so dead
            # reckoning it would drift away. Follow the measurement until a clean view arrives.
            # Once the piece is sliding out of the bottom edge the same creep runs the other way
            # (the centroid lags the true centre), so dead-reckon from the last position instead:
            # a piece clipped at a side edge all the way down never gets a clean view.
            tr.x_mm, tr.y_mm = det.x_mm, det.y_mm
        if tr.status == "tentative" and tr.hits >= self.confirm_hits:
            tr.status = "confirmed"

    def _blend_pose(self, s: _State, det: Detection) -> None:
        tr, a = s.track, self.t["center_alpha"]
        if s.full_hits == 0:  # first clean view replaces whatever a partial view gave
            tr.x_mm, tr.y_mm = det.x_mm, det.y_mm
        else:
            tr.x_mm += a * (det.x_mm - tr.x_mm)
            tr.y_mm += a * (det.y_mm - tr.y_mm)
        s.full_hits += 1
        s.angles = (s.angles + [det.angle_deg])[-self.t["angle_window"] :]
        s.fat_votes = (s.fat_votes + [det.fat_side])[-self.t["angle_window"] :]
        tr.angle_deg = _angle_median(s.angles)
        tr.fat_side = 1 if sum(s.fat_votes) >= 0 else -1
        tr.length_mm, tr.width_mm, tr.est_weight_g = det.length_mm, det.width_mm, det.est_weight_g
        a = np.radians(tr.angle_deg)
        s.half_extent_x = 0.5 * (det.length_mm * abs(np.cos(a)) + det.width_mm * abs(np.sin(a)))

    def _spawn(self, det: Detection, partial: bool, enc: int) -> None:
        tr = Track(
            id=self._next_id, x_mm=det.x_mm, y_mm=det.y_mm, ref_encoder=enc,
            angle_deg=det.angle_deg, fat_side=det.fat_side, confidence=det.confidence,
            deadline_s=float("inf"), status="tentative", touching=det.touching,
            length_mm=det.length_mm, width_mm=det.width_mm, est_weight_g=det.est_weight_g,
            hits=1, last_obs_encoder=enc,
        )  # fmt: skip
        self._next_id += 1
        s = _State(tr)
        if not partial:
            s.full_hits, s.angles, s.fat_votes = 1, [det.angle_deg], [det.fat_side]
            a = np.radians(det.angle_deg)
            s.half_extent_x = 0.5 * (det.length_mm * abs(np.cos(a)) + det.width_mm * abs(np.sin(a)))
        self._states[tr.id] = s

    def _retire(self, now: float) -> None:
        """Drop tracks that passed the reach zone (misses) and refresh every deadline."""
        for tid in list(self._states):
            tr = self._states[tid].track
            if tr.x_mm > self.reach_end:
                tr.status = "missed" if tr.status != "tentative" else tr.status
                if tr.status == "missed":
                    self.finished.append(replace(tr))
                del self._states[tid]
                continue
            if tr.status == "tentative" and tr.x_mm > self.x_bot + self.t["fov_margin_mm"]:
                del self._states[tid]  # left the view without ever being confirmed
                continue
            v = self.speed_mm_s
            tr.deadline_s = now + (self.reach_end - tr.x_mm) / v if v > 1e-6 else float("inf")
