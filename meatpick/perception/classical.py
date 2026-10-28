"""Classical detector: colour distance from the belt, morphology, watershed for touching pieces.

The belt colour is estimated from each frame (it covers most of the image), so a different belt
shade or white balance needs no retuning. Work happens at reduced resolution for speed; masks are
returned at the frame's full resolution.
"""

from __future__ import annotations

import cv2
import numpy as np

from meatpick.perception.detector import MaskInstance
from meatpick.types import Frame


def _median_u8(a: np.ndarray) -> float:
    """Median of a uint8 array from its histogram (np.median is too slow for the time budget)."""
    csum = np.cumsum(np.bincount(a.ravel(), minlength=256))
    return float(np.searchsorted(csum, csum[-1] / 2))


def _reaches_edge(part: np.ndarray, edges: np.ndarray) -> bool:
    """True if the part touches a side of the blob's window that lies on the image border."""
    top, bottom, left, right = edges
    return bool(
        (top and part[0].any()) or (bottom and part[-1].any())
        or (left and part[:, 0].any()) or (right and part[:, -1].any())
    )  # fmt: skip


def _smooth(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class ClassicalDetector:
    def __init__(self, cfg: dict, px_per_mm: float):
        c = cfg["perception"]["classical"]
        p = cfg["product"]
        self.c = c
        self.s = c["work_scale"]
        self.mm2_per_px = 1.0 / (px_per_mm * self.s) ** 2  # area of one working-resolution pixel
        # plausible whole-piece areas from the product spec (area ~ 0.75-0.85 of length x width)
        self.area_lo_mm2 = 0.75 * p["length_mm"][0] * p["width_mm"][0]
        self.area_hi_mm2 = 0.85 * p["length_mm"][1] * p["width_mm"][1]
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["open_px"],) * 2)
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["close_px"],) * 2)

    # --- segmentation ---------------------------------------------------------------------
    def foreground(self, image: np.ndarray) -> np.ndarray:
        """uint8 0/1 mask of everything that is not belt, at working resolution."""
        return self._foreground(image)[0]

    def _foreground(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        small = cv2.resize(image, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        ref = [_median_u8(lab[::3, ::3, k]) for k in range(3)]
        d = lab.astype(np.float32) - np.array(ref, np.float32)
        dist = np.sqrt(
            d[..., 1] ** 2 + d[..., 2] ** 2 + (self.c["lightness_weight"] * d[..., 0]) ** 2
        )
        mask = (dist > self.c["color_threshold"]).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)  # fills holes (glare, fat, stains)
        return filled, small

    def _split(
        self, blob: np.ndarray, small_roi: np.ndarray, edges: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Split a blob of touching pieces. Seeds are the cores of the distance transform; the
        blob is then partitioned between them. Returns [blob] when there is nothing to split."""
        pad = cv2.copyMakeBorder(blob, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        dist = cv2.distanceTransform(pad, cv2.DIST_L2, 3)
        seeds = (dist > self.c["marker_fraction"] * dist.max()).astype(np.uint8)
        n, seed_labels = cv2.connectedComponents(seeds)
        inside = pad > 0
        if n - 1 < 2:
            valley = self._split_at_valleys(pad, small_roi)
            if valley is None:
                return [blob]
            n, seed_labels, seeds = valley
            method = "nearest"
        else:
            method = self.c["split_method"]
        if method == "nearest":  # nearest-seed partition (Voronoi of the cores)
            _, lab = cv2.distanceTransformWithLabels(
                (seeds == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP
            )
            lut = np.zeros(lab.max() + 1, np.int32)
            lut[lab[seeds > 0]] = seed_labels[seeds > 0]
            out = lut[lab]
        else:  # "color": watershed on the real image, edges between pieces act as barriers
            img = cv2.copyMakeBorder(small_roi, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
            out = np.where(seeds > 0, seed_labels, 0).astype(np.int32)
            cv2.watershed(cv2.GaussianBlur(img, (3, 3), 0), out)
        parts = [((out == k) & inside)[2:-2, 2:-2].astype(np.uint8) for k in range(1, n)]
        min_px = self.c["min_piece_mm2"] / self.mm2_per_px
        for part in parts:
            if part.sum() < min_px and not (edges is not None and _reaches_edge(part, edges)):
                return [blob]  # a sliver away from the frame edge means the split was spurious
        return parts

    def _split_at_valleys(self, pad: np.ndarray, small_roi: np.ndarray):
        """Overlapping pieces can form a convex blob with no waist. Their shared edge still shows
        as a dark valley (rim shading and contact shadow), so cut along it and keep the cut only
        if both sides are plausible pieces. Returns (n, labels, seeds) or None."""
        c = self.c
        gray = cv2.cvtColor(
            cv2.copyMakeBorder(small_roi, 2, 2, 2, 2, cv2.BORDER_REPLICATE), cv2.COLOR_BGR2GRAY
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["valley_px"],) * 2)
        valley = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) > c["valley_threshold"]).astype(
            np.uint8
        )
        valley = cv2.dilate(valley, np.ones((3, 3), np.uint8))
        core = cv2.morphologyEx(pad & (1 - valley), cv2.MORPH_OPEN, self.k_close)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(core)
        min_px = c["min_piece_mm2"] / self.mm2_per_px
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
        if len(keep) < 2:
            return None
        seeds = np.zeros_like(pad)
        relabeled = np.zeros(pad.shape, np.int32)
        for new, i in enumerate(keep, start=1):
            seeds[labels == i] = 1
            relabeled[labels == i] = new
        return len(keep) + 1, relabeled, seeds

    # --- scoring --------------------------------------------------------------------------
    def _confidence(self, solidity: float, area_mm2: float, border: bool) -> float:
        shape = _smooth(solidity, self.c["solidity_zero"], self.c["solidity_one"])
        high = 1.0 - _smooth(area_mm2, 1.3 * self.area_hi_mm2, 2.0 * self.area_hi_mm2)
        low = 1.0 if border else _smooth(area_mm2, 0.4 * self.area_lo_mm2, 0.7 * self.area_lo_mm2)
        return float(np.sqrt(shape * min(low, high)))

    # --- public ---------------------------------------------------------------------------
    def detect(self, frame: Frame) -> list[MaskInstance]:
        fg, small = self._foreground(frame.image)
        hs, ws = fg.shape
        h_full, w_full = frame.image.shape[:2]
        inv = 1.0 / self.s
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_px = self.c["min_area_mm2"] / self.mm2_per_px
        out: list[MaskInstance] = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_px:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            blob = np.zeros((h, w), np.uint8)
            cv2.drawContours(blob, [cnt], -1, 1, thickness=-1, offset=(-x, -y))
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0.0
            edges = np.array([y <= 0, y + h >= hs, x <= 0, x + w >= ws])
            roi = small[y : y + h, x : x + w]
            parts = (
                [blob] if solidity >= self.c["split_solidity"] else self._split(blob, roi, edges)
            )
            for part in parts:
                pc, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not pc:
                    continue
                big = max(pc, key=cv2.contourArea)
                area = float(part.sum())
                if area < min_px:
                    continue
                hull = cv2.contourArea(cv2.convexHull(big))
                sol = cv2.contourArea(big) / hull if hull > 0 else 0.0
                px, py, pw, ph = cv2.boundingRect(big)
                border = x + px <= 0 or y + py <= 0 or x + px + pw >= ws or y + py + ph >= hs
                conf = self._confidence(sol, area * self.mm2_per_px, border)
                full = np.zeros((h_full, w_full), bool)
                big_mask = cv2.resize(
                    part * 255, (round(w * inv), round(h * inv)), interpolation=cv2.INTER_LINEAR
                )
                x0, y0 = round(x * inv), round(y * inv)
                crop = big_mask[: h_full - y0, : w_full - x0] > 127
                full[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
                out.append(MaskInstance(full, conf, border))
        return out
