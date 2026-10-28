"""Texture bank: real meat patches from the dataset, with a procedural fallback.

``TextureBank.sample(kind, size, rng)`` is the only interface the rest of the sim uses.
``kind`` is ``"lean"`` or ``"fat"``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

KINDS = ("lean", "fat")
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _value_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 4,
    base_cells: int = 4,
    anisotropy: float = 1.0,
) -> np.ndarray:
    """Smooth multi-octave noise in [0, 1]. anisotropy > 1 stretches features along x."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = base_cells * 2**o
        gh, gw = max(2, cells), max(2, int(cells / anisotropy))
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return np.clip((out - out.min()) / (np.ptp(out) + 1e-6), 0, 1)


def procedural_lean(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Red-pink noise with marbling streaks. size is (h, w). Returns BGR uint8."""
    h, w = size
    base = np.array(
        [rng.uniform(95, 125), rng.uniform(100, 135), rng.uniform(190, 225)], np.float32
    )
    n = _value_noise(size, rng, octaves=5, base_cells=3)
    img = base[None, None, :] * (0.78 + 0.4 * n[..., None])
    streaks = _value_noise(size, rng, octaves=4, base_cells=6, anisotropy=0.25)
    marble = np.clip((streaks - 0.72) / 0.2, 0, 1)[..., None]
    cream = np.array([185, 205, 235], np.float32)
    img = img * (1 - 0.4 * marble) + cream * 0.4 * marble
    grain = rng.normal(0, 4, (h, w, 1)).astype(np.float32)
    return np.clip(img + grain, 0, 255).astype(np.uint8)


def procedural_fat(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Cream, low-saturation fat. size is (h, w). Returns BGR uint8."""
    h, w = size
    base = np.array(
        [rng.uniform(195, 215), rng.uniform(220, 235), rng.uniform(235, 248)], np.float32
    )
    n = _value_noise(size, rng, octaves=4, base_cells=3)
    img = base[None, None, :] * (0.93 + 0.1 * n[..., None])
    grain = rng.normal(0, 2.5, (h, w, 1)).astype(np.float32)
    return np.clip(img + grain, 0, 255).astype(np.uint8)


_PROCEDURAL = {"lean": procedural_lean, "fat": procedural_fat}


def _class_of(label_path: Path) -> int | None:
    if not label_path.exists():
        return None
    for line in label_path.read_text().splitlines():
        if line.strip():
            return int(line.split()[0])
    return None


def _valid_integral(bgr: np.ndarray, black_v: int) -> np.ndarray:
    """Integral image of the 'real content' mask, so any square can be tested in O(1)."""
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    valid = (v > black_v).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((11, 11), np.uint8))
    return cv2.integral(valid)


_FAT_CREAM_BGR = np.array([205, 228, 242], np.float32)


def _tint_fat(patch: np.ndarray) -> np.ndarray:
    """Dataset 'fat' patches are pale pink; keep their detail but pull the color to cream."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = (gray - gray.mean()) * 0.6
    cream = _FAT_CREAM_BGR[None, None, :] + detail[..., None]
    return np.clip(0.7 * cream + 0.3 * patch, 0, 255).astype(np.uint8)


class TextureBank:
    def __init__(self, cfg: dict, rng: np.random.Generator, use_dataset: bool = True):
        self.cfg = cfg["textures"]
        self.patch_px = self.cfg["patch_px"]
        self.patches: dict[str, list[np.ndarray]] = {k: [] for k in KINDS}
        self.n_source_images = 0
        if use_dataset:
            self._load(Path(self.cfg["dataset_dir"]), rng)

    def _load(self, root: Path, rng: np.random.Generator) -> None:
        c = self.cfg
        img_dir, lbl_dir = root / c["split"] / "images", root / c["split"] / "labels"
        if not img_dir.is_dir():
            return
        files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for path in files:
            if _class_of(lbl_dir / (path.stem + ".txt")) not in c["classes"]:
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            self.n_source_images += 1
            self._extract(bgr, rng)

    def _extract(self, bgr: np.ndarray, rng: np.random.Generator) -> None:
        c, p = self.cfg, self.cfg["patch_px"]
        h, w = bgr.shape[:2]
        if h < p or w < p:
            return
        integ = _valid_integral(bgr, c["border_black_v"])
        for _ in range(c["patches_per_image"] * 3):  # oversample; some tries get rejected
            y, x = int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))
            if integ[y + p, x + p] - integ[y, x + p] - integ[y + p, x] + integ[y, x] < p * p:
                continue  # touches the black rotation border
            patch = bgr[y : y + p, x : x + p]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            if hsv[..., 2].mean() < c["min_brightness"]:
                continue
            if (
                cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                < c["min_sharpness"]
            ):
                continue
            is_fat = (
                hsv[..., 2].mean() >= c["fat_min_brightness"]
                and hsv[..., 1].mean() <= c["fat_max_saturation"]
            )
            self.patches["fat" if is_fat else "lean"].append(patch.copy())

    def has_real(self, kind: str) -> bool:
        return len(self.patches[kind]) >= self.cfg["min_patches_per_kind"]

    def sample(
        self, kind: str, size: int | tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        """Return a BGR uint8 texture of ``size`` (int -> square, or (h, w))."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        h, w = (size, size) if isinstance(size, int) else size
        if not self.has_real(kind):
            return _PROCEDURAL[kind]((h, w), rng)
        pool = self.patches[kind]
        patch = pool[int(rng.integers(len(pool)))]
        patch = np.rot90(patch, int(rng.integers(4)))
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        out = cv2.resize(np.ascontiguousarray(patch), (w, h), interpolation=cv2.INTER_AREA)
        return _tint_fat(out) if kind == "fat" else out


def contact_sheet(
    bank: TextureBank, rng: np.random.Generator, n: int = 48, cols: int = 8, tile: int = 128
) -> np.ndarray:
    """n samples, first half lean and second half fat, in a labeled grid."""
    tiles = []
    for i in range(n):
        kind = "lean" if i < n * 3 // 4 else "fat"
        t = bank.sample(kind, tile, rng).copy()
        cv2.putText(t, kind[0].upper(), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, n, cols)]
    return np.vstack(rows)
