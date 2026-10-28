"""Regenerates train_yolo_colab.ipynb. Run: python notebooks/build_notebook.py"""

import json
from pathlib import Path

MD_INTRO = """# Train the meat detector (yolo11n-seg) on Colab

Trains a one-class (`meat`) instance-segmentation model on the synthetic belt frames and saves
everything to Google Drive, so a disconnected session loses at most one epoch.

**Before you run**
1. `Runtime > Change runtime type > T4 GPU`.
2. Put `dataset_yolo.zip` (from `make dataset`, in `outputs/`) in your Drive at `MyDrive/meatpick/`.
3. `Runtime > Run all`. If Colab disconnects, reconnect and run all again: it resumes from `last.pt`.

**After it finishes**: download `MyDrive/meatpick/best.pt` into the repo as `models/best.pt`.
"""

CELLS = [
    ("markdown", MD_INTRO),
    (
        "code",
        '''# Settings. The first three can be lowered for a quick smoke test.
EPOCHS = 60        # early stopping (patience below) usually ends sooner
IMGSZ = 640
BATCH = 32
PATIENCE = 10
DEVICE = 0         # GPU 0 on Colab''',
    ),
    (
        "code",
        '''%pip install -q ultralytics
import torch

assert torch.cuda.is_available() or DEVICE == "cpu", (
    "No GPU found. Use Runtime > Change runtime type > T4 GPU, then run all again."
)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")''',
    ),
    (
        "code",
        '''from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/meatpick")
RUNS = DRIVE / "runs"          # checkpoints land here every epoch, so they survive a disconnect
DRIVE.mkdir(parents=True, exist_ok=True)''',
    ),
    (
        "code",
        '''import zipfile

zip_path = DRIVE / "dataset_yolo.zip"
assert zip_path.exists(), f"Upload dataset_yolo.zip to {DRIVE} first"
DATA = Path("/content/dataset_yolo")
if not DATA.exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("/content")
# the data.yaml in the zip is relative; write an absolute one for this machine
(DATA / "data.yaml").write_text(
    f"path: {DATA}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: meat\\n"
)
for split in ("train", "val"):
    print(split, len(list((DATA / "images" / split).glob("*.jpg"))), "images")''',
    ),
    (
        "code",
        '''from ultralytics import YOLO

last = RUNS / "meat" / "weights" / "last.pt"
finished = False
if last.exists():
    # Ultralytics sets epoch = -1 in last.pt once training has completed.
    finished = torch.load(last, map_location="cpu", weights_only=False).get("epoch") == -1

if finished:
    print("Training already finished; skipping to export.")
elif last.exists():
    print("Resuming from", last)
    YOLO(str(last)).train(resume=True)
else:
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(DATA / "data.yaml"),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=2,
        cache="ram",
        project=str(RUNS),
        name="meat",
        exist_ok=True,
        seed=0,
        flipud=0.5,      # top-down view: any orientation is valid
        degrees=0.0,     # pieces already appear at every angle
        hsv_h=0.01,      # the data already varies belt colour and lighting
    )''',
    ),
    (
        "code",
        '''import json
import shutil

best = RUNS / "meat" / "weights" / "best.pt"
shutil.copy(best, DRIVE / "best.pt")

m = YOLO(str(best)).val(data=str(DATA / "data.yaml"), imgsz=IMGSZ, device=DEVICE, split="val")
summary = {
    "weights": "best.pt",
    "imgsz": IMGSZ,
    "train_images": len(list((DATA / "images" / "train").glob("*.jpg"))),
    "val_images": len(list((DATA / "images" / "val").glob("*.jpg"))),
    "box": {"precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map},
    "mask": {"precision": m.seg.mp, "recall": m.seg.mr, "mAP50": m.seg.map50, "mAP50-95": m.seg.map},
}
(DRIVE / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))''',
    ),
    (
        "code",
        '''import matplotlib.pyplot as plt

files = sorted((DATA / "images" / "val").glob("*.jpg"))[::100][:6]
results = YOLO(str(best)).predict([str(f) for f in files], imgsz=IMGSZ, conf=0.25, device=DEVICE, verbose=False)
fig, axes = plt.subplots(2, 3, figsize=(18, 7))
for ax, r in zip(axes.ravel(), results):
    ax.imshow(r.plot()[..., ::-1])
    ax.axis("off")
plt.show()''',
    ),
    (
        "markdown",
        "**Done.** Download `MyDrive/meatpick/best.pt` into the repo as `models/best.pt`. "
        "`metrics_summary.json` is next to it.",
    ),
]


def cell(kind: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


nb = {
    "cells": [cell(k, t) for k, t in CELLS],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = Path(__file__).with_name("train_yolo_colab.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
