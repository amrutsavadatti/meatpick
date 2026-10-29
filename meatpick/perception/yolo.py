"""YOLO instance-segmentation detector behind the same interface as the classical one."""

from __future__ import annotations

from meatpick.perception.detector import MaskInstance, touches_border
from meatpick.types import Frame


class YoloDetector:
    def __init__(self, cfg: dict, model_path: str):
        from ultralytics import YOLO  # optional dependency: only the YOLO path needs it

        self.p = cfg["perception"]
        self.model = YOLO(model_path)

    def detect(self, frame: Frame) -> list[MaskInstance]:
        r = self.model.predict(
            frame.image,
            imgsz=self.p["yolo_imgsz"],
            conf=self.p["yolo_conf"],
            iou=self.p["yolo_iou"],
            retina_masks=True,  # masks come back at the frame's full resolution
            device="cpu",
            verbose=False,
        )[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        conf = r.boxes.conf.cpu().numpy()
        out = []
        for m, c in zip(masks, conf, strict=True):
            out.append(MaskInstance(m, float(c), touches_border(m)))
        return out
