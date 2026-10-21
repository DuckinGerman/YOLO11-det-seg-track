from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DETECTION_MODELS = {
    "det_1": BASE_DIR / "runs/train/det_1/weights/best.pt",
    "det_nopretrain_16": BASE_DIR / "vasco/train/det_nopretrain_16/weights/best.pt",
}

SEGMENTATION_MODELS = {
    "seg_1": BASE_DIR / "runs/train/seg_1/weights/best.pt",
    "seg_pre16_cls7": BASE_DIR / "vasco/train/seg_pre16_cls7/weights/best.pt",
}

DEFAULT_DET_MODEL = "det_nopretrain_16"
DEFAULT_SEG_MODEL = "seg_pre16_cls7"