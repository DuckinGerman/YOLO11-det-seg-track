# osnet_wrapper.py  (放在与你的 trackingBotsort.py 同一目录)
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from torchreid import models
from pathlib import Path

class OSNetEncoder:
    """Return L2-normalised 256-dim features for a list of xyxy boxes."""

    def __init__(self, pth=None,device='cpu'):
        if pth is None:
            pth = (
                Path(__file__).resolve().parents[2]
                / "osnet_x0_25_imagenet.pth"
            )
        self.device = torch.device(device)
        self.net = models.osnet_x0_25(num_classes=1000)  # ← 方案 B
        state = torch.load(pth, map_location='cpu')
        self.net.load_state_dict(state, strict=True)  # ← OK
        self.net.classifier = torch.nn.Identity()  # ← 去掉分类层
        self.net.to(device).eval()

        self.tfm = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img_bgr: np.ndarray, xyxy: np.ndarray):
        """
        Args:
            img_bgr : 原始帧 (H,W,3)  BGR uint8
            xyxy    : (N,≥4)  bbox，前 4 列为 x1,y1,x2,y2
        Returns:
            list[np.ndarray]  每个元素是 512-d 向量
        """
        if xyxy.shape[0] == 0:
            return []

        H, W = img_bgr.shape[:2]
        patches = []
        for x1, y1, x2, y2 in xyxy[:, :4].astype(int):
            # 裁剪到合法区域
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:  # 极小框容错
                patch = np.zeros((256, 128, 3), dtype=np.uint8)
            else:
                patch = img_bgr[y1:y2, x1:x2, ::-1]  # BGR→RGB
            patches.append(self.tfm(patch))

        batch = torch.stack(patches).to(self.device)
        with torch.no_grad():
            feats = torch.nn.functional.normalize(self.net(batch), dim=1)

        return [f.cpu().numpy() for f in feats]