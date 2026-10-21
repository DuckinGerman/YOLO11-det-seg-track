# deep_sort_realtime/embedder/osnet_encoder.py
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import cv2

# 直接用 torchreid 自带模型
from torchreid import models

__all__ = ["OSNetEncoder"]

class OSNetEncoder:
    def __init__(self,
                 weight_path: str,
                 device="cuda:2",
                 input_size=256,
                 half: bool = False,
                 bgr: bool = True):
        """
        weight_path :  .pth
        device      : cuda:x  / cpu
        half        : 是否半精度（True 速度快，显存小）
        bgr         : 外部送进来的 frame 是否是 BGR
        """
        self.device = torch.device(device)
        self.half   = half
        self.bgr    = bgr

        # ---- 1. 建模型 ----
        self.net = models.osnet_x0_25(num_classes=0)   #
        state = torch.load(weight_path, map_location="cpu")

        # 删掉 state-dict 里关于 classifier 的参数
        for k in list(state.keys()):
            if k.startswith('classifier.'):
                del state[k]

        self.net.load_state_dict(state, strict=False)
        self.net.eval().to(self.device)
        self.size = input_size

        self.mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225])[:, None, None]
    # ------- 主调用 -------
    @torch.no_grad()
    def predict(self, crops_bgr):
        """
        crops_bgr : List[np.ndarray]  每张 crop 为 H×W×3、BGR、uint8
        returns   : List[np.ndarray]  每个元素是 1×512 的 L2-norm 特征
        """
        if len(crops_bgr) == 0:
            return []

        batch = []
        for img in crops_bgr:
            img = cv2.resize(img, (self.size, self.size))
            img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255  # BGR→RGB、CHW、0-1
            img = (img - self.mean) / self.std
            batch.append(img)

        batch = torch.from_numpy(np.stack(batch)).float().to(self.device)

        feats = self.net(batch).cpu()
        feats = torch.nn.functional.normalize(feats, dim=1)  # L2-norm
        return [f.numpy() for f in feats]