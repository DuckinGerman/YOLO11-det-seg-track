# import torch
# import  torchvision
# 加载pth格式的模型
# model = torch.load('/home/tili/model_train/yolo11_deepsort/osnet_x0_25_imagenet.pth')
# 将模型保存为pt格式
# torch.save(model, '/home/tili/model_train/yolo11_deepsort/osnet_x0_25_imagenet.pt')
# 将模型转换为pb格式
# pb_model = torchvision.models.save_pb(model, 'model.pb')


#############
# save_osnet_ultra.py
import torch, types
from torch import nn

# 1. 载入 state_dict（官方 .pth）
state = torch.load('osnet_x0_25_imagenet.pth', map_location='gpu')

# 2. 构建 OSNet 主干
#    - 如果你已安装 torchreid：
from torchreid import models
backbone = models.build_model('osnet_x0_25', num_classes=0)
backbone.load_state_dict(state)
backbone.eval()

# 3. 包一层，保持输入→特征输出
class ReIDWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    @torch.no_grad()
    def forward(self, x):
        return self.net(x)

model = ReIDWrapper(backbone)

# 4. 按 Ultralytics 格式存 ckpt
ckpt = {'model': model, 'ema': None}
torch.save(ckpt, 'osnet_x0_25_ultra.pt')
print('✓ Saved as osnet_x0_25_ultra.pt')