import warnings
from PIL.ImageOps import scale

warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':

    # model config
    model = YOLO(model='/home/tili/model_train/yolo11_deepsort/ultralytics/cfg/models/11/yolo11.yaml'
                 ).load('/home/tili/model_train/yolo11_deepsort/vasco/train/det_cls20/weights/best.pt')

    # datasets config
    model.train(
        data='/home/tili/model_train/yolo11_deepsort/ultralytics/cfg/datasets/vasco_test_det_cls19.yaml',
        epochs=150,     # (int) 训练的周期数
        patience=50,    # (int) 等待无明显改善以进行早期停止的周期数
        batch=16,       # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=640,      # (int) 输入图像的大小，整数或w，h
        save=True,      # (bool) 保存训练检查点和预测结果
        save_period=-1, # (int) 每x周期保存检查点（如果小于1则禁用）
        device=2 ,     # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=4,      # (int) 数据加载的工作线程数（每个DDP进程）
        project='vasco/train',   # (str, optional) 项目名称
        name='det_cls20_pluspre',             # (str, optional) 实验名称，结果保存在'project/name'目录下
        pretrained=True,        # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
        optimizer='SGD',        # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        verbose=True,           # (bool) 是否打印详细输出
        seed=0,                 # (int) 用于可重复性的随机种子
        close_mosaic=0,         # (int) 在最后几个周期禁用马赛克增强
        resume=False,           # (bool) 从上一个检查点恢复训练
        amp=False,              # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
        fraction= 1.0,                 # 使用数据集的比例，默认1.0
        profile=False,                # 是否启用ONNX或TensorRT推理性能分析，默认False
        freeze= None,                  # 冻结层的数量，默认值None
        lr0= 0.01,                     # 初始学习率，默认0.01
        lrf= 0.01,                     # 最终学习率，默认值0.01
        momentum= 0.937,               # SGD动量，默认值0.937，范围0到1
        weight_decay= 0.0005,          # 权重衰减值（L2正则化），默认值0.0005
        warmup_epochs= 3.0,            # 预热的epoch数，默认值3.0
        warmup_bias_lr= 0.1,           # 预热时的偏置学习率，默认0.1
        box= 7.5,                      # 边界框损失系数，默认值7.5
        cls= 0.5,                      # 分类损失系数，默认值0.5
        dfl= 1.5,                      # 分布式焦点损失的系数，默认值1.5
        pose= 12.0,                    # 姿态估计的损失权重，默认值12.0
        kobj= 1.0,                     # 关键点检测损失系数，默认值1.0
        nbs= 64,                       # 规范化批次大小，默认值64
        overlap_mask= True,            # 是否使用重叠掩码，默认值True
        mask_ratio= 4,               # 掩码损失权重，默认值4.0
        dropout= 0,                  # Dropout比例，默认0.0
        val= True,                     # 是否进行验证，默认True
        plots= True,                  # 是否生成训练结果的可视化图表，默认True
        # 数据增强参数
        hsv_h= 0.3,                    # 色调增强幅度，范围(0.0 - 1.0)，默认0.015
        hsv_s= 0.7,                    # 饱和度增强幅度，范围(0.0 - 1.0)，默认0.7
        hsv_v= 0.4,                    # 亮度增强幅度，范围(0.0 - 1.0)，默认0.4
        degrees= 10,                  # 随机旋转角度，范围(-180 - 180)，默认0.0
        scale= 1,                    # 随机缩放，范围(0.0 - 10.0)，默认0.5
        shear= 0.0,                    # 随机错切，范围(0.0 - 10.0)，默认0.0
        perspective= 0.0,              # 透视变换，范围(0.0 - 0.001)，默认0.0
        flipud= 0.0,                   # 垂直翻转概率，默认0.0
        fliplr= 0.5,                   # 水平翻转概率，默认0.5
        mosaic= 1.0,                   # Mosaic数据增强概率，默认值1.0
        mixup= 0.3,                    # MixUp数据增强概率，默认值0.0
        copy_paste= 0.3,               # Copy-Paste数据增强，默认值0.0
        auto_augment= 'randaugment',   # 自动增强策略，默认'randaugment'
        crop_fraction= 1.0,             # 裁剪图像的比例，默认1.0

        )
    




