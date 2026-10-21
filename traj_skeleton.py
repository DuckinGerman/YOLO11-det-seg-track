from collections import defaultdict
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import torch
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes

def initialize_yolo(detection_model_path, segmentation_model_path):
    detection_model = YOLO(detection_model_path)
    segmentation_model = YOLO(segmentation_model_path)

    device = torch.device("cuda:2")
    detection_model.to(device)
    segmentation_model.to(device)
    return detection_model, segmentation_model

def initialize_tracker():
    track_history = defaultdict(list)
    tracker_name = "botsort.yaml"
    return  tracker_name, track_history

def initialize_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cant open：{video_path}")
        exit(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video solution: {width}x{height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, out, width, height


# def generate_smooth_trajectory(foot_mask_points, frame_width, frame_height):
#     # avoid BUG empty type
#     if foot_mask_points is None :
#         return [], []
#
#     # get mask points
#     mask_points = np.array(foot_mask_points)
#     # get min and max in y direction
#     min_y, max_y = np.min(mask_points[:, 1]), np.max(mask_points[:, 1])
#     # between min and max select 10 points
#     selected_y = np.linspace(min_y, max_y, num=10, dtype=int)
#     # initial x for next computation
#     x_values = []
#     # previous_x = frame_width // 2  # 初始化为帧中心
#
#     for y in selected_y:
#         x_at_y = mask_points[mask_points[:, 1] == y][:, 0]  # 获取该 y 值对应的 x 值
#         if len(x_at_y) > 0:
#             x_val = int(np.mean(x_at_y))
#             # previous_x = x_val  # 更新 previous_x
#         # x_values.append(previous_x)  # 赋值，即便没有数据，也使用上一点的 x 值
#         x_values.append(x_val)
#
#     # 过滤掉过大或过小的异常点
#     x_values = scipy.signal.medfilt(x_values, kernel_size=5)  # 使用中值滤波平滑
#
#     # 使用插值生成更平滑的轨迹
#     y_smooth = np.linspace(min_y, max_y, num=40)
#     x_smooth = interp1d(selected_y, x_values, kind='linear', fill_value='extrapolate')(y_smooth)
#     trajectory_points = list(zip(x_smooth.astype(int), y_smooth.astype(int)))
#
#     # 确保起点固定为帧底部中心
#     # trajectory_points[-1] = (frame_width // 2, frame_height )
#     # 在轨迹上均匀取 10 个点作为箭头标记
#     arrow_indices = np.linspace(0, len(trajectory_points) - 1, num=10, dtype=int)
#     arrow_points = [trajectory_points[i] for i in arrow_indices]
#
#     return trajectory_points, arrow_points


def select_foot_region(seg_results, frame_width, frame_height):
    """
    seg_result: 单帧的分割结果(ultralytics的 Result 对象)
    frame_width, frame_height: 当前帧宽和高
    offset: 离帧底部向上偏移多少像素，表示"脚下"

    return:
        foot_mask_points (np.ndarray) or None
    """
    # seg_result.masks.xy: list of 多边形, 每个多边形是一个 [N,2] 的数组 (x, y)
    if seg_results[0].masks is None or len(seg_results[0].masks.xy) == 0:
        return None

    # 定义 "脚下参考点"：帧中水平居中，在底部往上偏移 offset
    px = frame_width // 2
    py = frame_height - 30

    # 遍历每个mask
    for mask_points in seg_results[0].masks.xy:
        # 转成 int32
        polygon_int = np.array(mask_points, dtype=np.int32)
        # 调用 pointPolygonTest 判断 (px,py) 是否在多边形内部
        # 若返回值 >= 0，表示点在多边形边界或内部
        dist = cv2.pointPolygonTest(polygon_int, (px, py), False)
        if dist >= 0:
            # 找到第一个包含该点的mask，就返回
            return polygon_int

    return None

def extract_skeleton(mask_2d):
    """
    对二值图 mask_2d 做骨架提取。

    参数:
    -------
    mask_2d : np.ndarray
        二值掩码，形状 (H, W)，例如 0/1 或 0/255
    min_size : int
        保留的最小连通区域像素数(去除非常小的噪点)

    返回:
    -------
    skeleton : np.ndarray(dtype=bool)
        骨架图，形状 (H, W)，值为 True/False，表示是否在骨架上
    """

    # 1. 若 mask_2d 是 0~255 的 uint8，先把它转成 0/1 的 bool
    if mask_2d.dtype != bool:
        mask_bool = mask_2d > 127  # 简单阈值
    else:
        mask_bool = mask_2d

    # 2. 去除小区域 (remove_small_objects)
    #    这一步能剔除小噪点，比如 min_size=50 表示小于 50 像素的独立连通域会被移除
    mask_clean = remove_small_objects(mask_bool, min_size=50)

    # 3. 去除小孔洞 (remove_small_holes), 可以设置阈值
    mask_clean = remove_small_holes(mask_clean, area_threshold=50)

    # 4. (可选) 做形态学开/闭运算，进一步平滑边缘
    #    这里用OpenCV演示一个简单的开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 注意: remove_small_objects返回的是bool类型，需要转uint8再给cv2处理
    mask_clean_uint8 = (mask_clean.astype(np.uint8)) * 255
    mask_morph = cv2.morphologyEx(mask_clean_uint8, cv2.MORPH_OPEN, kernel)

    # 5. skeletonize 细化
    mask_morph_bool = (mask_morph > 127)
    skeleton = skeletonize(mask_morph_bool)  # 返回 bool 类型，True=骨架像素

    return skeleton

def polygon_to_bool_mask(foot_mask_points, height, width):
    """
    将单个多边形顶点数组 foot_mask_points (N×2) 转换成 bool 类型的 2D mask，shape=(height, width)。
    """
    # 1) 创建一张空白 (height, width) 的uint8数组
    mask_2d = np.zeros((height, width), dtype=np.uint8)

    # 2) 用 fillPoly 把多边形区域填充成前景 (255)
    #    注意 fillPoly 的坐标形式是 (x, y)
    cv2.fillPoly(mask_2d, [foot_mask_points], 255)

    # 3) 转为bool => True/False
    mask_bool = (mask_2d > 127)

    return mask_bool


def process_frame(frame, detection_model, segmentation_model, tracker_name):

    """
    detection_model
    segmentation_model
    tracker_name
    det_results
    seg_results
    """
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # det+track
    det_results = list(detection_model.track(frame, tracker_name,conf=0.3, iou=0.4,persist=True,save=False)) # get result
    if len(det_results) > 0:
        frame = det_results[0].plot()

    # segmentation
    seg_results = segmentation_model(frame, conf=0.3, iou=0.4)
    # first color all seg area with green
    if len(seg_results) > 0:
        masks = seg_results[0].masks
        if masks is not None:
            for mask in masks.xy:
                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(frame, [mask], (0, 255, 0))  # mask area with green

        # trajectory
        # 1. select current seg walkable area
        foot_mask_points = select_foot_region(seg_results,frame_width,frame_height)
        if foot_mask_points is not None:
            cv2.fillPoly(frame, [foot_mask_points], (0, 0, 255))
            # 2. get mask bool
            mask_bool = polygon_to_bool_mask(foot_mask_points, frame_height, frame_width)
            # skeleton is bool type
            skeleton = skeletonize(mask_bool)

            points = np.argwhere(skeleton == True)  # shape = (N, 2), [row, col] = [y, x]
            # 这里 points[:,0] 是 y 坐标, points[:,1] 是 x 坐标

            for (y, x) in points:
                cv2.circle(frame, (x, y), radius=1, color=(0, 255, 255), thickness=-1)

    return frame


def main():
    base_path = os.getcwd()

    # video_path = 0
    # video_path = os.path.join(base_path, "video_test_handy/IMG_1967.mp4")
    video_path = os.path.join(base_path, "GX010067.MP4")
    output_path = os.path.join(base_path, "GX010067_tra_skeleton.mp4")
    detection_model_path = os.path.join(base_path, "vasco/train/det_pretrain/weights/best.pt")
    segmentation_model_path = os.path.join(base_path, "vasco/train/seg_nopretrain/weights/best.pt")

    # initial model
    detection_model, segmentation_model = initialize_yolo(detection_model_path, segmentation_model_path)

    # initial tracker
    tracker_name, track_history = initialize_tracker()

    # initial video and read
    cap, out, width, height = initialize_video(video_path, output_path)

    all_frame_start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # check if its end
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Reading video is over")
                break
            else:
                print("Reading frame error")
                continue


        single_frame_start_time = time.time() # 单帧开始时间

        # Processing
        processed_frame = process_frame(frame, detection_model, segmentation_model, tracker_name)
        frame_count += 1
        single_frame_end_time = time.time() # 单帧结束时间

        # Show result
        # cv2.imshow("YOLO11 Tracking", processed_frame)

        # write the video
        out.write(processed_frame)
        single_frame_processing_time = single_frame_end_time - single_frame_start_time
        print(f"Single frame time: {single_frame_processing_time:.2f} s")

        # press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    all_frame_end_time = time.time()
    all_frame_processing_time = all_frame_end_time - all_frame_start_time
    print(f"Total {frame_count} frames。Saved to : {output_path}")
    print(f"All done for time: {all_frame_processing_time:.2f} s")

if __name__ == "__main__":
    main()
