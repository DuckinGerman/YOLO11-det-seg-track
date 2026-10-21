from collections import defaultdict
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import random
import torch

def initialize_yolo(detection_model_path, segmentation_model_path):
    detection_model_byte = YOLO(detection_model_path)
    segmentation_model_byte = YOLO(segmentation_model_path)
    device = torch.device("cuda:3")
    detection_model_byte.to(device)
    segmentation_model_byte.to(device)
    # detection_model.model.half()
    # segmentation_model.model.half()
    return detection_model_byte, segmentation_model_byte

def initialize_tracker():
    track_history = defaultdict(list)
    tracker_name = "bytetrack.yaml"
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

# def generate_smooth_trajectory(segmentation_result, frame_width, frame_height):
#     # avoid BUG empty type
#     if segmentation_result.masks is None or len(segmentation_result.masks.xy) == 0:
#         return [], []
#     # get mask points
#     mask_points = segmentation_result.masks.xy[0]  # 获取分割区域的边界点
#     mask_points = np.array(mask_points)
#
#     min_y, max_y = np.min(mask_points[:, 1]), np.max(mask_points[:, 1])
#     selected_y = np.linspace(min_y, max_y, num=20, dtype=int)
#     x_values = []
#     previous_x = frame_width // 2  # 初始化为帧中心
#
#     for y in selected_y:
#         x_at_y = mask_points[mask_points[:, 1] == y][:, 0]  # 获取该 y 值对应的 x 值
#         if len(x_at_y) > 0:
#             x_val = int(np.mean(x_at_y))
#             previous_x = x_val  # 更新 previous_x
#         x_values.append(previous_x)  # 赋值，即便没有数据，也使用上一点的 x 值
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
#     trajectory_points[-1] = (frame_width // 2, frame_height )
#     # 在轨迹上均匀取 10 个点作为箭头标记
#     arrow_indices = np.linspace(0, len(trajectory_points) - 1, num=10, dtype=int)
#     arrow_points = [trajectory_points[i] for i in arrow_indices]
#
#     return trajectory_points, arrow_points

def process_frame(frame, detection_model_byte, segmentation_model_byte,
                  tracker_name,track_history,detection_colors,segmentation_colors,
                  tracking_txt_file, frame_id):

    """
    detection_model
    segmentation_model
    tracker_name
    det_results
    seg_results
    """

    # det+track
    results = list(detection_model_byte.track(frame,tracker_name,conf=0.4, iou=0.5,persist=True))
    # print(detection_model.track.)
    if len(results) > 0:
        # frame = det_results[0].plot()  # engine/results/plot
        # TODO: add botsort reid tracker
        result = results[0]
        boxes = result.boxes  # Boxes 对象
        class_names = result.names

        for box in boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())  # 获取 track ID

            # 记录 track_history
            if track_id not in track_history:
                track_history[track_id] = []  # 初始化列表

            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 左上(x1,y1), 右下(x2,y2)
            w = x2 - x1
            h = y2 - y1
            # 获取置信度 (可能是一个0~1浮点数)
            conf = float(box.conf[0].item())

            # 获取类别ID
            cls_id = int(box.cls[0].item())
            class_name = class_names[cls_id]

            # pick a color from the 29 detection colors
            det_color = detection_colors[cls_id]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),det_color, 2)

            label_str = f"{track_id}{class_name} "
            cv2.putText(frame, label_str, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
            # 获取 Track ID（必需）
            # if box.id is not None:
            #     track_id = int(box.id.item())

                # 写入 MOT 格式追踪行
            tracking_txt_file.write(
                f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls_id},-1\n"
            )

            # 存入 track_history
            track_history[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))

    # segmentation
    seg_results = segmentation_model_byte(frame, conf=0.5, iou=0.6)
    if len(seg_results) > 0:
        # frame = seg_results[0].plot()
        masks = seg_results[0].masks
        boxes = seg_results[0].boxes
        if masks is not None:
            # overlay = frame.copy()
            for box, mask in zip(boxes, masks.xy):
                seg_id = int(box.cls[0].item())
                seg_color = segmentation_colors[seg_id]

                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(frame, [mask], seg_color)  # mask area with green
            # cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        # TODO : try to find a stable traj
        # trajectory
        # trajectory , arrow_points = generate_smooth_trajectory(seg_results[0], frame.shape[1], frame.shape[0])
        # if trajectory:
        #     for i in range(1, len(trajectory)):
        #         cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 3)
        #     # star and end point add
        #     # cv2.circle(frame, trajectory[0], 5, (0, 255, 255), -1)
        #     # cv2.circle(frame, trajectory[-1], 5, (0, 255, 255), -1)
        #     # arrowhead
        #     for pt in arrow_points:
        #         cv2.arrowedLine(frame, pt, (pt[0], pt[1] - 15), (0, 255, 255), 5, tipLength=1)

    return frame

def vivid_palette(n):
    """返回 n 个高对比度 BGR 颜色"""
    hues = np.linspace(0, 180, n, endpoint=False)          # OpenCV: H ∈ [0,179]
    colors = []
    for h in hues:
        hsv = np.uint8([[[h, 255, 255]]])                  # S,V 都取满
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))

    random.seed(42)
    random.shuffle(colors)     # 打乱一下，避免相邻帧出现相近色
    return colors

def soft_palette(n):
    """生成‘次要’颜色，用于检测框，不要求强对比"""
    hues = np.linspace(0, 180, n, endpoint=False)
    colors = []
    for i, h in enumerate(hues):
        s = 200 if i % 2 else 150        # 交替不同饱和度
        v = 240 if i % 3 else 200        # 交替不同亮度
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))

    random.seed(2025)
    random.shuffle(colors)
    return colors

def main():
    base_path = os.getcwd()

    # video_path = 0
    # video_path = os.path.join(base_path, "video_test_handy/IMG_1967.mp4")
    video_path = os.path.join(base_path, "test_video/GX010067.MP4")
    output_path = os.path.join(base_path, "GX010067_byte.mp4")
    detection_model_path = os.path.join(base_path, "vasco/train/det_nopretrain_16/weights/best.pt")
    segmentation_model_path = os.path.join(base_path, "vasco/train/seg_pre16_cls7/weights/best.pt")

    tracking_txt_path = os.path.join(base_path, "results", "GX010067_bytesort.txt")
    os.makedirs(os.path.dirname(tracking_txt_path), exist_ok=True)
    tracking_txt_file = open(tracking_txt_path, "w")

    # initial model
    detection_model_byte, segmentation_model_byte = initialize_yolo(detection_model_path, segmentation_model_path)

    # initial tracker
    tracker_name, track_history = initialize_tracker()

    # initial video and read
    cap, out, width, height = initialize_video(video_path, output_path)

    # initial color
    NUM_DET = 29   # det cls
    NUM_SEG = 7    # seg cls
    # all_colors = generate_color_palette(NUM_DET + NUM_SEG)  # 36 distinct BGR colors

    # 前 29 个用于检测, 后 7 个用于分割
    segmentation_colors = vivid_palette(NUM_SEG)
    detection_colors = soft_palette(NUM_DET)

    all_frame_start_time = time.time()
    frame_count = 1

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
        processed_frame = process_frame(frame, detection_model_byte, segmentation_model_byte,
                                        tracker_name,track_history,detection_colors,segmentation_colors,
                                        tracking_txt_file, frame_count)
        frame_count += 1
        single_frame_end_time = time.time() # 单帧结束时间

        single_frame_processing_time = single_frame_end_time - single_frame_start_time
        print(f"Single frame time: {single_frame_processing_time:.2f} s")
        # if frame_count > 10:
            # print(type(detection_model.predictor._tracker))
        # Show result
        # cv2.imshow("YOLO11 Tracking", processed_frame)

        # write the video
        out.write(processed_frame)

        # press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    all_frame_end_time = time.time()
    all_frame_processing_time = all_frame_end_time - all_frame_start_time
    tracking_txt_file.close()
    # print(type(detection_model.predictor.tracker))
    # if hasattr(detection_model, "predictor"):
    #     delattr(detection_model, "predictor")
    print(f"Total {frame_count} frames。Saved to : {output_path}")
    print(f"All done for time: {all_frame_processing_time:.2f} s")

if __name__ == "__main__":
    main()