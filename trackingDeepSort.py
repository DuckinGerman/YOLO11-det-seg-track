import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os
import numpy as np
import torch
import random

def initialize_yolo(det_model_path, seg_model_path):
    det_model = YOLO(det_model_path)  # 目标检测 + 追踪
    seg_model = YOLO(seg_model_path)  # 实例分割

    device = torch.device("cuda:2")
    det_model.to(device)
    seg_model.to(device)

    # det_model.model.half()
    # seg_model.model.half()
    return det_model, seg_model

def initialize_tracker():
    device = torch.device("cuda:2")
    torch.cuda.set_device(2)
    return DeepSort(
        max_age=40,            # 最大未更新帧数
        n_init=6,              # 初始化需要的最少帧数
        max_iou_distance=0.6,  # 最大IOU匹配距离
        embedder="osnet",
        embedder_gpu=True,     # 使用 mobilenet 作为特征提取器
        half=False,              # 使用半精度 (FP16) 提升性能（需要支持）
    )

def initialize_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cant open file：{video_path}")
        exit(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"frame resolution: {width}x{height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, out, width, height

def process_frame(frame, det_model,seg_model, tracker,
                  class_names,detection_colors,segmentation_colors,
                  tracking_txt_file, frame_id):
    """for single frame , detect and track"""
    # 1. det
    det_results = det_model(frame, conf=0.5, iou=0.5) # get result from detector
    detections = []

    for result in det_results:
        boxes = result.boxes.xywh  # box info
        confidences = result.boxes.conf  # conf
        class_ids = result.boxes.cls  # cls

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            box_np = box.cpu().numpy()
            if np.any(np.isnan(box_np)) or np.any(np.isinf(box_np)):
                print("NaN or Inf in detection box:", box, conf)
                continue
            if box_np[2] <= 0 or box_np[3] <= 0:
                print("Invalid w,h:", box, conf)
                continue
            detections.append((box, conf, int(class_id)))

    # 2. for every frame deepsort
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        sort_x1, sort_y1, sort_x2, sort_y2 = map(float, track.to_ltrb()) # center ， rightunder
        class_id = track.det_class
        class_name = class_names[class_id]

        w = int(sort_x2 - sort_x1)
        h = int(sort_y2 - sort_y1)
        re_x1 = int(sort_x1 - w/2)  # 左上角坐标
        re_x2 = int(sort_x1 + w/2)
        re_y1 = int(sort_y1 + h/2)
        re_y2 = int(sort_y1 - h/2)

        # pick a color from the 29 detection colors
        det_color = detection_colors[class_id]  # or: DETECTION_COLORS[cls_id % 29]

        cv2.rectangle(frame, (re_x1, re_y1), (re_x2, re_y2), det_color, 2)
        # cv2.rectangle(frame, (int(sort_x1), int(sort_y1)), (int(sort_x2), int(sort_y2)), det_color, 2)

        label = f"ID:{track_id} {class_name}"
        cv2.putText(frame, label, (re_x1, re_y1 - h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1)

        print(f"[Track ID {track_id}]")
        # print(f"  sort: x1={sort_x1:.2f}, y1={sort_y1:.2f}, x2={sort_x2:.2f}, y2={sort_y2:.2f}")
        print(f"  re  : x1={re_x1}, y1={re_y1}, x2= {re_x2},y2={re_y2},w={re_x2 - re_x1}--> h={re_y2 - re_y1}")

        # 计算 MOT 格式的 bbox: x, y, w, h
        w = sort_x2 - sort_x1
        h = sort_y2 - sort_y1

        # 2. 计算左上角
        x = sort_x1
        y = sort_y1
        # 3. 可选安全检查
        if w <= 0 or h <= 0:
            print(f"  跳过非法框  frame={frame_id}  id={track_id}")
            continue

        # 置信度暂用 1.0（因为 deep sort 没返回原始 confidence）
        print(f"  re  : x1={x}, y1={y}, w={w}, h={h}")
        tracking_txt_file.write(
            f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.00,{class_id},-1\n"
        )

    # 3. seg
    seg_results = seg_model(frame, conf=0.5, iou=0.6)
    # first color all seg area with green
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
            # alpha = 0.5  # 0~1
            # cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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
    video_path = os.path.join(base_path,  "test_video/GX010067.MP4")
    output_path = os.path.join(base_path, "GX010067_deep.mp4")
    det_model_path = os.path.join(base_path, "vasco/train/det_nopretrain_16/weights/best.pt")
    seg_model_path = os.path.join(base_path, "vasco/train/seg_pre16_cls7/weights/best.pt")

    tracking_txt_path = os.path.join(base_path, "results", "GX010067_deepsort.txt")
    os.makedirs(os.path.dirname(tracking_txt_path), exist_ok=True)
    tracking_txt_file = open(tracking_txt_path, "w")
    # 1. initial model
    det_model, seg_model = initialize_yolo(det_model_path, seg_model_path)
    # 2. initial tracker
    tracker = initialize_tracker()

    # 3. initial video and read
    cap, out, width, height = initialize_video(video_path, output_path)

    # 4. initial color
    NUM_DET = 29  # det cls
    NUM_SEG = 7  # seg cls

    # 前 29 个用于检测, 后 7 个用于分割
    segmentation_colors = vivid_palette(NUM_SEG)
    detection_colors = soft_palette(NUM_DET)

    all_frame_start_time = time.time()
    frame_count = 1

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # 检查是否到达视频结尾
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Reading video is over ")
                break
            else:
                print("Reading current frame wrong !")
                continue

        frame_count += 1
        single_frame_start_time = time.time() # 单帧开始时间

        # process
        class_names = list(det_model.names.values())
        processed_frame = process_frame(frame, det_model,seg_model, tracker,
                                        class_names,detection_colors,segmentation_colors,
                                        tracking_txt_file, frame_count)
        single_frame_end_time = time.time() # 单帧结束时间

        # show result
        # cv2.imshow("YOLO11 + DeepSORT", processed_frame)

        # write video
        out.write(processed_frame)
        single_frame_processing_time = single_frame_end_time - single_frame_start_time
        print(f"Single frame time: {single_frame_processing_time:.2f} s")

        # exit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    all_frame_end_time = time.time()
    all_frame_processing_time = all_frame_end_time - all_frame_start_time
    tracking_txt_file.close()
    print(f"Total {frame_count} frames。Saved to : {output_path}")
    print(f"All done for time: {all_frame_processing_time:.2f} s")

if __name__ == "__main__":
    main()
