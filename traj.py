from collections import defaultdict, deque
import cv2
import cv2.ximgproc as xip
from ultralytics import YOLO
import time
import os
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
import torch

# skeleton_state = {
#     "current_class_id": -1,          # 当前初始点所属的分割类别
#     "last_center_line": [],          # 上一次的中心线像素坐标列表
#     "frame_since_update": 0,         # 距离上次更新的帧数
#     "update_interval": 6             # 每 6 帧更新一次
# }

def initialize_yolo(detection_model_path, segmentation_model_path):
    detection_model = YOLO(detection_model_path)
    segmentation_model = YOLO(segmentation_model_path)
    device = torch.device("cuda:2")
    detection_model.to(device)
    segmentation_model.to(device)
    return detection_model, segmentation_model

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


def thinning_skeleton(bin_mask):
    """
    对 bin_mask(0/255 uint8) 做骨骼化，并返回骨骼图(同尺寸, 单通道0/255)
    """
    skeleton = xip.thinning(src=bin_mask, thinningType=xip.THINNING_ZHANGSUEN)
    return skeleton

def process_frame(frame, detection_model, segmentation_model, tracker_name):

    """
    detection_model
    segmentation_model
    tracker_name
    det_results
    seg_results
    """

    # det+track
    results = list(detection_model.track(frame,tracker_name,conf=0.4, iou=0.5))

    if len(results) > 0:
        # frame = det_results[0].plot()  # engine/results/plot
        # TODO: add botsort reid tracker
        result = results[0]
        boxes = result.boxes  # Boxes 对象
        class_names = result.names
        for box in boxes:
            # track_id = None
            if box.id is not None:
                track_id = int(box.id.item())  # 转成int

            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 左上(x1,y1), 右下(x2,y2)

            # 获取置信度 (可能是一个0~1浮点数)
            # conf = float(box.conf[0].item())

            # 获取类别ID
            cls_id = int(box.cls[0].item())
            class_name = class_names[cls_id]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            label_str = f"{class_name}"
            cv2.putText(frame, label_str, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            init_point = (frame.shape[1] // 2, frame.shape[0] - 30)
            cv2.circle(frame, init_point, 5, (0, 0, 255), -1)

        # ========== 2) 判断是否需要更新中心线 ===========
        # =========== 2) 分割 ===========
        seg_results = segmentation_model(frame, conf=0.4, iou=0.5)
        if len(seg_results) == 0:
            return frame  # 没有分割结果

        seg_r = seg_results[0]
        if seg_r.masks is None or seg_r.boxes is None:
            return frame

        # 在调试时，可以先用 seg_r.plot() 看看是否有分割
        # debug_img = seg_r.plot()
        # cv2.imshow("debug seg", debug_img)

        # =========== 3) 在图上画初始点(用于调试可视化) ===========
        global INIT_POINT
        init_x, init_y = INIT_POINT
        cv2.circle(frame, (init_x, init_y), 5, (0, 0, 255), -1)  # 红点
        # h, w = frame.shape[:2]
        # seg_r.masks.scale((h, w))  # 把分割掩码映射回原图大小

        # =========== 4) 找到包含 init_point 的那个实例 =============
        found_idx = -1
        for i, poly_xy in enumerate(seg_r.masks.xy):
            polygon_pts = np.array(poly_xy, dtype=np.int32)
            inside_val = cv2.pointPolygonTest(polygon_pts, (init_x, init_y), False)
            if inside_val >= 0:
                found_idx = i
                break

        if found_idx < 0:
            # 初始点不在任何掩码内
            return frame

        # =========== 5) 拿到该实例的二值mask并骨骼化 ===========
        # seg_r.masks.data[i] => shape(H,W), 0~1
        mask_data = seg_r.masks.data[found_idx].cpu().numpy()
        bin_mask = (mask_data > 0.5).astype(np.uint8) * 255

        skeleton = thinning_skeleton(bin_mask)

        poly_pts = np.array(seg_r.masks.xy[found_idx], dtype=np.int32)
        cv2.polylines(frame, [poly_pts], True, (0, 255, 0), 2)

        sy, sx = np.where(skeleton > 0)
        frame[sy, sx] = (0, 255, 255)


        return frame


def main():
    base_path = os.getcwd()


    # video_path = 0
    # video_path = os.path.join(base_path, "video_test_handy/IMG_1967.mp4")
    video_path = os.path.join(base_path, "GX010067.MP4")
    output_path = os.path.join(base_path, "GX010067_traj_test_2.mp4")
    detection_model_path = os.path.join(base_path, "vasco/train/det_nopretrain3/weights/best.pt")
    segmentation_model_path = os.path.join(base_path, "vasco/train/seg_nopretrain/weights/best.pt")

    # initial model
    detection_model, segmentation_model = initialize_yolo(detection_model_path, segmentation_model_path)

    # initial tracker
    tracker_name, track_history = initialize_tracker()

    # initial video and read
    cap, out, width, height = initialize_video(video_path, output_path)

    global INIT_POINT
    INIT_POINT = (width // 2, height - 30)

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

        single_frame_processing_time = single_frame_end_time - single_frame_start_time
        print(f"Single frame time: {single_frame_processing_time:.2f} s")
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
    print(f"Total {frame_count} frames。Saved to : {output_path}")
    print(f"All done for time: {all_frame_processing_time:.2f} s")

if __name__ == "__main__":
    main()
