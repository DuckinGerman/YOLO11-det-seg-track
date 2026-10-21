from collections import defaultdict
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
import heapq
from traj_dji_path_utils import distance_transform_path_local

def initialize_yolo(detection_model_path, segmentation_model_path):
    detection_model = YOLO(detection_model_path)
    segmentation_model = YOLO(segmentation_model_path)
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



def polygon_to_bool_mask(foot_mask_points, height, width):
    mask_2d = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_2d, [foot_mask_points], 255)
    return (mask_2d > 127)

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
    py = frame_height - frame_height // 4

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
        # 2. trajectory generation
        # 起点(底部中心), 终点(顶部中心) 示例
        if foot_mask_points is not None:
            # get mask points
            mask_points = np.array(foot_mask_points)
            band = 5
            # get min and max in y direction
            min_y, max_y = np.min(mask_points[:, 1]), np.max(mask_points[:, 1])
            min_x, max_x = np.min(mask_points[:, 0]), np.max(mask_points[:, 0])
            # average weight
            z = (max_y - min_y) // 20
            min_y = min_y + 2 * z
            max_y = max_y - 2 * z
            previous_x = (max_x + min_x) // 2
            # use sample on the band to elimination noise point
            # top
            top_band_y = [y for y in foot_mask_points[:, 1] if y <= min_y + band]
            top_band_x = [foot_mask_points[i, 0] for i in range(len(foot_mask_points))
                          if foot_mask_points[i, 1] <= min_y + band]
            x_top = np.mean(top_band_x) if len(top_band_x) > 0 else np.mean(foot_mask_points[:, 0])
            # bottom
            bottom_band_x = [foot_mask_points[i, 0] for i in range(len(foot_mask_points))
                             if foot_mask_points[i, 1] >= max_y - band]
            x_bottom = np.mean(bottom_band_x) if len(bottom_band_x) > 0 else np.mean(foot_mask_points[:, 0])
            # start and goal
            sx, sy = int(x_bottom), int(max_y)
            gx, gy = int(x_top), int(min_y)

            # 多边形转 uint8 mask
            mask_2d = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(mask_2d, [foot_mask_points], 255)
            roi = (min_x, min_y, max_x, max_y)
            path, dist_map = distance_transform_path_local(mask_2d, (sx, sy), (gx, gy), roi, margin =10)
            # 用蓝线画出最优路径
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], (0, 255, 255), 3)

    return frame


def main():
    base_path = os.getcwd()

    # video_path = 0
    # video_path = os.path.join(base_path, "video_test_handy/IMG_1967.mp4")
    video_path = os.path.join(base_path, "GX010067.MP4")
    output_path = os.path.join(base_path, "GX010067_tra_dij.mp4")
    detection_model_path = os.path.join(base_path, "runs/train/det_1/weights/best.pt")
    segmentation_model_path = os.path.join(base_path, "runs/train/seg_2/weights/best.pt")

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
