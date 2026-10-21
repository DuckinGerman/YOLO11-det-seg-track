
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

import tempfile
import time
import os

from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import Response

from deployment.config import (
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
    DEFAULT_DET_MODEL,
    DEFAULT_SEG_MODEL,
)

from functools import lru_cache
import subprocess



det_model = YOLO(str(DETECTION_MODELS[DEFAULT_DET_MODEL]))
seg_model = YOLO(str(SEGMENTATION_MODELS[DEFAULT_SEG_MODEL]))


BASE_DIR = Path(__file__).resolve().parents[1]


app = FastAPI(title="YOLO11 Perception API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    # Detection
    det_result = det_model.predict(
        image, conf=0.4, verbose=False
    )[0]

    detections = []

    for box in det_result.boxes:
        cls_id = int(box.cls.item())

        detections.append({
            "class_id": cls_id,
            "class_name": det_result.names[cls_id],
            "confidence": float(box.conf.item()),
            "bbox": box.xyxy[0].tolist()
        })

    # Segmentation
    seg_result = seg_model.predict(
        image, conf=0.5, verbose=False
    )[0]

    segments = []

    if seg_result.masks is not None:
        for box, polygon in zip(
            seg_result.boxes,
            seg_result.masks.xy
        ):
            cls_id = int(box.cls.item())

            segments.append({
                "class_id": cls_id,
                "class_name": seg_result.names[cls_id],
                "confidence": float(box.conf.item()),
                "polygon": polygon.tolist()
            })

    return {
        "detections": detections,
        "segments": segments
    }

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...), 
    conf: float = Form(0.4),
    det_model_name: str = Form("det_nopretrain_16"),
    seg_model_name: str = Form("seg_pre16_cls7"),):

    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    

    selected_det_model = load_det_model(det_model_name)
    selected_seg_model = load_seg_model(seg_model_name)

    # Detection
    det_result = selected_det_model.predict(
        image,
        conf=conf,
        verbose=False,
    )[0]

    # Segmentation
    seg_result = selected_seg_model.predict(
        image,
        conf=conf,
        verbose=False,
    )[0]

    # Draw segmentation first
    annotated = seg_result.plot(img=image.copy())

    # Draw detection on top
    annotated = det_result.plot(img=annotated)

    success, encoded = cv2.imencode(".jpg", annotated)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode image",
        )

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
    )

@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    det_model_name: str = Form("det_nopretrain_16"),
    seg_model_name: str = Form("seg_pre16_cls7"),):

    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.mp4")
        output_path = os.path.join(temp_dir, "output.mp4")

        with open(input_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        cap = cv2.VideoCapture(input_path)

        print("Video opened:", cap.isOpened())
        print("Backend:", cap.getBackendName() if cap.isOpened() else "None")


        if not cap.isOpened():
            raise HTTPException(400, "Invalid video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        print("Writer opened:", writer.isOpened())
        if not writer.isOpened():
            cap.release()
            raise HTTPException(500, "Video writer failed")

        frame_count = 0

        if det_model_name not in DETECTION_MODELS:
            raise HTTPException(400, "Unknown detection model")

        selected_seg_model = load_seg_model(seg_model_name)

        # 每次请求使用独立 tracker，避免复用历史状态
        video_det_model = YOLO(
            str(DETECTION_MODELS[det_model_name])
        )

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Detection + BoT-SORT
                det_result = video_det_model.track(
                    frame,
                    tracker="botsort.yaml",
                    conf=conf,
                    iou=0.5,
                    persist=True,
                    verbose=False
                )[0]

                # Segmentation
                seg_result = selected_seg_model.predict(
                    frame,
                    conf=conf,
                    verbose=False
                )[0]

                # Draw detection and tracking
                annotated = det_result.plot()

                # Draw segmentation masks with transparency
                if seg_result.masks is not None:
                    overlay = annotated.copy()

                    for polygon in seg_result.masks.xy:
                        points = np.asarray(
                            polygon, dtype=np.int32
                        )
                        if len(points) >= 3:
                            cv2.fillPoly(
                                overlay,
                                [points],
                                (0, 200, 0)
                            )

                    annotated = cv2.addWeighted(
                        overlay, 0.35,
                        annotated, 0.65, 0
                    )

                writer.write(annotated)
                frame_count += 1

        finally:
            cap.release()
            writer.release()
            # Convert MP4V to browser-compatible H.264
            h264_path = output_path.replace(".mp4", "_h264.mp4")

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-movflags", "+faststart",
                    h264_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )



        elapsed = time.perf_counter() - start_time

        print(
            f"Frames: {frame_count}, "
            f"Time: {elapsed:.2f}s, "
            f"FPS: {frame_count / elapsed:.2f}"
        )

        # 临时目录会在函数结束时删除。
        # 因此不能直接从 TemporaryDirectory 返回 FileResponse。
        # 将输出文件移动到独立临时路径。
        fd, final_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        os.replace(h264_path, final_path)

    return FileResponse(
        path=final_path,
        media_type="video/mp4",
        filename="processed_video.mp4",
        background=BackgroundTask(os.unlink, final_path)
    )


@lru_cache(maxsize=4)
def load_det_model(name: str):
    if name not in DETECTION_MODELS:
        raise HTTPException(400, "Unknown detection model")

    return YOLO(str(DETECTION_MODELS[name]))


@lru_cache(maxsize=4)
def load_seg_model(name: str):
    if name not in SEGMENTATION_MODELS:
        raise HTTPException(400, "Unknown segmentation model")

    return YOLO(str(SEGMENTATION_MODELS[name]))