
import streamlit as st
import requests

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Visual Perception Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("Visual Perception Dashboard")
st.caption("YOLO11 Detection · Segmentation · BoT-SORT Tracking")

mode = st.radio(
    "Input Type",
    ["Image", "Video"],
    horizontal=True,)

confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,)
det_model_name = st.selectbox(
    "Detection Model",
    ["det_nopretrain_16", "det_1"],
)

seg_model_name = st.selectbox(
    "Segmentation Model",
    ["seg_pre16_cls7", "seg_1"],
)


if mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi"],
    )

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")

        if mode == "Image":
            st.image(uploaded_file, use_container_width=True)
        else:
            st.video(uploaded_file)

    if st.button("Run Inference", type="primary"):
        endpoint = (
            "/predict/image"
            if mode == "Image"
            else "/predict/video"
        )

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }

        try:
            with st.spinner("Running inference..."):
                response = requests.post(
                    API_URL + endpoint,
                    files=files,
                    data={"conf": confidence,
                          "det_model_name": det_model_name,
                          "seg_model_name": seg_model_name,},
                    timeout=1800,
                )

                response.raise_for_status()

            with col2:
                st.subheader("Result")

                if mode == "Image":
                    st.image(
                        response.content,
                        use_container_width=True,
                        )

                    st.download_button(
                            "Download Processed Image",
                            data=response.content,
                            file_name="processed_image.jpg",
                            mime="image/jpeg",
                        )
                else:
                    st.video(response.content)

                    st.download_button(
                        "Download Processed Video",
                        data=response.content,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )

        except requests.RequestException as e:
            st.error(f"API request failed: {e}")
