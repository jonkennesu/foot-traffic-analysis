import streamlit as st
import cv2
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import os

@st.cache_resource
def load_model(path):
    return YOLO(path)

st.set_page_config(page_title="YOLOv8 People Detection with Heatmap", layout="centered")
st.title("People Detection and Foot Traffic Heatmap")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    model = load_model("best.pt")

    cap = cv2.VideoCapture(temp_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    heatmap_frame = np.zeros((frame_height, frame_width))
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img_rgb, conf=0.5, classes=[0], verbose=False)
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if 0 <= cy < frame_height and 0 <= cx < frame_width:
                    heatmap_frame[cy, cx] += 1

        annotated_frame = results[0].plot()
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        frame_count += 1
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()

    st.success("Processing complete.")

    st.subheader("Annotated Video")
    st.video(output_path)

    st.subheader("Foot Traffic Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_frame, cmap="hot", ax=ax, cbar=True)
    st.pyplot(fig)
