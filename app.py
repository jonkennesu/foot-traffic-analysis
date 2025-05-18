import streamlit as st
import cv2
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Streamlit UI
st.set_page_config(page_title="YOLOv11n People Detection with Heatmap", layout="centered")
st.title("People Detection and Foot Traffic Heatmap")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # Load model
    model = YOLO("best.pt")  

    cap = cv2.VideoCapture(temp_video_path)
    frame_count = 0
    footfall_points = []

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            results = model.predict(source=frame, conf=0.5, classes=[0], verbose=False)
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    footfall_points.append([cx, cy])

            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        frame_count += 1

    cap.release()

    # Generate heatmap
    st.subheader("Foot Traffic Heatmap")
    if footfall_points:
        heatmap_frame = np.zeros((frame.shape[0], frame.shape[1]))
        for pt in footfall_points:
            x, y = pt
            if 0 <= y < heatmap_frame.shape[0] and 0 <= x < heatmap_frame.shape[1]:
                heatmap_frame[y, x] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_frame, cmap="hot", ax=ax, cbar=True)
        st.pyplot(fig)
    else:
        st.warning("No people detected. Heatmap could not be generated.")
