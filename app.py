import streamlit as st
import cv2
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load model only once
@st.cache_resource
def load_model(path):
    return YOLO(path)

# Process video and generate heatmaps (cached to avoid re-running)
@st.cache_data(show_spinner="Processing video, please wait...")
def process_video(video_path, model_path, time_slice_sec):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    slice_frame_count = int(fps * time_slice_sec)
    num_slices = int(np.ceil(duration_sec / time_slice_sec))

    heatmaps = [np.zeros((frame_height, frame_width), dtype=np.float32) for _ in range(num_slices)]

    output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0

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
                    slice_index = frame_count // slice_frame_count
                    if slice_index < len(heatmaps):
                        cv2.circle(heatmaps[slice_index], (cx, cy), 10, 1, -1)

        annotated_frame = results[0].plot()
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        frame_count += 1

    cap.release()
    out.release()
    return output_path, heatmaps, time_slice_sec


# Streamlit UI starts here
st.set_page_config(page_title="YOLOv11 People Detection with Heatmap", layout="centered")
st.title("People Detection and Time-Sliced Foot Traffic Heatmaps")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # Process once and reuse
    output_path, heatmaps, interval_sec = process_video(temp_video_path, "best.pt", time_slice_sec=10)

    st.success("Processing complete.")

    st.subheader("Annotated Video")
    st.video(output_path)

    st.subheader("Select Heatmap Interval")
    intervals = [f"{i*interval_sec}s - {(i+1)*interval_sec}s" for i in range(len(heatmaps))]
    selected_slice = st.selectbox("Select interval:", intervals)
    selected_index = int(selected_slice.split('s')[0]) // interval_sec

    # Normalize and display heatmap
    heat = heatmaps[selected_index]
    if heat.max() > 0:
        heat = heat / heat.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heat, cmap="Blues", ax=ax, cbar=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Foot Traffic Heatmap: {selected_slice}")
    ax.axis('off')  # Remove axis lines
    st.pyplot(fig)

    st.download_button(
        label="Download Annotated Video",
        data=open(output_path, "rb").read(),
        file_name="annotated_output.mp4",
        mime="video/mp4"
    )
