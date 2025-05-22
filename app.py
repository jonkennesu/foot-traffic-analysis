import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import os
import hashlib

@st.cache_resource
def load_model(path):
    return YOLO(path)

def compute_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

st.set_page_config(page_title="YOLOv11 People Detection with Heatmap", layout="centered")
st.title("People Detection and Time-Sliced Foot Traffic Heatmaps")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Initialize session state
if 'processed_hash' not in st.session_state:
    st.session_state.processed_hash = None
if 'heatmaps' not in st.session_state:
    st.session_state.heatmaps = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'sample_frame' not in st.session_state:
    st.session_state.sample_frame = None

if uploaded_video is not None:
    video_bytes = uploaded_video.read()
    current_hash = compute_file_hash(video_bytes)

    if current_hash != st.session_state.processed_hash:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            temp_video_path = tmp.name

        model = load_model("best.pt")

        cap = cv2.VideoCapture(temp_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        time_slice_sec = 10
        slice_frame_count = int(fps * time_slice_sec)
        num_slices = int(np.ceil(duration_sec / time_slice_sec))

        heatmaps = [np.zeros((frame_height, frame_width), dtype=np.float32) for _ in range(num_slices)]
        output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        stframe = st.empty()
        progress = st.progress(0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count == int(fps * 5):  # Save a sample frame at ~5s mark
                st.session_state.sample_frame = frame.copy()

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

            if frame_count % int(fps) == 0:
                stframe.image(annotated_frame, channels="RGB", use_container_width=True)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()

        st.session_state.processed_hash = current_hash
        st.session_state.heatmaps = heatmaps
        st.session_state.output_path = output_path

        st.success("Processing complete.")

    # Heatmap visualization
    st.subheader("Select Heatmap Interval")
    interval_sec = 10
    intervals = [f"{i*interval_sec}s - {(i+1)*interval_sec}s" for i in range(len(st.session_state.heatmaps))]
    selected_slice = st.selectbox("Select interval:", intervals)
    selected_index = int(selected_slice.split('s')[0]) // interval_sec

    heat = st.session_state.heatmaps[selected_index]
    global_max = max(h.max() for h in st.session_state.heatmaps)
    if global_max > 0:
        heat = heat / global_max

    if st.session_state.sample_frame is not None:
        base = cv2.cvtColor(st.session_state.sample_frame, cv2.COLOR_BGR2RGB)

        heatmap_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_OCEAN)

        if heatmap_color.shape[:2] != base.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (base.shape[1], base.shape[0]))

        overlay = cv2.addWeighted(base, 0.6, heatmap_color, 0.4, 0)

        start_time = selected_index * interval_sec
        end_time = (selected_index + 1) * interval_sec

        st.subheader(f"Foot Traffic Heatmap: {start_time}s - {end_time}s (Raw)")
        st.image(heatmap_color, channels="BGR", use_container_width=True)

        st.subheader(f"Foot Traffic Heatmap: {start_time}s - {end_time}s (Overlay)")
        st.image(overlay, channels="RGB", use_container_width=True)
    else:
        st.warning("Sample frame not available to overlay heatmap.")

    st.download_button(
        label="Download Annotated Video",
        data=open(st.session_state.output_path, "rb").read(),
        file_name="annotated_output.mp4",
        mime="video/mp4"
    )
