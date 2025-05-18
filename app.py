import streamlit as st
import cv2
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

@st.cache_resource
def load_model(path):
    return YOLO(path)

def process_video(video_path, model, fps, frame_height, frame_width, total_frames, time_slice_sec):
    cap = cv2.VideoCapture(video_path)

    slice_frame_count = int(fps * time_slice_sec)
    duration_sec = total_frames / fps
    num_slices = int(np.ceil(duration_sec / time_slice_sec))

    heatmaps = [np.zeros((frame_height, frame_width), dtype=np.float32) for _ in range(num_slices)]

    output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    progress_bar = st.progress(0)
    stframe = st.empty()

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

        if frame_count % int(fps) == 0:
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()

    return heatmaps, output_path


st.set_page_config(page_title="YOLOv11 People Detection with Heatmap", layout="centered")
st.title("People Detection and Time-Sliced Foot Traffic Heatmaps")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    model = load_model("best.pt")

    cap = cv2.VideoCapture(temp_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    time_slice_sec = 10  # time slice length in seconds

    if not st.session_state.processed:
        st.info("Running model inference on the uploaded video, please wait...")
        heatmaps, output_path = process_video(temp_video_path, model, fps, frame_height, frame_width, total_frames, time_slice_sec)
        st.session_state.heatmaps = heatmaps
        st.session_state.output_path = output_path
        st.session_state.processed = True
    else:
        heatmaps = st.session_state.heatmaps
        output_path = st.session_state.output_path

    st.success("Processing complete.")

    st.subheader("Annotated Video")
    st.video(output_path)

    st.subheader("Select Heatmap Interval")
    interval_sec = time_slice_sec
    intervals = [f"{i*interval_sec}s - {(i+1)*interval_sec}s" for i in range(len(heatmaps))]
    selected_slice = st.selectbox("Select interval:", intervals)
    selected_index = int(selected_slice.split('s')[0]) // interval_sec

    heat = heatmaps[selected_index]
    if heat.max() > 0:
        heat = heat / heat.max()  

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heat,
        cmap="hot",
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        linecolor=None,
        linewidth=0,
        ax=ax
    )
    ax.set_title(f"Foot Traffic Heatmap: {selected_slice}")
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)

    st.download_button(
        label="Download Annotated Video",
        data=open(output_path, "rb").read(),
        file_name="annotated_output.mp4",
        mime="video/mp4"
    )
