import streamlit as st
import cv2
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import uuid

@st.cache_resource
def load_model(path):
    return YOLO(path)

def process_frame(frame, model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, conf=0.5, classes=[0], verbose=False)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    centers = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            centers.append((cx, cy))
    return annotated_frame, centers

st.set_page_config(page_title="YOLOv11 People Detection with Heatmap", layout="centered")
st.title("People Detection and Time-Sliced Foot Traffic Heatmaps")

uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg","jpeg","png","mp4", "mov", "avi"])

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'last_uploaded_name' not in st.session_state:
    st.session_state.last_uploaded_name = None
if 'heatmaps' not in st.session_state:
    st.session_state.heatmaps = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'first_annotated_frame' not in st.session_state:
    st.session_state.first_annotated_frame = None

model = load_model("best.pt")

def reset_states_for_new_file(name):
    st.session_state.processed = False
    st.session_state.heatmaps = None
    st.session_state.output_path = None
    st.session_state.first_annotated_frame = None
    st.session_state.last_uploaded_name = name

if uploaded_file is not None:
    if st.session_state.last_uploaded_name != uploaded_file.name:
        reset_states_for_new_file(uploaded_file.name)

    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext in ['jpg','jpeg','png']:
        file_bytes = uploaded_file.read()
        np_img = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        annotated_img, centers = process_frame(img, model)

        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), caption="Annotated Image", use_column_width=True)

        st.success("Image processing complete — no heatmaps for images.")

    elif file_ext in ['mp4', 'mov', 'avi']:
        if not st.session_state.processed:

            unique_filename = f"temp_{uuid.uuid4()}.mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=unique_filename) as tmp:
                tmp.write(uploaded_file.read())
                temp_video_path = tmp.name

            cap = cv2.VideoCapture(temp_video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0

            time_slice_sec = 10
            slice_frame_count = int(fps * time_slice_sec) if fps > 0 else 1
            num_slices = int(np.ceil(duration_sec / time_slice_sec)) if fps > 0 else 1

            heatmaps = [np.zeros((frame_height, frame_width), dtype=np.float32) for _ in range(num_slices)]

            output_path = os.path.join(tempfile.gettempdir(), f"output_annotated_{uuid.uuid4()}.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            stframe = st.empty()
            progress = st.progress(0)
            frame_count = 0
            first_frame_set = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, centers = process_frame(frame, model)

                slice_index = frame_count // slice_frame_count if slice_frame_count > 0 else 0
                if slice_index < len(heatmaps):
                    for (cx, cy) in centers:
                        if 0 <= cy < frame_height and 0 <= cx < frame_width:
                            cv2.circle(heatmaps[slice_index], (cx, cy), 10, 1, -1)

                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

                if not first_frame_set:
                    st.session_state.first_annotated_frame = annotated_frame
                    first_frame_set = True

                if frame_count % int(fps) == 0:
                    stframe.image(annotated_frame, channels="RGB", use_container_width=True)

                frame_count += 1
                progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()

            st.session_state.heatmaps = heatmaps
            st.session_state.output_path = output_path
            st.session_state.processed = True

        st.success("Video processing complete.")

        if st.session_state.first_annotated_frame is not None:
            st.subheader("Video Preview (First Annotated Frame)")
            st.image(st.session_state.first_annotated_frame, channels="RGB", use_container_width=True)

        st.subheader("Annotated Video")

        with open(st.session_state.output_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.subheader("Select Heatmap Interval")
        interval_sec = 10
        intervals = [f"{i*interval_sec}s - {(i+1)*interval_sec}s" for i in range(len(st.session_state.heatmaps))]
        selected_slice = st.selectbox("Select interval:", intervals)
        selected_index = intervals.index(selected_slice)

        heat = st.session_state.heatmaps[selected_index]
        if heat.max() > 0:
            heat = heat / heat.max()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heat, cmap="Blues", ax=ax, cbar=True, xticklabels=False, yticklabels=False)
        ax.tick_params(left=False, bottom=False)
        ax.set_title(f"Foot Traffic Heatmap: {selected_slice}")
        st.pyplot(fig)

        # Download button for annotated video
        with open(st.session_state.output_path, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label="Download Annotated Video",
            data=video_bytes,
            file_name="annotated_output.mp4",
            mime="video/mp4"
        )
    else:
        st.error("Unsupported file type. Please upload an image or video file.")

else:
    st.info("Please upload an image or video file to start detection.")
