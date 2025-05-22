import streamlit as st
import cv2
import tempfile
import numpy as np
import os
import hashlib
from ultralytics import YOLO
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd

@st.cache_resource
def load_model(path):
    """Load YOLO model with caching"""
    return YOLO(path)

def compute_file_hash(file_bytes):
    """Compute MD5 hash of file bytes"""
    return hashlib.md5(file_bytes).hexdigest()

def detect_overcrowding(heatmap, threshold_multiplier=1.5):
    """
    Detect crowd locations based on heatmap analysis
    Returns crowded locations if any crowd is detected
    """
    # Check if there's any crowd at all
    if np.max(heatmap) == 0:
        return []
    
    # Find any areas with activity (simplified approach)
    # Use a lower threshold to detect any significant activity
    activity_threshold = np.max(heatmap) * 0.3  # 30% of max activity
    
    # Find crowded regions
    activity_mask = heatmap > activity_threshold
    
    # Find crowded areas (regions)
    crowded_locations = []
    if np.any(activity_mask):
        # Find contours of active regions
        activity_uint8 = (activity_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(activity_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = heatmap.shape
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 50:  # Filter very small regions
                # Get bounding box
                x, y, box_w, box_h = cv2.boundingRect(contour)
                center_x, center_y = x + box_w//2, y + box_h//2
                
                # Determine location description
                location = []
                if center_x < w//3:
                    location.append("Left")
                elif center_x > 2*w//3:
                    location.append("Right")
                else:
                    location.append("Center")
                    
                if center_y < h//3:
                    location.append("Top")
                elif center_y > 2*h//3:
                    location.append("Bottom")
                else:
                    location.append("Middle")
                
                crowded_locations.append(" ".join(location))
    
    return crowded_locations

def resize_for_display(image, target_size=640):
    """Resize image for display to standard size while maintaining aspect ratio"""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        # Always resize to target_size for consistency
        if h > w:
            new_h, new_w = target_size, int(w * target_size / h)
        else:
            new_h, new_w = int(h * target_size / w), target_size
        return cv2.resize(image, (new_w, new_h))
    return image

def create_clean_heatmap(heatmap, title="", cmap="Blues", target_size=640):
    """Create a clean heatmap without ticks, numbers, or colorbar at standard size"""
    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
    else:
        heatmap_norm = heatmap
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(heatmap_norm, cmap=cmap, cbar=False, 
                xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    heatmap_img = Image.open(buf).convert("RGB")
    plt.close(fig)
    
    # Resize to standard size
    heatmap_array = np.array(heatmap_img)
    return resize_for_display(heatmap_array, target_size)
def create_heatmap_overlay(base_image, heatmap, alpha=0.4):
    """Create heatmap overlay on base image at standard size"""
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
    else:
        heatmap_norm = heatmap
    
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(heatmap_norm, cmap="hot", cbar=False, 
                xticklabels=False, yticklabels=False, ax=ax)
    ax.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    heatmap_img = Image.open(buf).convert("RGB")
    heatmap_np = np.array(heatmap_img)
    plt.close(fig)
    
    # Resize both images to standard display size
    base_resized = resize_for_display(base_image)
    heatmap_resized = resize_for_display(heatmap_np)
    
    # Ensure same dimensions
    if heatmap_resized.shape[:2] != base_resized.shape[:2]:
        heatmap_resized = cv2.resize(heatmap_resized, (base_resized.shape[1], base_resized.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(base_resized, 1-alpha, heatmap_resized, alpha, 0)
    return overlay

def process_image(image_bytes, model, conf_threshold, nms_threshold):
    """Process single image for people detection"""
    # Convert bytes to image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict(
        source=image_rgb, 
        conf=conf_threshold, 
        iou=nms_threshold,
        classes=[0],  # person class
        verbose=False
    )
    
    boxes = results[0].boxes
    people_count = len(boxes) if boxes is not None else 0
    
    # Create heatmap
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    detections = []
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            if 0 <= cy < image.shape[0] and 0 <= cx < image.shape[1]:
                cv2.circle(heatmap, (cx, cy), 15, 1, -1)
                detections.append((cx, cy))
    
    # Get annotated image
    annotated_image = results[0].plot()
    
    # Add people count to annotated image
    cv2.putText(annotated_image, f'People Count: {people_count}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image_rgb, annotated_image, heatmap, people_count, detections

def process_video(video_path, model, conf_threshold, nms_threshold, progress_bar, status_text):
    """Process video for people detection and create time-sliced heatmaps"""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    
    # Time slice configuration
    time_slice_sec = 10
    slice_frame_count = int(fps * time_slice_sec)
    num_slices = int(np.ceil(duration_sec / time_slice_sec))
    
    # Initialize storage
    heatmaps = [np.zeros((frame_height, frame_width), dtype=np.float32) for _ in range(num_slices)]
    slice_frames = {}
    slice_people_counts = {}
    all_annotated_frames = []
    
    # Output video setup
    output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        slice_index = frame_count // slice_frame_count
        
        # Store representative frame for each slice (with annotations)
        if slice_index < num_slices and slice_index not in slice_frames:
            # Run inference for this frame first
            temp_results = model.predict(
                source=img_rgb, 
                conf=conf_threshold, 
                iou=nms_threshold,
                classes=[0], 
                verbose=False
            )
            temp_annotated = temp_results[0].plot()
            temp_people_count = len(temp_results[0].boxes) if temp_results[0].boxes is not None else 0
            cv2.putText(temp_annotated, f'People Count: {temp_people_count}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            slice_frames[slice_index] = temp_annotated
            # Also store original frame for overlay
            slice_frames[f"{slice_index}_original"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(
            source=img_rgb, 
            conf=conf_threshold, 
            iou=nms_threshold,
            classes=[0], 
            verbose=False
        )
        
        boxes = results[0].boxes
        people_count = len(boxes) if boxes is not None else 0
        
        # Update people count for this slice
        if slice_index not in slice_people_counts:
            slice_people_counts[slice_index] = []
        slice_people_counts[slice_index].append(people_count)
        
        # Update heatmap
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if 0 <= cy < frame_height and 0 <= cx < frame_width:
                    if slice_index < len(heatmaps):
                        cv2.circle(heatmaps[slice_index], (cx, cy), 10, 1, -1)
        
        # Create annotated frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f'People Count: {people_count}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write to output video
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        frame_count += 1
        
        # Update progress
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    # Calculate average people count per slice
    avg_people_counts = {}
    for slice_idx, counts in slice_people_counts.items():
        avg_people_counts[slice_idx] = np.mean(counts) if counts else 0
    
    return heatmaps, slice_frames, output_path, avg_people_counts, num_slices

# Streamlit App Configuration
st.set_page_config(page_title="Retail Foot Traffic Analyzer", layout="wide")
st.title("🏪 Retail Foot Traffic Analysis System")
st.markdown("Upload an image or video to analyze foot traffic patterns and detect overcrowding in your retail store.")

# Sidebar for controls
st.sidebar.header("⚙️ Detection Settings")

# Model loading
@st.cache_resource
def get_model():
    try:
        return load_model("best.pt")
    except:
        st.error("Model file 'best.pt' not found. Please ensure the YOLO model is available.")
        return None

model = get_model()

if model is None:
    st.stop()

# Threshold controls
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Minimum confidence score for detections"
)

nms_threshold = st.sidebar.slider(
    "NMS (Non-Maximum Suppression) Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.4, 
    step=0.05,
    help="Controls overlap between detections. Lower values = less overlap allowed"
)

st.sidebar.markdown("**NMS Threshold Explanation:**")
st.sidebar.info("NMS removes duplicate detections of the same person. Lower values (0.1-0.3) are stricter and remove more overlapping boxes, while higher values (0.5-0.9) allow more overlap. For crowded scenes, use lower values.")

overcrowding_sensitivity = st.sidebar.slider(
    "Overcrowding Sensitivity",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Lower values = more sensitive to overcrowding"
)

# File upload
uploaded_file = st.file_uploader(
    "Upload Image or Video", 
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    help="Supported formats: JPG, PNG for images; MP4, MOV, AVI for videos"
)

# Initialize session state
if 'processed_hash' not in st.session_state:
    st.session_state.processed_hash = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'results' not in st.session_state:
    st.session_state.results = None

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    current_hash = compute_file_hash(file_bytes)
    file_type = uploaded_file.type.split('/')[0]  # 'image' or 'video'
    
    # Check if we need to reprocess
    need_processing = (
        current_hash != st.session_state.processed_hash or 
        st.session_state.results is None
    )
    
    if need_processing:
        st.session_state.processed_hash = current_hash
        st.session_state.file_type = file_type
        
        if file_type == 'image':
            st.subheader("📊 Image Analysis Results")
            
            with st.spinner("Processing image..."):
                original_img, annotated_img, heatmap, people_count, detections = process_image(
                    file_bytes, model, conf_threshold, nms_threshold
                )
                
                st.session_state.results = {
                    'original_img': original_img,
                    'annotated_img': annotated_img,
                    'heatmap': heatmap,
                    'people_count': people_count,
                    'detections': detections
                }
            
            st.success(f"✅ Processing complete! Detected {people_count} people.")
            
        elif file_type == 'video':
            st.subheader("🎥 Video Analysis Results")
            
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file_bytes)
                temp_video_path = tmp.name
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing video..."):
                heatmaps, slice_frames, output_path, avg_people_counts, num_slices = process_video(
                    temp_video_path, model, conf_threshold, nms_threshold, 
                    progress_bar, status_text
                )
                
                st.session_state.results = {
                    'heatmaps': heatmaps,
                    'slice_frames': slice_frames,
                    'output_path': output_path,
                    'avg_people_counts': avg_people_counts,
                    'num_slices': num_slices
                }
            
            # Clean up temp file
            os.unlink(temp_video_path)
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Video processing complete!")
    
    # Display results
    if st.session_state.results is not None:
        if st.session_state.file_type == 'image':
            results = st.session_state.results
            
            # Display original and annotated images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                # Resize for display
                display_original = resize_for_display(results['original_img'])
                st.image(display_original, use_container_width=False)
            
            with col2:
                st.subheader(f"Detected People: {results['people_count']}")
                # Resize for display
                display_annotated = resize_for_display(results['annotated_img'])
                st.image(display_annotated, use_container_width=False)
            
            # Heatmap analysis
            st.subheader("🔥 Foot Traffic Heatmap Analysis")
            
            # Detect crowd locations
            crowded_locations = detect_overcrowding(results['heatmap'], overcrowding_sensitivity)
            
            # Show crowded locations if any
            if crowded_locations:
                locations_text = ", ".join(crowded_locations)
                st.markdown(f"**Crowded Areas:** {locations_text}")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Raw Heatmap")
                if results['heatmap'].max() > 0:
                    clean_heatmap = create_clean_heatmap(results['heatmap'], "Foot Traffic Density", "Blues")
                    st.image(clean_heatmap, use_container_width=False)
                else:
                    st.info("No foot traffic detected")
            
            with col4:
                st.subheader("Overlay Heatmap")
                if results['heatmap'].max() > 0:
                    overlay_img = create_heatmap_overlay(results['original_img'], results['heatmap'])
                    st.image(overlay_img, use_container_width=False)
                else:
                    display_original_2 = resize_for_display(results['original_img'])
                    st.image(display_original_2, use_container_width=False)
            
            # Download annotated image
            annotated_pil = Image.fromarray(results['annotated_img'])
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            
            st.download_button(
                label="📥 Download Annotated Image",
                data=buf.getvalue(),
                file_name="annotated_image.png",
                mime="image/png"
            )
            
        elif st.session_state.file_type == 'video':
            results = st.session_state.results
            
            # Frame selection
            st.subheader("🎯 Frame Analysis")
            
            interval_options = []
            for i in range(results['num_slices']):
                start_time = i * 10
                end_time = min((i + 1) * 10, int(len(results['heatmaps']) * 10))
                avg_count = results['avg_people_counts'].get(i, 0)
                interval_options.append(f"{start_time}s-{end_time}s (Avg: {avg_count:.1f} people)")
            
            selected_interval = st.selectbox("Select Time Interval:", interval_options)
            selected_index = interval_options.index(selected_interval)
            
            if selected_index in results['slice_frames']:
                frame = results['slice_frames'][selected_index]
                heatmap = results['heatmaps'][selected_index]
                
                # Display frame with bounding boxes
                st.subheader(f"Frame from {selected_interval}")
                display_frame = resize_for_display(frame)
                st.image(display_frame, use_container_width=False)
                
                # Crowd location analysis
                crowded_locations = detect_overcrowding(heatmap, overcrowding_sensitivity)
                
                # Show crowded locations if any
                if crowded_locations:
                    locations_text = ", ".join(crowded_locations)
                    st.markdown(f"**Crowded Areas:** {locations_text}")
                
                # Heatmap visualization
                st.subheader("🔥 10-Second Interval Heatmap")
                
                col5, col6 = st.columns(2)
                
                with col5:
                    st.subheader("Raw Heatmap")
                    if heatmap.max() > 0:
                        clean_heatmap = create_clean_heatmap(heatmap, f"Traffic Density: {selected_interval}", "Blues")
                        st.image(clean_heatmap, use_container_width=False)
                    else:
                        st.info("No traffic detected in this interval")
                
                with col6:
                    st.subheader("Overlay Heatmap")
                    if heatmap.max() > 0:
                        # Use original frame (without annotations) for overlay
                        original_key = f"{selected_index}_original"
                        if original_key in results['slice_frames']:
                            original_frame = results['slice_frames'][original_key]
                            overlay_img = create_heatmap_overlay(original_frame, heatmap)
                            st.image(overlay_img, use_container_width=False)
                        else:
                            # Fallback to annotated frame if original not available
                            overlay_img = create_heatmap_overlay(frame, heatmap)
                            st.image(overlay_img, use_container_width=False)
                    else:
                        display_frame_2 = resize_for_display(frame)
                        st.image(display_frame_2, use_container_width=False)
            
            # Overall statistics
            st.subheader("📈 Video Statistics")
            
            col7, col8, col9 = st.columns(3)
            
            with col7:
                total_people = sum(results['avg_people_counts'].values())
                st.metric("Total Average People", f"{total_people:.1f}")
            
            with col8:
                max_people = max(results['avg_people_counts'].values()) if results['avg_people_counts'] else 0
                st.metric("Peak Occupancy", f"{max_people:.1f}")
            
            with col9:
                avg_people = np.mean(list(results['avg_people_counts'].values())) if results['avg_people_counts'] else 0
                st.metric("Average Occupancy", f"{avg_people:.1f}")
            
            # Time series plot
            if results['avg_people_counts']:
                fig3, ax3 = plt.subplots(figsize=(12, 4))
                times = [i * 10 for i in results['avg_people_counts'].keys()]
                counts = list(results['avg_people_counts'].values())
                
                ax3.plot(times, counts, marker='o', linewidth=2, markersize=6)
                ax3.set_xlabel("Time (seconds)")
                ax3.set_ylabel("Average People Count")
                ax3.set_title("Foot Traffic Over Time")
                ax3.grid(True, alpha=0.3)
                
                st.pyplot(fig3)
            
            # Download annotated video
            if os.path.exists(results['output_path']):
                with open(results['output_path'], "rb") as video_file:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=video_file.read(),
                        file_name="annotated_video.mp4",
                        mime="video/mp4"
                    )

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    "This app uses YOLOv11 to detect people in retail environments and create foot traffic heatmaps. "
    "Upload images or videos to analyze crowd patterns and identify potential overcrowding areas."
)

st.sidebar.subheader("🎯 Key Features")
st.sidebar.markdown("""
- **People Detection**: Real-time person detection with confidence scores
- **Heatmap Generation**: Visual representation of foot traffic patterns  
- **Overcrowding Detection**: Automatic identification of crowded areas
- **Time Analysis**: For videos, analyze traffic patterns over time
- **Downloadable Results**: Export annotated images and videos
""")
