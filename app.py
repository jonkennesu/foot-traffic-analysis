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

# Configure page with minimal theme
st.set_page_config(
    page_title="Retail Analytics", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Modern minimal CSS
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global styling */
    .stApp {
        background-color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .app-header {
        background: #ffffff;
        padding: 2.5rem 0;
        margin-bottom: 3rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
        text-align: center;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: 0.5rem;
        text-align: center;
        font-weight: 400;
    }
    
    /* Card components */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #f3f4f6;
    }
    
    /* Metrics styling */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f3f4f6;
    }
    
    /* Image containers */
    .image-card {
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .image-title {
        background: #f9fafb;
        padding: 1rem 1.5rem;
        font-weight: 500;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
        margin: 0;
    }
    
    .image-content {
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    .sidebar-card {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-title {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stDownloadButton > button {
        background: #111827;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background: #374151;
        transform: translateY(-1px);
    }
    
    /* Upload area */
    .stFileUploader > div > div {
        background: #ffffff;
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #9ca3af;
        background: #f9fafb;
    }
    
    /* Alert boxes */
    .alert {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: #f0fdf4;
        border-left-color: #22c55e;
        color: #166534;
    }
    
    .alert-info {
        background: #eff6ff;
        border-left-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-warning {
        background: #fffbeb;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: #111827;
        border-radius: 4px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: #f3f4f6;
    }
    
    .stSlider > div > div > div > div {
        background: #111827;
    }
    
    /* Remove default streamlit spacing */
    .element-container {
        margin-bottom: 0;
    }
    
    /* Custom spacing utilities */
    .mt-4 { margin-top: 2rem; }
    .mb-4 { margin-bottom: 2rem; }
    .text-center { text-align: center; }
</style>
""", unsafe_allow_html=True)

# All your existing functions remain the same
@st.cache_resource
def load_model(path):
    """Load YOLO model with caching"""
    return YOLO(path)

def compute_file_hash(file_bytes):
    """Compute MD5 hash of file bytes"""
    return hashlib.md5(file_bytes).hexdigest()

def detect_overcrowding(heatmap, threshold_multiplier=1.5):
    if np.max(heatmap) == 0:
        return []
    
    # Apply sensitivity multiplier properly
    activity_threshold = (np.max(heatmap) * 0.3) * threshold_multiplier
    
    # Find crowded regions
    activity_mask = heatmap > activity_threshold
    
    if not np.any(activity_mask):
        return []
    
    # Use morphological operations to consolidate nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    consolidated_mask = cv2.morphologyEx(activity_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Find contours with minimum area threshold
    contours, _ = cv2.findContours(consolidated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crowded_locations = []
    h, w = heatmap.shape
    min_area = (h * w) * 0.02  # At least 2% of image area
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Get center of mass instead of bounding box center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Determine location
                location = []
                if cx < w//3:
                    location.append("Left")
                elif cx > 2*w//3:
                    location.append("Right")
                else:
                    location.append("Center")
                    
                if cy < h//3:
                    location.append("Top")
                elif cy > 2*h//3:
                    location.append("Bottom")
                else:
                    location.append("Middle")
                
                crowded_locations.append(" ".join(location))
    
    return list(set(crowded_locations))  # Remove duplicates

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
    
    # Convert to numpy array and resize to standard size
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
    
    # Output video setup
    output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB first (FIXED: moved this before first use)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
            slice_frames[f"{slice_index}_original"] = img_rgb
        
        # Run inference
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

# Modern header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Retail Analytics</h1>
    <p class="app-subtitle">AI-powered customer flow analysis for retail optimization</p>
</div>
""", unsafe_allow_html=True)

# Clean sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Detection Settings</div>', unsafe_allow_html=True)
    
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

    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score for detections"
    )

    nms_threshold = st.slider(
        "NMS Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.4, 
        step=0.05,
        help="Controls overlap between detections"
    )

    overcrowding_sensitivity = st.slider(
        "Overcrowding Sensitivity",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Lower values detect crowding more easily"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Features</div>', unsafe_allow_html=True)
    st.markdown("""
    **Real-time Detection** with confidence scoring
    
    **Visual Heatmaps** showing traffic patterns
    
    **Crowding Analysis** with location identification
    
    **Time-based Insights** for video content
    
    **Export Options** for reporting
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# File upload section
st.markdown('<h2 class="section-title">Upload Content</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image or video file", 
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    help="Supported: JPG, PNG, MP4, MOV, AVI"
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
    file_type = uploaded_file.type.split('/')[0]
    
    # Check if we need to reprocess
    need_processing = (
        current_hash != st.session_state.processed_hash or 
        st.session_state.results is None
    )
    
    if need_processing:
        st.session_state.processed_hash = current_hash
        st.session_state.file_type = file_type
        
        if file_type == 'image':
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
            
            st.markdown(f'<div class="alert alert-success">✓ Processing complete! Detected {people_count} people</div>', unsafe_allow_html=True)
            
        elif file_type == 'video':
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
            st.markdown('<div class="alert alert-success">✓ Video processing complete</div>', unsafe_allow_html=True)
    
    # Display results
    if st.session_state.results is not None:
        if st.session_state.file_type == 'image':
            results = st.session_state.results
            
            # Metrics
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{results['people_count']}</div>
                    <div class="metric-label">People Detected</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                crowded_count = len(detect_overcrowding(results['heatmap'], overcrowding_sensitivity))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{crowded_count}</div>
                    <div class="metric-label">Crowded Areas</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                density = "High" if results['people_count'] > 10 else "Medium" if results['people_count'] > 5 else "Low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{density}</div>
                    <div class="metric-label">Traffic Density</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image comparison
            st.markdown('<h2 class="section-title">Detection Results</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="image-card">
                    <h4 class="image-title">Overlay Analysis</h4>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                if results['heatmap'].max() > 0:
                    overlay_img = create_heatmap_overlay(results['original_img'], results['heatmap'])
                    st.image(overlay_img, use_container_width=True)
                else:
                    display_original_2 = resize_for_display(results['original_img'])
                    st.image(display_original_2, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Download section
            st.markdown('<h2 class="section-title">Export Results</h2>', unsafe_allow_html=True)
            
            col_center = st.columns([1, 2, 1])[1]
            with col_center:
                annotated_pil = Image.fromarray(results['annotated_img'])
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                
                st.download_button(
                    label="Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="retail_analysis_annotated.png",
                    mime="image/png"
                )
            
        elif st.session_state.file_type == 'video':
            results = st.session_state.results
            
            # Video metrics
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            
            with col_v1:
                total_people = sum(results['avg_people_counts'].values())
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_people:.1f}</div>
                    <div class="metric-label">Total Avg People</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_v2:
                max_people = max(results['avg_people_counts'].values()) if results['avg_people_counts'] else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{max_people:.1f}</div>
                    <div class="metric-label">Peak Occupancy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_v3:
                avg_people = np.mean(list(results['avg_people_counts'].values())) if results['avg_people_counts'] else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_people:.1f}</div>
                    <div class="metric-label">Average Occupancy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_v4:
                duration = results['num_slices'] * 10
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{duration}s</div>
                    <div class="metric-label">Duration</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Time interval analysis
            st.markdown('<h2 class="section-title">Time Interval Analysis</h2>', unsafe_allow_html=True)
            
            interval_options = []
            for i in range(results['num_slices']):
                start_time = i * 10
                end_time = (i + 1) * 10
                avg_count = results['avg_people_counts'].get(i, 0)
                interval_options.append(f"{start_time}s-{end_time}s (Avg: {avg_count:.1f} people)")
            
            selected_interval = st.selectbox("Select Time Interval:", interval_options, key="interval_select")
            selected_index = interval_options.index(selected_interval)
            
            if selected_index in results['slice_frames']:
                frame = results['slice_frames'][selected_index]
                heatmap = results['heatmaps'][selected_index]
                
                # Display frame
                st.markdown(f"""
                <div class="image-card">
                    <h4 class="image-title">Frame Analysis: {selected_interval}</h4>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                display_frame = resize_for_display(frame)
                st.image(display_frame, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Crowd analysis for this interval
                crowded_locations = detect_overcrowding(heatmap, overcrowding_sensitivity)
                
                if crowded_locations:
                    st.markdown(f'<div class="alert alert-warning">⚠ Crowded areas in this interval: {", ".join(crowded_locations)}</div>', unsafe_allow_html=True)
                
                # Interval heatmaps
                st.markdown('<h3 class="section-title">Interval Heatmap Analysis</h3>', unsafe_allow_html=True)
                
                col5, col6 = st.columns(2)
                
                with col5:
                    st.markdown("""
                    <div class="image-card">
                        <h4 class="image-title">Traffic Heatmap</h4>
                        <div class="image-content">
                    """, unsafe_allow_html=True)
                    if heatmap.max() > 0:
                        clean_heatmap = create_clean_heatmap(heatmap, "", "Blues")
                        st.image(clean_heatmap, use_container_width=True)
                    else:
                        st.info("No traffic detected in this interval")
                    st.markdown('</div></div>', unsafe_allow_html=True)
                
                with col6:
                    st.markdown("""
                    <div class="image-card">
                        <h4 class="image-title">Overlay Analysis</h4>
                        <div class="image-content">
                    """, unsafe_allow_html=True)
                    if heatmap.max() > 0:
                        # Use original frame for overlay
                        original_key = f"{selected_index}_original"
                        if original_key in results['slice_frames']:
                            original_frame = results['slice_frames'][original_key]
                            overlay_img = create_heatmap_overlay(original_frame, heatmap)
                            st.image(overlay_img, use_container_width=True)
                        else:
                            overlay_img = create_heatmap_overlay(frame, heatmap)
                            st.image(overlay_img, use_container_width=True)
                    else:
                        display_frame_2 = resize_for_display(frame)
                        st.image(display_frame_2, use_container_width=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Timeline chart
            st.markdown('<h2 class="section-title">Traffic Timeline</h2>', unsafe_allow_html=True)
            
            if results['avg_people_counts']:
                times = [i * 10 for i in results['avg_people_counts'].keys()]
                counts = list(results['avg_people_counts'].values())
                
                # Create clean matplotlib chart
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # Minimal styling
                ax.set_facecolor('#fafafa')
                fig.patch.set_facecolor('#ffffff')
                
                # Plot with clean styling
                ax.plot(times, counts, linewidth=2.5, color='#111827', marker='o', 
                       markersize=6, markerfacecolor='#111827', markeredgecolor='white', 
                       markeredgewidth=2)
                
                # Fill area with subtle color
                ax.fill_between(times, counts, alpha=0.1, color='#111827')
                
                # Clean styling
                ax.set_xlabel("Time (seconds)", fontsize=12, color='#374151')
                ax.set_ylabel("Average People Count", fontsize=12, color='#374151')
                ax.set_title("Foot Traffic Over Time", fontsize=16, fontweight='600', 
                           color='#111827', pad=20)
                
                # Minimal grid
                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)
                
                # Clean spines
                for spine in ax.spines.values():
                    spine.set_color('#e5e7eb')
                    spine.set_linewidth(1)
                
                # Set y-axis to start from 0
                ax.set_ylim(bottom=0)
                
                # Clean tick styling
                ax.tick_params(colors='#6b7280', labelsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # Download section for video
            st.markdown('<h2 class="section-title">Export Results</h2>', unsafe_allow_html=True)
            
            col_center_video = st.columns([1, 2, 1])[1]
            with col_center_video:
                if os.path.exists(results['output_path']):
                    with open(results['output_path'], "rb") as video_file:
                        st.download_button(
                            label="Download Annotated Video",
                            data=video_file.read(),
                            file_name="retail_analysis_video.mp4",
                            mime="video/mp4"
                        )

else:
    # Welcome message
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: #111827;">Get Started</h3>
        <p style="color: #6b7280; margin-bottom: 1.5rem;">
            Upload an image or video to analyze customer foot traffic patterns with AI-powered detection.
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">👥</div>
                <div style="font-weight: 500; color: #111827;">People Detection</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Accurate customer counting</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔥</div>
                <div style="font-weight: 500; color: #111827;">Traffic Heatmaps</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Visual flow patterns</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 500; color: #111827;">Analytics</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Crowding & optimization insights</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True) class="image-title">Original Image</h4>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                display_original = resize_for_display(results['original_img'])
                st.image(display_original, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="image-card">
                    <h4 class="image-title">People Detection ({results['people_count']} found)</h4>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                display_annotated = resize_for_display(results['annotated_img'])
                st.image(display_annotated, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Heatmap analysis
            st.markdown('<h2 class="section-title">Traffic Analysis</h2>', unsafe_allow_html=True)
            
            # Crowding analysis
            crowded_locations = detect_overcrowding(results['heatmap'], overcrowding_sensitivity)
            
            if crowded_locations:
                st.markdown(f'<div class="alert alert-warning">⚠ Crowded areas detected: {", ".join(crowded_locations)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert alert-info">✓ No overcrowding detected - good traffic distribution</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("""
                <div class="image-card">
                    <h4 class="image-title">Traffic Heatmap</h4>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                if results['heatmap'].max() > 0:
                    clean_heatmap = create_clean_heatmap(results['heatmap'], "", "Blues")
                    st.image(clean_heatmap, use_container_width=True)
                else:
                    st.info("No traffic detected")
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="image-card">
                    <h4
