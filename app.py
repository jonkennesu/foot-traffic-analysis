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

# Configure the page
st.set_page_config(
    page_title="Retail Analytics", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Custom header */
    .app-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .app-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 400;
        margin: 0;
        opacity: 0.8;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Metrics grid */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Image containers */
    .image-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .image-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .image-content {
        padding: 1.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.95);
        transform: translateY(-2px);
    }
    
    /* Alerts */
    .success-alert {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    .info-alert {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 8px;
    }
    
    /* Selectbox and sliders */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stSlider > div > div > div {
        background: #f1f5f9;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2rem;
        }
        
        .app-header {
            padding: 2rem 1.5rem;
        }
        
        .metrics-container {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Modern matplotlib styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    sns.heatmap(heatmap_norm, cmap=cmap, cbar=False, 
                xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='600', color='#1e293b', pad=20)
    ax.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', dpi=150)
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
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    sns.heatmap(heatmap_norm, cmap="hot", cbar=False, 
                xticklabels=False, yticklabels=False, ax=ax)
    ax.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0,
                facecolor='white', dpi=150)
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
        
        # Convert frame to RGB first
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        slice_index = frame_count // slice_frame_count
        
        # Store representative frame for each slice
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

# App Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Retail Analytics</h1>
    <p class="app-subtitle">AI-powered customer flow analysis and crowd detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚙️ Detection Settings</div>', unsafe_allow_html=True)
    
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
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">💡 NMS Explanation</div>', unsafe_allow_html=True)
    st.markdown("""
    **Non-Maximum Suppression** removes duplicate detections:
    - **Lower values (0.1-0.3)**: Stricter, removes more overlaps
    - **Higher values (0.5-0.9)**: Allows more overlap
    - **For crowded scenes**: Use lower values
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎯 Features</div>', unsafe_allow_html=True)
    st.markdown("""
    • **Real-time Detection** with confidence scoring  
    • **Visual Heatmaps** of traffic patterns  
    • **Crowding Analysis** with location identification  
    • **Time-based Insights** for video content  
    • **Export Options** for reporting  
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# File Upload Section
st.markdown('<h2 class="section-header">📁 Upload Content</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image or video file", 
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
            with st.spinner("🔄 Processing image..."):
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
            
            st.markdown(f'<div class="success-alert">✅ Processing complete! Detected {people_count} people</div>', unsafe_allow_html=True)
            
        elif file_type == 'video':
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file_bytes)
                temp_video_path = tmp.name
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎥 Processing video..."):
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
            st.markdown('<div class="success-alert">✅ Video processing complete!</div>', unsafe_allow_html=True)
    
    # Display results
    if st.session_state.results is not None:
        if st.session_state.file_type == 'image':
            results = st.session_state.results
            
            # Metrics Section
            st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Create metrics
            crowded_count = len(detect_overcrowding(results['heatmap'], overcrowding_sensitivity))
            density = "High" if results['people_count'] > 10 else "Medium" if results['people_count'] > 5 else "Low"
            
            st.markdown(f"""
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-value">{results['people_count']}</div>
                    <div class="metric-label">People Detected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{crowded_count}</div>
                    <div class="metric-label">Crowded Areas</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{density}</div>
                    <div class="metric-label">Traffic Density</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Image Comparison
            st.markdown('<h2 class="section-header">🖼️ Detection Results</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("""
                <div class="image-container">
                    <div class="image-header">Original Image</div>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                display_original = resize_for_display(results['original_img'])
                st.image(display_original, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="image-container">
                    <div class="image-header">People Detection ({results['people_count']} found)</div>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                display_annotated = resize_for_display(results['annotated_img'])
                st.image(display_annotated, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Heatmap Analysis
            st.markdown('<h2 class="section-header">🔥 Traffic Flow Analysis</h2>', unsafe_allow_html=True)
            
            # Crowding analysis
            crowded_locations = detect_overcrowding(results['heatmap'], overcrowding_sensitivity)
            
            if crowded_locations:
                locations_text = ", ".join(crowded_locations)
                st.markdown(f'<div class="warning-alert">⚠️ Crowded areas detected: {locations_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-alert">✅ No overcrowding detected - optimal traffic distribution</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2, gap="large")
            
            with col3:
                st.markdown("""
                <div class="image-container">
                    <div class="image-header">Traffic Heatmap</div>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                if results['heatmap'].max() > 0:
                    clean_heatmap = create_clean_heatmap(results['heatmap'], "", "Blues")
                    st.image(clean_heatmap, use_container_width=True)
                else:
                    st.info("No foot traffic detected")
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="image-container">
                    <div class="image-header">Overlay Analysis</div>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                if results['heatmap'].max() > 0:
                    overlay_img = create_heatmap_overlay(results['original_img'], results['heatmap'])
                    st.image(overlay_img, use_container_width=True)
                else:
                    display_original_2 = resize_for_display(results['original_img'])
                    st.image(display_original_2, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Download Section
            st.markdown('<h2 class="section-header">📥 Export Results</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col_center = st.columns([1, 2, 1])[1]
            with col_center:
                annotated_pil = Image.fromarray(results['annotated_img'])
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="retail_analysis_annotated.png",
                    mime="image/png"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif st.session_state.file_type == 'video':
            results = st.session_state.results
            
            # Video Metrics
            st.markdown('<h2 class="section-header">📊 Video Analysis Results</h2>', unsafe_allow_html=True)
            
            total_people = sum(results['avg_people_counts'].values())
            max_people = max(results['avg_people_counts'].values()) if results['avg_people_counts'] else 0
            avg_people = np.mean(list(results['avg_people_counts'].values())) if results['avg_people_counts'] else 0
            duration = results['num_slices'] * 10
            
            st.markdown(f"""
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-value">{total_people:.1f}</div>
                    <div class="metric-label">Total Avg People</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{max_people:.1f}</div>
                    <div class="metric-label">Peak Occupancy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_people:.1f}</div>
                    <div class="metric-label">Average Occupancy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{duration}s</div>
                    <div class="metric-label">Duration</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Time Interval Analysis
            st.markdown('<h2 class="section-header">🎯 Time Interval Analysis</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            interval_options = []
            for i in range(results['num_slices']):
                start_time = i * 10
                end_time = (i + 1) * 10
                avg_count = results['avg_people_counts'].get(i, 0)
                interval_options.append(f"{start_time}s-{end_time}s (Avg: {avg_count:.1f} people)")
            
            selected_interval = st.selectbox("Select Time Interval:", interval_options)
            selected_index = interval_options.index(selected_interval)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if selected_index in results['slice_frames']:
                frame = results['slice_frames'][selected_index]
                heatmap = results['heatmaps'][selected_index]
                
                # Display frame
                st.markdown(f"""
                <div class="image-container">
                    <div class="image-header">Frame Analysis: {selected_interval}</div>
                    <div class="image-content">
                """, unsafe_allow_html=True)
                display_frame = resize_for_display(frame)
                st.image(display_frame, use_container_width=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Crowd analysis for this interval
                crowded_locations = detect_overcrowding(heatmap, overcrowding_sensitivity)
                
                if crowded_locations:
                    locations_text = ", ".join(crowded_locations)
                    st.markdown(f'<div class="warning-alert">⚠️ Crowded areas in this interval: {locations_text}</div>', unsafe_allow_html=True)
                
                # Interval Heatmaps
                st.markdown('<h3 class="section-header">🔥 Interval Heatmap Analysis</h3>', unsafe_allow_html=True)
                
                col5, col6 = st.columns(2, gap="large")
                
                with col5:
                    st.markdown("""
                    <div class="image-container">
                        <div class="image-header">Traffic Heatmap</div>
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
                    <div class="image-container">
                        <div class="image-header">Overlay Analysis</div>
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
            
            # Timeline Chart
            st.markdown('<h2 class="section-header">📈 Traffic Timeline</h2>', unsafe_allow_html=True)
            
            if results['avg_people_counts']:
                times = [i * 10 for i in results['avg_people_counts'].keys()]
                counts = list(results['avg_people_counts'].values())
                
                # Create modern chart
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
                
                # Modern styling
                ax.set_facecolor('#fafafa')
                
                # Plot with modern colors
                ax.plot(times, counts, linewidth=3, color='#667eea', marker='o', 
                       markersize=8, markerfacecolor='#764ba2', markeredgecolor='white', 
                       markeredgewidth=2, alpha=0.9)
                
                # Fill area with gradient effect
                ax.fill_between(times, counts, alpha=0.2, color='#667eea')
                
                # Modern styling
                ax.set_xlabel("Time (seconds)", fontsize=14, fontweight='600', color='#1e293b')
                ax.set_ylabel("Average People Count", fontsize=14, fontweight='600', color='#1e293b')
                ax.set_title("Foot Traffic Over Time", fontsize=18, fontweight='700', 
                           color='#1e293b', pad=25)
                
                # Clean grid
                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
                ax.set_axisbelow(True)
                
                # Modern spines
                for spine in ax.spines.values():
                    spine.set_color('#e2e8f0')
                    spine.set_linewidth(1.2)
                
                # Set y-axis to start from 0
                ax.set_ylim(bottom=0)
                
                # Clean tick styling
                ax.tick_params(colors='#64748b', labelsize=11)
                
                plt.tight_layout()
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig)
            
            # Download Section for Video
            st.markdown('<h2 class="section-header">📥 Export Results</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col_center_video = st.columns([1, 2, 1])[1]
            with col_center_video:
                if os.path.exists(results['output_path']):
                    with open(results['output_path'], "rb") as video_file:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=video_file.read(),
                            file_name="retail_analysis_video.mp4",
                            mime="video/mp4"
                        )
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome Screen
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top: 0; color: #1e293b; font-weight: 600; font-size: 1.5rem;">🚀 Get Started</h3>
        <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
            Upload an image or video to analyze customer foot traffic patterns with AI-powered detection and generate actionable insights for your retail space.
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 2rem;">
            <div style="text-align: center; padding: 1.5rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">👥</div>
                <div style="font-weight: 600; color: #1e293b; font-size: 1.1rem; margin-bottom: 0.5rem;">People Detection</div>
                <div style="font-size: 0.95rem; color: #64748b;">Accurate real-time customer counting with confidence scores</div>
            </div>
            <div style="text-align: center; padding: 1.5rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🔥</div>
                <div style="font-weight: 600; color: #1e293b; font-size: 1.1rem; margin-bottom: 0.5rem;">Traffic Heatmaps</div>
                <div style="font-size: 0.95rem; color: #64748b;">Visual flow patterns and high-activity zone identification</div>
            </div>
            <div style="text-align: center; padding: 1.5rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📊</div>
                <div style="font-weight: 600; color: #1e293b; font-size: 1.1rem; margin-bottom: 0.5rem;">Smart Analytics</div>
                <div style="font-size: 0.95rem; color: #64748b;">Crowding detection and space optimization insights</div>
            </div>
        </div>
        
        <div style="margin-top: 2.5rem; padding: 1.5rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px; border-left: 4px solid #667eea;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">💡 Pro Tip</div>
            <div style="color: #64748b;">For best results, ensure good lighting and clear view of people in your images or videos. The AI works best with high-resolution content.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
