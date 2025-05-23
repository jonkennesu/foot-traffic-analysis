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
import plotly.express as px
import plotly.graph_objects as go

# Configure page with custom theme
st.set_page_config(
    page_title="Retail Foot Traffic Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏪"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background and typography */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #2c3e50;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #4facfe;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-content {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Image containers */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Stats container */
    .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        border: none;
        border-radius: 10px;
    }
    
    /* Upload area */
    .uploadedFile {
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        border: 2px dashed #4facfe;
    }
    
    /* Button styling */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        text-align: center;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4facfe;
        margin: 1rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
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

# Main header with beautiful styling
st.markdown("""
<div class="main-header">
    <h1>🏪 Retail Foot Traffic Analyzer</h1>
    <p>Analyze customer movement patterns and optimize your retail space with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with better styling
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Detection Settings")
    
    # Model loading
    @st.cache_resource
    def get_model():
        try:
            return load_model("best.pt")
        except:
            st.error("🚨 Model file 'best.pt' not found. Please ensure the YOLO model is available.")
            return None

    model = get_model()

    if model is None:
        st.stop()

    # Threshold controls with better spacing
    st.markdown("#### 🎯 Detection Parameters")
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
        help="Lower values = more sensitive to overcrowding"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced info section
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. **Upload** your image or video
    2. **AI Detection** identifies all people
    3. **Heatmap Creation** shows traffic patterns
    4. **Analysis** reveals crowded areas
    5. **Insights** help optimize your space
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 🎯 Key Features")
    st.markdown("""
    • **Real-time Detection** with confidence scores  
    • **Visual Heatmaps** of foot traffic patterns  
    • **Automatic Crowding** detection  
    • **Time-based Analysis** for videos  
    • **Downloadable Results** for reports  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced file upload section
st.markdown('<div class="section-header"><h2>📁 Upload Your Content</h2></div>', unsafe_allow_html=True)

col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
with col_upload2:
    uploaded_file = st.file_uploader(
        "Choose an image or video file", 
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
        help="Supported formats: JPG, PNG for images; MP4, MOV, AVI for videos",
        label_visibility="collapsed"
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
            st.markdown('<div class="section-header"><h2>📊 Image Analysis Results</h2></div>', unsafe_allow_html=True)
            
            with st.spinner("🔄 Processing your image..."):
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
            
            st.success(f"✅ Processing complete! Detected **{people_count}** people in your image.")
            
        elif file_type == 'video':
            st.markdown('<div class="section-header"><h2>🎥 Video Analysis Results</h2></div>', unsafe_allow_html=True)
            
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file_bytes)
                temp_video_path = tmp.name
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔄 Processing your video... This may take a few minutes."):
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
            st.success("✅ Video processing complete! Your analysis is ready.")
    
    # Display results with enhanced styling
    if st.session_state.results is not None:
        if st.session_state.file_type == 'image':
            results = st.session_state.results
            
            # Quick stats at the top
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("👥 People Detected", results['people_count'], delta=None)
            with col_stat2:
                crowded_count = len(detect_overcrowding(results['heatmap'], overcrowding_sensitivity))
                st.metric("🚨 Crowded Areas", crowded_count, delta=None)
            with col_stat3:
                density = "High" if results['people_count'] > 10 else "Medium" if results['people_count'] > 5 else "Low"
                st.metric("📊 Traffic Density", density, delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display original and annotated images
            st.markdown('<div class="section-header"><h3>🖼️ Image Comparison</h3></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.markdown("#### Original Image")
                display_original = resize_for_display(results['original_img'])
                st.image(display_original, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.markdown(f"#### Detected People: {results['people_count']}")
                display_annotated = resize_for_display(results['annotated_img'])
                st.image(display_annotated, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Heatmap analysis with better styling
            st.markdown('<div class="section-header"><h3>🔥 Foot Traffic Heatmap Analysis</h3></div>', unsafe_allow_html=True)
            
            # Detect crowd locations
            crowded_locations = detect_overcrowding(results['heatmap'], overcrowding_sensitivity)
            
            # Show crowded locations if any
            if crowded_locations:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**🚨 Crowded Areas Detected:** {', '.join(crowded_locations)}")
                st.markdown("Consider optimizing these areas to improve customer flow and reduce congestion.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**✅ No Overcrowding Detected** - Your space appears to have good traffic distribution!")
                st.markdown('</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.markdown("#### Raw Heatmap")
                if results['heatmap'].max() > 0:
                    clean_heatmap = create_clean_heatmap(results['heatmap'], "Foot Traffic Density", "Blues")
                    st.image(clean_heatmap, use_container_width=True)
                else:
                    st.info("No foot traffic detected")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.markdown("#### Overlay Heatmap")
                if results['heatmap'].max() > 0:
                    overlay_img = create_heatmap_overlay(results['original_img'], results['heatmap'])
                    st.image(overlay_img, use_container_width=True)
                else:
                    display_original_2 = resize_for_display(results['original_img'])
                    st.image(display_original_2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Download section
            st.markdown('<div class="section-header"><h3>📥 Download Results</h3></div>', unsafe_allow_html=True)
            col_download = st.columns([1, 2, 1])[1]
            with col_download:
                annotated_pil = Image.fromarray(results['annotated_img'])
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="retail_analysis_annotated.png",
                    mime="image/png"
                )
            
        elif st.session_state.file_type == 'video':
            results = st.session_state.results
            
            # Video stats overview
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            with col_v1:
                total_people = sum(results['avg_people_counts'].values())
                st.metric("👥 Total Avg People", f"{total_people:.1f}")
            with col_v2:
                max_people = max(results['avg_people_counts'].values()) if results
