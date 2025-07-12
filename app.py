import streamlit as st
from pathlib import Path
import datetime
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import numpy as np

from model import run, load_image

def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .content-section {
        background: rgba(255, 255, 255, 0.01);
        backdrop-filter: blur(20px);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.03);
        transition: all 0.3s ease;
    }
    
    .content-section:hover {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .sidebar-container {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .primary-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 1rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    .title-header {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 300;
        text-align: center;
        margin: 2rem 0 1rem 0;
        letter-spacing: 2px;
    }
    
    .subtitle-text {
        color: rgba(255, 255, 255, 0.6);
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 3rem;
        letter-spacing: 1px;
    }
    
    .section-title {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }
    
    .info-text {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-2px);
    }
    
    .image-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.01);
    }
    
    .error-container {
        background: rgba(220, 53, 69, 0.1);
        border: 1px solid rgba(220, 53, 69, 0.2);
        backdrop-filter: blur(20px);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .success-container {
        background: rgba(40, 167, 69, 0.1);
        border: 1px solid rgba(40, 167, 69, 0.2);
        backdrop-filter: blur(20px);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-container {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.2);
        backdrop-filter: blur(20px);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .sidebar-header {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 300;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    
    .sidebar-section {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        font-weight: 400;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        font-weight: 300;
    }
    </style>
    """, unsafe_allow_html=True)

def resize_image_safely(image, max_size):
    width, height = image.size
    
    if width > height:
        new_width = max_size
        new_height = int((height * max_size) / width)
    else:
        new_height = max_size
        new_width = int((width * max_size) / height)
    
    new_width = max(new_width, 224)
    new_height = max(new_height, 224)
    
    if new_width % 4 != 0:
        new_width = (new_width // 4) * 4
    if new_height % 4 != 0:
        new_height = (new_height // 4) * 4
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def preprocess_image_for_display(image_file, max_size):
    try:
        if hasattr(image_file, 'read'):
            image = Image.open(image_file)
        else:
            image = Image.open(image_file)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = resize_image_safely(image, max_size)
        
        return image
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def preprocess_image(image_file, max_size):
    try:
        if hasattr(image_file, 'read'):
            image_file.seek(0)
            return image_file
        else:
            return open(image_file, 'rb')
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def apply_filter_effects(image, filter_type):
    if filter_type == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1))
    elif filter_type == "Enhance":
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)
    elif filter_type == "Smooth":
        return image.filter(ImageFilter.SMOOTH)
    return image

def main():
    st.set_page_config(
        page_title="ImaginAI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    add_custom_css()
    
    st.sidebar.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 class="sidebar-header">ImaginAI</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<p class="sidebar-section">Image Sources</p>', unsafe_allow_html=True)
    
    content_image_file = st.sidebar.file_uploader(
        "Content Image",
        type=["jpg", "jpeg", "png"],
        help="Select the main image to be stylized"
    )
    
    style_mode = st.sidebar.radio(
        "Style Source",
        ("Upload", "Gallery"),
        help="Choose style source method"
    )
    
    if style_mode == "Upload":
        style_image_file = st.sidebar.file_uploader(
            "Style Image",
            type=["jpg", "jpeg", "png"],
            help="Select the style reference image"
        )
    else:
        style_files = sorted(
            list(Path("styles").glob("*.jpg")) + list(Path("styles").glob("*.png"))
        )
        if style_files:
            style_names = [p.name for p in style_files]
            selected_style = st.sidebar.selectbox("Select Style", style_names)
            style_path = Path("styles") / selected_style
            style_image_file = open(style_path, "rb")
            
            with st.sidebar:
                st.markdown('<p class="sidebar-section">Preview</p>', unsafe_allow_html=True)
                preview_img = Image.open(style_path)
                st.image(preview_img, use_column_width=True)
            
            style_image_file.seek(0)
        else:
            st.sidebar.warning("No style images found")
            style_image_file = None
    
    st.sidebar.markdown('<p class="sidebar-section">Processing Settings</p>', unsafe_allow_html=True)
    
    max_size = st.sidebar.slider(
        "Output Resolution",
        256, 1024, 400, 32,
        help="Image output resolution in pixels"
    )
    
    epochs = st.sidebar.number_input(
        "Training Epochs",
        min_value=1, max_value=5000, value=200, step=25,
        help="Number of training iterations"
    )
    
    style_weight = st.sidebar.slider(
        "Style Intensity",
        0.1, 2.0, 1.0, 0.1,
        help="Style application strength"
    )
    
    st.sidebar.markdown('<p class="sidebar-section">Post Processing</p>', unsafe_allow_html=True)
    
    post_filter = st.sidebar.selectbox(
        "Image Filter",
        ["None", "Sharpen", "Enhance", "Smooth", "Blur"],
        help="Apply post-processing filter"
    )
    
    save_intermediate = st.sidebar.checkbox(
        "Save Intermediate Results",
        help="Save images during processing"
    )
    
    st.markdown('<h1 class="title-header">ImaginAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Advanced AI-powered artistic style transformation</p>', unsafe_allow_html=True)
    
    if content_image_file and style_image_file:
        try:
            content_img = preprocess_image_for_display(content_image_file, max_size)
            style_img = preprocess_image_for_display(style_image_file, max_size)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown('<div class="content-section">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">Content</h3>', unsafe_allow_html=True)
                st.image(content_img, use_column_width=True)
                st.markdown(f'<p class="info-text">Resolution: {content_img.size[0]} × {content_img.size[1]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text">Format: RGB</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="content-section">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">Style</h3>', unsafe_allow_html=True)
                st.image(style_img, use_column_width=True)
                st.markdown(f'<p class="info-text">Resolution: {style_img.size[0]} × {style_img.size[1]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text">Format: RGB</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="content-section">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">Configuration</h3>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text">Output: {max_size}px</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text">Epochs: {epochs}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text">Style Weight: {style_weight}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text">Filter: {post_filter}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            generate_col1, generate_col2, generate_col3 = st.columns([1, 1, 1])
            with generate_col2:
                if st.button("Generate", use_container_width=True):
                    st.markdown('<div class="content-section">', unsafe_allow_html=True)
                    st.markdown('<h3 class="section-title">Processing</h3>', unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("Initializing...")
                        
                        content_file = preprocess_image(content_image_file, max_size)
                        style_file = preprocess_image(style_image_file, max_size)
                        
                        content_tensor = load_image(content_file, max_size=max_size)
                        style_tensor = load_image(style_file, max_size=max_size)
                        
                        gen_image = None
                        intermediate_images = []
                        
                        status_text.text("Processing style transfer...")
                        
                        try:
                            for step, gen_image in run(content_tensor, style_tensor, epochs):
                                progress = (step + 1) / epochs
                                progress_bar.progress(progress)
                                status_text.text(f"Epoch {step + 1}/{epochs}")
                                
                                if save_intermediate and step % 50 == 0:
                                    intermediate_images.append(gen_image.copy())
                        except TypeError:
                            gen_image = run(content_tensor, style_tensor, epochs)
                            progress_bar.progress(1.0)
                            status_text.text("Transfer complete")
                        
                        if gen_image is None:
                            raise Exception("Processing failed - no output generated")
                        
                        if post_filter != "None":
                            status_text.text("Applying filters...")
                            gen_image = apply_filter_effects(gen_image, post_filter)
                        
                        status_text.text("Complete")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="success-container">', unsafe_allow_html=True)
                        st.markdown('<h3 class="section-title">Result</h3>', unsafe_allow_html=True)
                        st.image(gen_image, use_column_width=True)
                        
                        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
                        
                        with result_col1:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            out_path = Path("outputs")
                            out_path.mkdir(exist_ok=True)
                            final_file = out_path / f"result_{ts}.jpg"
                            gen_image.save(final_file)
                            
                            buf = io.BytesIO()
                            gen_image.save(buf, format="JPEG", quality=95)
                            st.download_button(
                                "Download JPEG",
                                buf.getvalue(),
                                file_name=f"result_{ts}.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                        
                        with result_col2:
                            buf_png = io.BytesIO()
                            gen_image.save(buf_png, format="PNG")
                            st.download_button(
                                "Download PNG",
                                buf_png.getvalue(),
                                file_name=f"result_{ts}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        with result_col3:
                            if st.button("Process Again", use_container_width=True):
                                st.rerun()
                        
                        if save_intermediate and intermediate_images:
                            st.markdown('<h3 class="section-title">Intermediate Results</h3>', unsafe_allow_html=True)
                            cols = st.columns(len(intermediate_images))
                            for idx, img in enumerate(intermediate_images):
                                with cols[idx]:
                                    st.image(img, caption=f"Step {(idx+1)*50}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="error-container">', unsafe_allow_html=True)
                        st.markdown('<h3 class="section-title">Processing Error</h3>', unsafe_allow_html=True)
                        st.markdown(f'<p class="info-text">Error: {str(e)}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="info-text">Try reducing resolution or using different images</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
        except Exception as e:
            st.markdown('<div class="error-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Image Processing Error</h3>', unsafe_allow_html=True)
            st.markdown(f'<p class="info-text">Error: {str(e)}</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-text">Please check image files and try again</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Getting Started</h3>', unsafe_allow_html=True)
        st.markdown('''
        <p class="info-text">
        Neural style transfer uses deep learning to combine the content of one image 
        with the artistic style of another. Upload both images to begin the transformation process.
        </p>
        
        <p class="info-text">
        <strong>Process:</strong><br>
        1. Upload or select your content image<br>
        2. Upload or choose a style reference<br>
        3. Configure processing parameters<br>
        4. Generate your stylized result
        </p>
        
        <p class="info-text">
        <strong>Recommendations:</strong><br>
        • Start with 400px resolution for faster processing<br>
        • Use 200-300 epochs for optimal results<br>
        • Experiment with style weight values<br>
        • Enable intermediate saving to monitor progress
        </p>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
        st.markdown('''
        <div class="stat-card">
            <h4 class="metric-value">Neural Networks</h4>
            <p class="metric-label">Deep learning architecture</p>
        </div>
        <div class="stat-card">
            <h4 class="metric-value">Style Transfer</h4>
            <p class="metric-label">Artistic transformation</p>
        </div>
        <div class="stat-card">
            <h4 class="metric-value">High Quality</h4>
            <p class="metric-label">Professional output</p>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="warning-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">System Requirements</h3>', unsafe_allow_html=True)
        st.markdown('''
        <p class="info-text">
        • Supported formats: JPEG, PNG<br>
        • Images are converted to RGB format<br>
        • Processing time varies with resolution and epochs<br>
        • Model requires specific input dimensions
        </p>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()