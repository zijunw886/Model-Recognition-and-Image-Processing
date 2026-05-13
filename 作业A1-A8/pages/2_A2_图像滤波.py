"""
Computer Vision Assignment A2
Spatial Filtering, Gradient Analysis, and Frequency Domain Filtering
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="A2: Spatial & Frequency Domain Processing",
    page_icon="🔍",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create a built-in test image if pic.jpg doesn't exist
def create_test_image():
    """Create a test image with various patterns for filtering demonstration"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add a gradient background
    for i in range(300):
        for j in range(400):
            img[i, j] = [int(255 * j / 400), int(255 * i / 300), 128]
    
    # Add a square
    img[50:150, 50:150] = [255, 0, 0]
    
    # Add a circle
    yy, xx = np.ogrid[:300, :400]
    circle_mask = (xx - 300)**2 + (yy - 150)**2 <= 40**2
    img[circle_mask] = [0, 255, 0]
    
    return img

def load_image():
    """Load image from root directory or create test image"""
    img_path = os.path.join(ROOT_DIR, 'pic.jpg')
    
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            return np.array(img)
        except:
            st.warning("无法读取图片，使用内置测试图像")
            return create_test_image()
    else:
        st.warning("图片文件不存在，使用内置测试图像")
        return create_test_image()

# ==================== Spatial Domain Filtering ====================
def spatial_filtering(image):
    """Apply spatial domain filters"""
    st.header("空间域滤波")
    st.markdown("---")
    
    # Convert to grayscale for filtering
    gray = np.mean(image, axis=2).astype(np.uint8)
    
    # Filter type selection
    filter_type = st.selectbox("选择滤波类型", ["均值滤波", "高斯滤波", "中值滤波", "Sobel边缘检测"])
    
    if filter_type == "均值滤波":
        kernel_size = st.slider("卷积核大小", 3, 15, 5, step=2)
        if st.button("应用均值滤波", key="mean_filter"):
            result = Image.fromarray(gray).filter(ImageFilter.BoxBlur(kernel_size//2))
            show_comparison(gray, np.array(result), "原图", f"均值滤波 ({kernel_size}x{kernel_size})")
    
    elif filter_type == "高斯滤波":
        kernel_size = st.slider("卷积核大小", 3, 15, 5, step=2)
        sigma = st.slider("高斯标准差", 0.1, 5.0, 1.0)
        if st.button("应用高斯滤波", key="gaussian_filter"):
            result = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=sigma))
            show_comparison(gray, np.array(result), "原图", f"高斯滤波 (sigma={sigma})")
    
    elif filter_type == "中值滤波":
        kernel_size = st.slider("卷积核大小", 3, 15, 5, step=2)
        if st.button("应用中值滤波", key="median_filter"):
            result = Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=kernel_size))
            show_comparison(gray, np.array(result), "原图", f"中值滤波 ({kernel_size}x{kernel_size})")
    
    elif filter_type == "Sobel边缘检测":
        direction = st.selectbox("边缘方向", ["水平边缘", "垂直边缘", "双向边缘"])
        if st.button("应用Sobel边缘检测", key="sobel_filter"):
            # Sobel kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            
            h, w = gray.shape
            out = np.zeros_like(gray, dtype=np.float32)
            
            if direction == "水平边缘":
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        neighborhood = gray[i-1:i+2, j-1:j+2].astype(np.float32)
                        out[i,j] = np.sum(sobel_y * neighborhood)
            elif direction == "垂直边缘":
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        neighborhood = gray[i-1:i+2, j-1:j+2].astype(np.float32)
                        out[i,j] = np.sum(sobel_x * neighborhood)
            else:
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        neighborhood = gray[i-1:i+2, j-1:j+2].astype(np.float32)
                        gx = np.sum(sobel_x * neighborhood)
                        gy = np.sum(sobel_y * neighborhood)
                        out[i,j] = np.sqrt(gx**2 + gy**2)
            
            # Normalize to 0-255
            out = np.clip(np.abs(out), 0, 255).astype(np.uint8)
            show_comparison(gray, out, "原图", f"Sobel边缘检测 ({direction})")

def show_comparison(original, processed, title1, title2):
    """Show side-by-side comparison of two images"""
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption=title1, use_container_width=True, clamp=True)
    with col2:
        st.image(processed, caption=title2, use_container_width=True, clamp=True)

# ==================== Gradient Analysis ====================
def gradient_analysis(image):
    """Analyze image gradients and local regions"""
    st.header("图像梯度与局部区域分析")
    st.markdown("---")
    
    # Convert to grayscale
    gray = np.mean(image, axis=2).astype(np.uint8)
    h, w = gray.shape
    
    # Select region with user input
    st.subheader("选择局部区域")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x1 = st.number_input("起始X", 0, w-1, 50, key="grad_x1")
    with col2:
        y1 = st.number_input("起始Y", 0, h-1, 50, key="grad_y1")
    with col3:
        x2 = st.number_input("结束X", 0, w-1, 150, key="grad_x2")
    with col4:
        y2 = st.number_input("结束Y", 0, h-1, 150, key="grad_y2")
    
    # Ensure x1 < x2, y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    if st.button("分析梯度", key="analyze_grad"):
        # Extract region
        region = gray[y1:y2, x1:x2]
        r_h, r_w = region.shape
        
        # Check if region is large enough for Sobel
        if r_h < 3 or r_w < 3:
            st.error("选择的区域太小！请选择至少3x3的区域")
            return
        
        # Calculate gradients using Sobel
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Initialize magnitude array
        magnitude = np.zeros_like(region, dtype=np.float32)
        
        # Apply Sobel filter
        for i in range(1, r_h - 1):
            for j in range(1, r_w - 1):
                # Extract neighborhood
                neighborhood = region[i-1:i+2, j-1:j+2].astype(np.float32)
                # Calculate gradients
                gx = np.sum(sobel_x * neighborhood)
                gy = np.sum(sobel_y * neighborhood)
                # Magnitude
                magnitude[i, j] = np.sqrt(gx**2 + gy**2)
        
        # Normalize magnitude to 0-255
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        # Display results
        st.subheader("局部区域与梯度结果")
        col1, col2 = st.columns(2)
        with col1:
            st.image(region, caption=f"局部区域 ({x1},{y1}) - ({x2},{y2})", use_container_width=True, clamp=True)
        with col2:
            st.image(magnitude, caption="梯度幅值", use_container_width=True, clamp=True)
        
        # Plot gradient histogram
        st.subheader("梯度幅值直方图")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(magnitude.flatten(), bins=50, range=[0, 255], color='blue', alpha=0.7)
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_title('Gradient Magnitude Histogram')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ==================== Frequency Domain Filtering ====================
def frequency_domain_filtering(image):
    """Frequency domain processing with FFT"""
    st.header("频域滤波（傅里叶变换）")
    st.markdown("---")
    
    # Convert to grayscale
    gray = np.mean(image, axis=2).astype(np.uint8)
    
    # Original FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fft_shift) + 1e-6)  # Add epsilon to avoid log(0)
    
    # Display original spectrum
    st.subheader("原始图像频谱")
    col1, col2 = st.columns(2)
    with col1:
        st.image(gray, caption="原图", use_container_width=True, clamp=True)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(magnitude, cmap='gray')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('FFT Spectrum')
        st.pyplot(fig)
        plt.close(fig)

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("🔍 作业A2: 空间域滤波与频域分析")
    st.markdown("**Spatial & Frequency Domain Processing**")
    st.markdown("---")
    
    # Load image
    image = load_image()
    
    # Sidebar preview
    st.sidebar.header("图像预览")
    st.sidebar.image(image, use_container_width=True)
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["空间域滤波", "图像梯度分析", "频域滤波"])
    
    with tab1:
        spatial_filtering(image)
    
    with tab2:
        gradient_analysis(image)
    
    with tab3:
        frequency_domain_filtering(image)

if __name__ == "__main__":
    main()
