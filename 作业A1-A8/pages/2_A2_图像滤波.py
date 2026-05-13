"""
Computer Vision Assignment A2
Spatial Filtering, Gradient Analysis, and Frequency Domain Filtering
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
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
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    
    # Add a circle
    cv2.circle(img, (300, 150), 40, (0, 255, 0), -1)
    
    # Add diagonal line
    cv2.line(img, (50, 250), (350, 50), (0, 0, 255), 3)
    
    # Add text
    cv2.putText(img, "CV A2", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return img

def load_image():
    """Load image from root directory"""
    img_path = os.path.join(ROOT_DIR, 'pic.jpg')
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            st.warning(f"无法读取图片: {img_path}")
            return create_test_image()
        return img
    else:
        st.warning(f"图片文件不存在: {img_path}")
        img = create_test_image()
        cv2.imwrite(img_path, img)
        st.info(f"已创建测试图片到: {img_path}")
        return img

# ==================== Spatial Domain Filtering ====================
def spatial_filtering(image):
    """Apply spatial domain filters"""
    st.header("空间域滤波")
    st.markdown("---")
    
    # Convert to grayscale for filtering
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Filter type selection
    filter_type = st.selectbox("选择滤波类型", ["均值滤波", "高斯滤波", "中值滤波", "Sobel边缘检测"])
    
    if filter_type == "均值滤波":
        kernel_size = st.slider("卷积核大小", 3, 15, 5, step=2)
        if st.button("应用均值滤波", key="mean_filter"):
            result = cv2.blur(gray, (kernel_size, kernel_size))
            show_comparison(gray, result, "原图", f"均值滤波 ({kernel_size}x{kernel_size})")
    
    elif filter_type == "高斯滤波":
        kernel_size = st.slider("卷积核大小", 3, 15, 5, step=2)
        sigma = st.slider("高斯标准差", 0.1, 5.0, 1.0)
        if st.button("应用高斯滤波", key="gaussian_filter"):
            result = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
            show_comparison(gray, result, "原图", f"高斯滤波 (kernel={kernel_size}, sigma={sigma})")
    
    elif filter_type == "中值滤波":
        kernel_size = st.slider("卷积核大小", 3, 15, 5, step=2)
        if st.button("应用中值滤波", key="median_filter"):
            result = cv2.medianBlur(gray, kernel_size)
            show_comparison(gray, result, "原图", f"中值滤波 ({kernel_size}x{kernel_size})")
    
    elif filter_type == "Sobel边缘检测":
        kernel_size = st.slider("卷积核大小", 3, 7, 3, step=2)
        direction = st.selectbox("边缘方向", ["水平边缘", "垂直边缘", "双向边缘"])
        
        if st.button("应用Sobel边缘检测", key="sobel_filter"):
            if direction == "水平边缘":
                result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
            elif direction == "垂直边缘":
                result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            else:
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
                result = cv2.magnitude(sobel_x, sobel_y)
            
            result = cv2.convertScaleAbs(result)
            show_comparison(gray, result, "原图", f"Sobel边缘检测 ({direction})")

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Select region
    st.subheader("选择局部区域")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x1 = st.number_input("起始X", 0, gray.shape[1]-1, 50, key="grad_x1")
    with col2:
        y1 = st.number_input("起始Y", 0, gray.shape[0]-1, 50, key="grad_y1")
    with col3:
        x2 = st.number_input("结束X", 0, gray.shape[1]-1, 150, key="grad_x2")
    with col4:
        y2 = st.number_input("结束Y", 0, gray.shape[0]-1, 150, key="grad_y2")
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    if st.button("分析梯度", key="analyze_grad"):
        # Extract region
        region = gray[y1:y2, x1:x2]
        
        # Calculate gradients
        sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        
        # Display region
        st.subheader("局部区域")
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="原图", use_container_width=True, clamp=True)
            st.caption(f"选择区域: ({x1},{y1}) - ({x2},{y2})")
        with col2:
            st.image(region, caption="局部区域", use_container_width=True, clamp=True)
        
        # Display gradient magnitude and direction
        st.subheader("梯度分析结果")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.convertScaleAbs(magnitude), caption="梯度幅值", use_container_width=True, clamp=True)
        with col2:
            st.image(cv2.convertScaleAbs(direction), caption="梯度方向", use_container_width=True, clamp=True)
        
        # Plot gradient histogram
        st.subheader("梯度直方图")
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Original FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fft_shift))
    
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
    
    # Image transformation effects
    st.subheader("图像变换后的频域变化")
    transform_type = st.selectbox("选择变换类型", ["旋转", "平移", "缩放"])
    
    if transform_type == "旋转":
        angle = st.slider("旋转角度", 0, 360, 45, key="fft_rotate")
        if st.button("应用旋转并分析频域", key="apply_rotate_fft"):
            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(gray, M, (cols, rows))
            
            # FFT of rotated image
            fft_rot = np.fft.fft2(rotated)
            fft_shift_rot = np.fft.fftshift(fft_rot)
            mag_rot = 20 * np.log(np.abs(fft_shift_rot))
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(rotated, caption=f"旋转{angle}度", use_container_width=True, clamp=True)
            with col2:
                fig, ax = plt.subplots()
                ax.imshow(mag_rot, cmap='gray')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Spectrum (Rotated {angle}deg)')
                st.pyplot(fig)
                plt.close(fig)
    
    elif transform_type == "平移":
        tx = st.slider("X方向平移", -100, 100, 30, key="fft_tx")
        ty = st.slider("Y方向平移", -100, 100, 30, key="fft_ty")
        if st.button("应用平移并分析频域", key="apply_trans_fft"):
            rows, cols = gray.shape
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            translated = cv2.warpAffine(gray, M, (cols, rows))
            
            # FFT of translated image
            fft_trans = np.fft.fft2(translated)
            fft_shift_trans = np.fft.fftshift(fft_trans)
            mag_trans = 20 * np.log(np.abs(fft_shift_trans))
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(translated, caption=f"平移 (dx={tx}, dy={ty})", use_container_width=True, clamp=True)
            with col2:
                fig, ax = plt.subplots()
                ax.imshow(mag_trans, cmap='gray')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Spectrum (Translated dx={tx}, dy={ty})')
                st.pyplot(fig)
                plt.close(fig)
    
    elif transform_type == "缩放":
        scale = st.slider("缩放比例", 0.2, 2.0, 0.5, key="fft_scale")
        if st.button("应用缩放并分析频域", key="apply_scale_fft"):
            rows, cols = gray.shape
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
            # Pad to original size
            pad_rows = rows - scaled.shape[0]
            pad_cols = cols - scaled.shape[1]
            scaled = cv2.copyMakeBorder(scaled, 0, max(0, pad_rows), 0, max(0, pad_cols), cv2.BORDER_CONSTANT)
            scaled = scaled[:rows, :cols]
            
            # FFT of scaled image
            fft_scale = np.fft.fft2(scaled)
            fft_shift_scale = np.fft.fftshift(fft_scale)
            mag_scale = 20 * np.log(np.abs(fft_shift_scale))
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(scaled, caption=f"缩放 ({scale}x)", use_container_width=True, clamp=True)
            with col2:
                fig, ax = plt.subplots()
                ax.imshow(mag_scale, cmap='gray')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Spectrum (Scaled {scale}x)')
                st.pyplot(fig)
                plt.close(fig)
    
    # Inverse FFT demo
    st.subheader("傅里叶逆变换")
    if st.button("执行逆变换", key="ifft"):
        # Inverse FFT
        fft_ishift = np.fft.ifftshift(fft_shift)
        img_back = np.fft.ifft2(fft_ishift)
        img_back = np.abs(img_back)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="原图", use_container_width=True, clamp=True)
        with col2:
            st.image(img_back, caption="逆变换还原", use_container_width=True, clamp=True)
        st.success("逆变换完成！图像已还原")

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
    st.sidebar.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
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
