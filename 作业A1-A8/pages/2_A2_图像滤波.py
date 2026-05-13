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
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        for j in range(400):
            img[i, j] = [int(255 * j / 400), int(255 * i / 300), 128]
    img[50:150, 50:150] = [255, 0, 0]
    yy, xx = np.ogrid[:300, :400]
    mask = (xx - 300)**2 + (yy - 150)**2 <= 40**2
    img[mask] = [0, 255, 0]
    return img

def load_image():
    img_path = os.path.join(ROOT_DIR, 'pic.jpg')
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            return np.array(img)
        except:
            return create_test_image()
    else:
        return create_test_image()

# ==================== Spatial Domain Filtering ====================
def spatial_filtering(image):
    st.header("空间域滤波")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    filter_type = st.selectbox("选择滤波类型", ["均值滤波", "高斯滤波", "中值滤波", "Sobel边缘检测"])

    if filter_type == "均值滤波":
        k = st.slider("卷积核大小", 3, 15, 5, step=2)
        if st.button("应用均值滤波"):
            res = Image.fromarray(gray).filter(ImageFilter.BoxBlur(k//2))
            show_comparison(gray, np.array(res), "原图", f"均值滤波 {k}x{k}")

    elif filter_type == "高斯滤波":
        k = st.slider("卷积核大小", 3, 15, 5, step=2)
        sigma = st.slider("高斯标准差", 0.1, 5.0, 1.0)
        if st.button("应用高斯滤波"):
            res = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=sigma))
            show_comparison(gray, np.array(res), "原图", f"高斯滤波")

    elif filter_type == "中值滤波":
        k = st.slider("卷积核大小", 3, 15, 5, step=2)
        if st.button("应用中值滤波"):
            res = Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=k))
            show_comparison(gray, np.array(res), "原图", f"中值滤波 {k}x{k}")

    elif filter_type == "Sobel边缘检测":
        direction = st.selectbox("边缘方向", ["水平边缘", "垂直边缘", "双向边缘"])
        if st.button("应用Sobel边缘检测"):
            gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            h, w = gray.shape
            out = np.zeros_like(gray, dtype=np.float32)
            for i in range(1,h-1):
                for j in range(1,w-1):
                    s1 = np.sum(gx * gray[i-1:i+2, j-1:j+2])
                    s2 = np.sum(gy * gray[i-1:i+2, j-1:j+2])
                    out[i,j] = np.sqrt(s1**2 + s2**2)
            out = np.clip(out, 0, 255).astype(np.uint8)
            show_comparison(gray, out, "原图", "Sobel边缘")

def show_comparison(original, processed, title1, title2):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption=title1, use_container_width=True)
    with col2:
        st.image(processed, caption=title2, use_container_width=True)

# ==================== Gradient Analysis ====================
def gradient_analysis(image):
    st.header("图像梯度与局部区域分析")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    h, w = gray.shape
    x1, y1, x2, y2 = 50, 50, 150, 150
    if st.button("分析梯度"):
        roi = gray[y1:y2, x1:x2]
        sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        mag = np.zeros_like(roi, dtype=np.float32)
        for i in range(1,roi.shape[0]-1):
            for j in range(1,roi.shape[1]-1):
                s1 = np.sum(sobel_x * roi[i-1:i+2, j-1:j+2])
                s2 = np.sum(sobel_y * roi[i-1:i+2, j-1:j+2])
                mag[i,j] = np.sqrt(s1**2 + s2**2)
        mag = np.clip(mag,0,255)
        col1, col2 = st.columns(2)
        with col1:
            st.image(roi, caption="区域", use_container_width=True)
        with col2:
            st.image(mag.astype(np.uint8), caption="梯度幅值", use_container_width=True)

# ==================== Frequency Domain Filtering ====================
def frequency_domain_filtering(image):
    st.header("频域滤波")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    fft = np.fft.fft2(gray)
    shift = np.fft.fftshift(fft)
    mag = 20*np.log(np.abs(shift)+1e-6)
    col1, col2 = st.columns(2)
    with col1:
        st.image(gray, use_container_width=True)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(mag, cmap='gray')
        st.pyplot(fig)
        plt.close(fig)

# ==================== Main ====================
def main():
    if st.button("🏠 返回首页"):
        st.switch_page("Home.py")
    st.title("🔍 作业A2: 空间域滤波与频域分析")
    image = load_image()
    st.sidebar.image(image, use_container_width=True)
    t1, t2, t3 = st.tabs(["空间域滤波", "梯度分析", "频域滤波"])
    with t1: spatial_filtering(image)
    with t2: gradient_analysis(image)
    with t3: frequency_domain_filtering(image)

if __name__ == "__main__":
    main()
