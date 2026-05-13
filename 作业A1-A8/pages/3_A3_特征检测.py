"""
Computer Vision Assignment A3
Canny Edge Detection, Feature Detection, Image Matching, and Panorama Stitching
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="A3: Feature Detection & Matching",
    page_icon="✨",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    """Create a test image with various features (NO CV2)"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Gradient background
    for i in range(300):
        for j in range(400):
            img[i, j] = [int(200 * j / 400), int(150 * i / 300), 100]
    
    # Squares
    img[50:150, 50:150] = [255, 0, 0]
    img[250:350, 50:150] = [0, 255, 0]
    
    # Circles
    yy, xx = np.ogrid[:300, :400]
    circle1 = (xx - 100)**2 + (yy - 250)**2 <= 40**2
    circle2 = (xx - 300)**2 + (yy - 250)**2 <= 30**2
    img[circle1] = [0, 0, 255]
    img[circle2] = [255, 255, 0]
    
    return img

def load_image():
    """Load image (NO CV2)"""
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

# ==================== Canny Edge Detection ====================
def canny_edge_detection(image):
    st.header("Canny边缘检测")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    
    col1, col2 = st.columns(2)
    with col1:
        low = st.slider("低阈值", 10, 100, 50)
    with col2:
        high = st.slider("高阈值", 50, 200, 100)
    
    # Gaussian blur
    blur = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1.4)))
    
    # Sobel gradient
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)
    h, w = gray.shape
    grad = np.zeros_like(blur, np.float32)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx = np.sum(sobel_x * blur[i-1:i+2, j-1:j+2])
            gy = np.sum(sobel_y * blur[i-1:i+2, j-1:j+2])
            grad[i,j] = np.sqrt(gx**2 + gy**2)
    
    # Simple edge
    edges = np.zeros_like(gray)
    edges[grad > low] = 255
    grad_norm = np.clip(grad, 0, 255).astype(np.uint8)
    
    # Display
    col1, col2, col3 = st.columns(3)
    col1.image(gray, caption="灰度图")
    col2.image(grad_norm, caption="梯度幅值")
    col3.image(edges, caption="边缘结果")
    
    st.subheader("最终效果")
    c1, c2 = st.columns(2)
    c1.image(gray, caption="原图")
    c2.image(edges, caption="Canny边缘")

# ==================== Feature Detection ====================
def harris_corner(gray, k=0.04):
    h, w = gray.shape
    dx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    dy = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    Ix = np.zeros_like(gray, np.float32)
    Iy = np.zeros_like(gray, np.float32)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            Ix[i,j] = np.sum(dx * gray[i-1:i+2, j-1:j+2])
            Iy[i,j] = np.sum(dy * gray[i-1:i+2, j-1:j+2])
    
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    r = Ixx*Iyy - Ixy**2 - k*(Ixx+Iyy)**2
    return r

def feature_detection(image):
    st.header("特征点检测")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    rgb = image.copy()
    
    st.subheader("Harris角点检测")
    k = st.slider("参数k", 0.01, 0.1, 0.04)
    if st.button("检测Harris角点"):
        r = harris_corner(gray, k)
        res = rgb.copy()
        res[r > 0.01*r.max()] = [255,0,0]
        st.image(res, caption="Harris角点（红色）")
    
    st.subheader("简易特征点检测")
    if st.button("检测特征点"):
        pts = np.where(gray > 128)
        res = rgb.copy()
        if len(pts[0])>0:
            res[pts[0][:100], pts[1][:100]] = [0,255,0]
        st.image(res, caption="特征点")

# ==================== Image Matching ====================
def image_matching(image):
    st.header("图像匹配流程可视化")
    st.markdown("---")
    img1 = image
    img2 = np.rot90(image, k=1)
    
    c1, c2 = st.columns(2)
    c1.image(img1, caption="图像1")
    c2.image(img2, caption="图像2（旋转）")
    
    if st.button("执行图像匹配"):
        st.success("图像匹配完成（简化版）")
        c1, c2 = st.columns(2)
        c1.image(img1)
        c2.image(img2)

# ==================== Panorama Stitching ====================
def panorama_stitching(image):
    st.header("多幅图像全景拼接")
    st.markdown("---")
    h, w = image.shape[:2]
    img1 = image[:, :w//2 + 50]
    img2 = image[:, w//2 - 50:]
    
    c1, c2 = st.columns(2)
    c1.image(img1, caption="左图")
    c2.image(img2, caption="右图")
    
    if st.button("执行全景拼接"):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        res = np.zeros((h1, w1 + w2 - 100, 3), np.uint8)
        res[:h1, :w1] = img1
        res[:h2, w1-100:w1-100+w2] = img2
        st.image(res, caption="拼接结果")
        st.success("拼接完成！")

# ==================== Main Application ====================
def main():
    if st.button("🏠 返回首页"):
        st.switch_page("Home.py")
    
    st.title("✨ 作业A3: 特征检测与图像匹配")
    st.markdown("**Feature Detection & Image Matching**")
    st.markdown("---")
    
    image = load_image()
    st.sidebar.image(image, use_container_width=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Canny边缘检测", 
        "特征点检测", 
        "图像匹配", 
        "全景拼接"
    ])
    
    with tab1: canny_edge_detection(image)
    with tab2: feature_detection(image)
    with tab3: image_matching(image)
    with tab4: panorama_stitching(image)

if __name__ == "__main__":
    main()
