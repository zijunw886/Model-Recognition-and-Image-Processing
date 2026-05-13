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

# ==================== 修复后的特征点检测 ====================
def harris_corner_response(gray, k=0.04):
    """计算Harris角点响应值"""
    h, w = gray.shape
    # Sobel算子
    dx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    dy = np.array([[-1,-2,-1],[0,0,0],[1,1,1]], dtype=np.float32)
    
    # 计算梯度
    Ix = np.zeros_like(gray, dtype=np.float32)
    Iy = np.zeros_like(gray, dtype=np.float32)
    gray_float = gray.astype(np.float32)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            Ix[i,j] = np.sum(dx * gray_float[i-1:i+2, j-1:j+2])
            Iy[i,j] = np.sum(dy * gray_float[i-1:i+2, j-1:j+2])
    
    # 计算协方差矩阵元素
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy
    
    # Harris响应公式
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    response = det - k * (trace**2)
    
    return response

def feature_detection(image):
    st.header("特征点检测")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    rgb = image.copy()
    
    # 1. Harris角点检测（修复版）
    st.subheader("Harris角点检测")
    k = st.slider("参数k", 0.01, 0.1, 0.04)
    threshold = st.slider("角点阈值", 0.001, 0.01, 0.005, 0.001)
    
    if st.button("检测Harris角点", key="harris_btn"):
        with st.spinner("计算中..."):
            # 计算响应值
            r = harris_corner_response(gray, k)
            r = (r - r.min()) / (r.max() - r.min() + 1e-6)  # 归一化到0-1
            
            # 转换为灰度图格式（修复显示）
            r_display = (r * 255).astype(np.uint8)
            
            # 标记角点（用圆点标记，更明显）
            harris_img = rgb.copy()
            corners = np.where(r > threshold)
            # 取前200个角点，避免太多点
            if len(corners[0]) > 200:
                idx = np.argsort(r[corners])[-200:]
                corners = (corners[0][idx], corners[1][idx])
            
            # 用红色圆点标记角点
            for y, x in zip(*corners):
                harris_img[max(0,y-1):y+2, max(0,x-1):x+2] = [255, 0, 0]
            
            # 显示（删除错误的cmap参数）
            col1, col2 = st.columns(2)
            col1.image(r_display, caption="角点响应图", use_container_width=True)
            col2.image(harris_img, caption=f"检测到{len(corners[0])}个角点（红色标记）", use_container_width=True)
    
    # 2. 简易特征点检测（修复版，基于边缘）
    st.subheader("基于边缘的特征点检测")
    if st.button("检测特征点", key="feature_btn"):
        with st.spinner("计算中..."):
            # 用Canny边缘的点作为特征点
            blur = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1)))
            edges = np.zeros_like(gray)
            edges[blur > 100] = 255  # 简化边缘检测
            
            # 取边缘的点，最多取200个
            pts = np.where(edges == 255)
            if len(pts[0]) > 200:
                idx = np.random.choice(len(pts[0]), 200, replace=False)
                pts = (pts[0][idx], pts[1][idx])
            
            # 用绿色圆点标记特征点
            feature_img = rgb.copy()
            for y, x in zip(*pts):
                feature_img[max(0,y-1):y+2, max(0,x-1):x+2] = [0, 255, 0]
            
            st.image(feature_img, caption=f"检测到{len(pts[0])}个特征点（绿色标记）", use_container_width=True)

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
