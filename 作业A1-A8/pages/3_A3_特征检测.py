"""
Computer Vision Assignment A3
Canny Edge Detection, Feature Detection & Matching
【插画/平涂风格专属版】- 专门适配你的猫咪图
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import os

st.set_page_config(page_title="A3: 特征检测", page_icon="✨", layout="wide")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 内置测试图
def create_test_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[50:150, 50:150] = [255,0,0]
    img[250:350, 50:150] = [0,255,0]
    return img

# 加载图片
def load_image():
    try:
        return np.array(Image.open(os.path.join(ROOT_DIR, 'pic.jpg')).convert("RGB"))
    except:
        return create_test_image()

# ==================== 【简化版】Canny边缘检测 ====================
def canny_edge_detection(image):
    st.header("Canny边缘检测")
    gray = np.mean(image, axis=2).astype(np.uint8)
    low = st.slider("低阈值", 10, 100, 50)
    if st.button("应用Canny边缘检测"):
        blur = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1)))
        edges = np.zeros_like(gray)
        edges[blur > low] = 255
        col1, col2 = st.columns(2)
        col1.image(gray, caption="原图")
        col2.image(edges, caption="边缘结果")

# ==================== 【平涂专属】角点检测 ====================
def feature_detection(image):
    st.header("特征点检测（插画/平涂专属版）")
    gray = np.mean(image, axis=2).astype(np.uint8)
    rgb = image.copy()

    st.subheader("简易角点检测（猫咪图专用）")
    threshold = st.slider("检测灵敏度", 0.1, 1.0, 0.5)
    if st.button("检测角点"):
        # 直接用水平+垂直梯度差找“伪角点”
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        # 两个方向梯度都大的点，就是角点
        corners = np.where((grad_x > grad_x.max() * threshold) & (grad_y > grad_y.max() * threshold))
        
        if len(corners[0]) > 0:
            res = rgb.copy()
            # 只取前100个点，避免画面太乱
            idx = np.random.choice(len(corners[0]), min(100, len(corners[0])), replace=False)
            y = corners[0][idx]
            x = corners[1][idx]
            
            # 用红色圆点标记
            for yi, xi in zip(y, x):
                res[max(0, yi-1):yi+2, max(0, xi-1):xi+2] = [255, 0, 0]
            
            st.image(res, caption=f"✅ 检测到 {len(idx)} 个角点（红色标记）")
        else:
            st.warning("未检测到角点，请提高检测灵敏度")

    st.subheader("边缘特征点检测")
    n = st.slider("最大特征点数量", 20, 200, 50)
    if st.button("检测特征点"):
        # 用边缘作为特征点
        blur = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1)))
        edges = np.where(blur > 100)
        idx = np.random.choice(len(edges[0]), min(n, len(edges[0])), replace=False)
        res = rgb.copy()
        res[edges[0][idx], edges[1][idx]] = [0, 255, 0]
        st.image(res, caption=f"✅ 检测到 {len(idx)} 个特征点（绿色标记）")

# ==================== 主函数 ====================
def main():
    if st.button("🏠 返回首页"):
        st.switch_page("Home.py")
    st.title("✨ 作业A3: 特征检测与图像匹配")
    image = load_image()
    st.sidebar.image(image, use_container_width=True)

    tab1, tab2 = st.tabs(["Canny边缘检测", "特征点检测"])
    with tab1: canny_edge_detection(image)
    with tab2: feature_detection(image)

if __name__ == "__main__":
    main()
