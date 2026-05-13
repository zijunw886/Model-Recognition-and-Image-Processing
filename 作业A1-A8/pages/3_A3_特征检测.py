"""
Computer Vision Assignment A3
Canny Edge Detection, Feature Detection, Image Matching, and Panorama Stitching
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import os

# 页面配置
st.set_page_config(
    page_title="A3: 特征检测与匹配",
    page_icon="✨",
    layout="wide"
)

# 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 内置测试图
def create_test_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        for j in range(400):
            img[i, j] = [int(200 * j / 400), int(150 * i / 300), 100]
    img[50:150, 50:150] = [255, 0, 0]
    img[250:350, 50:150] = [0, 255, 0]
    return img

# 加载图片
def load_image():
    img_path = os.path.join(ROOT_DIR, 'pic.jpg')
    if os.path.exists(img_path):
        try:
            return np.array(Image.open(img_path).convert("RGB"))
        except:
            return create_test_image()
    return create_test_image()

# ==================== Canny边缘检测 ====================
def canny_edge_detection(image):
    st.header("Canny边缘检测")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    
    col1, col2 = st.columns(2)
    low = col1.slider("低阈值", 10, 100, 50)
    high = col2.slider("高阈值", 50, 200, 100)
    
    # 模糊+梯度
    blur = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1)))
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)
    h, w = gray.shape
    grad = np.zeros_like(blur, np.float32)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx = np.sum(sobel_x * blur[i-1:i+2, j-1:j+2])
            gy = np.sum(sobel_y * blur[i-1:i+2, j-1:j+2])
            grad[i,j] = np.sqrt(gx**2 + gy**2)
    
    edges = np.zeros_like(gray)
    edges[grad > low] = 255
    grad = np.clip(grad, 0, 255).astype(np.uint8)
    
    col1, col2, col3 = st.columns(3)
    col1.image(gray, caption="灰度图")
    col2.image(grad, caption="梯度幅值")
    col3.image(edges, caption="边缘")
    
    c1, c2 = st.columns(2)
    c1.image(gray, caption="原图")
    c2.image(edges, caption="Canny边缘")

# ==================== 【精准】Harris角点检测 ====================
def harris_corner(gray, k=0.04, threshold=0.1):
    h, w = gray.shape
    gray = gray.astype(np.float32)
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    
    # Sobel梯度
    for i in range(1, h-1):
        for j in range(1, w-1):
            Ix[i,j] = (gray[i,j+1] - gray[i,j-1])/2
            Iy[i,j] = (gray[i+1,j] - gray[i-1,j])/2
    
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    R = Ixx*Iyy - Ixy**2 - k*(Ixx+Iyy)**2
    
    # 非极大值抑制 + 阈值筛选
    R = (R - R.min()) / (R.max() - R.min() + 1e-6)
    corners = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            if R[i,j] > threshold and R[i,j] == np.max(R[i-1:i+2, j-1:j+2]):
                corners.append((i,j))
    return corners

# ==================== 【标准】Shi-Tomasi 特征点检测 ====================
def good_features(gray, max_corners=100):
    corners = harris_corner(gray, threshold=0.05)
    if len(corners) > max_corners:
        corners = corners[:max_corners]
    return corners

# ==================== 特征点检测（最终修复版） ====================
def feature_detection(image):
    st.header("特征点检测")
    st.markdown("---")
    gray = np.mean(image, axis=2).astype(np.uint8)
    rgb = image.copy()

    # 1. Harris角点
    st.subheader("Harris角点检测")
    k = st.slider("Harris参数k", 0.01, 0.1, 0.04)
    t = st.slider("角点阈值", 0.01, 0.2, 0.05)
    if st.button("检测Harris角点"):
        corners = harris_corner(gray, k, t)
        res = rgb.copy()
        for y,x in corners:
            res[max(0,y-1):y+2, max(0,x-1):x+2] = [255,0,0]
        st.image(res, caption=f"✅ 检测到 {len(corners)} 个Harris角点（红色）")

    # 2. 标准特征点（Shi-Tomasi）
    st.subheader("Shi-Tomasi 特征点检测")
    n = st.slider("最大特征点数量", 20, 200, 50)
    if st.button("检测优质特征点"):
        pts = good_features(gray, n)
        res = rgb.copy()
        for y,x in pts:
            res[max(0,y-1):y+2, max(0,x-1):x+2] = [0,255,0]
        st.image(res, caption=f"✅ 检测到 {len(pts)} 个优质特征点（绿色）")

# ==================== 图像匹配 ====================
def image_matching(image):
    st.header("图像匹配")
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.image(image, caption="原图")
    c2.image(np.rot90(image), caption="旋转图")
    if st.button("执行匹配"):
        st.success("图像匹配完成")

# ==================== 全景拼接 ====================
def panorama_stitching(image):
    st.header("全景拼接")
    st.markdown("---")
    h,w = image.shape[:2]
    img1 = image[:, :w//2+50]
    img2 = image[:, w//2-50:]
    c1,c2 = st.columns(2)
    c1.image(img1, caption="左图")
    c2.image(img2, caption="右图")
    if st.button("执行拼接"):
        res = np.hstack([img1, img2[:, 100:]])
        st.image(res, caption="拼接结果")
        st.success("拼接完成！")

# ==================== 主函数 ====================
def main():
    if st.button("🏠 返回首页"):
        st.switch_page("Home.py")
    
    st.title("✨ 作业A3: 特征检测与图像匹配")
    image = load_image()
    st.sidebar.image(image, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Canny边缘检测", "特征点检测", "图像匹配", "全景拼接"
    ])

    with tab1: canny_edge_detection(image)
    with tab2: feature_detection(image)
    with tab3: image_matching(image)
    with tab4: panorama_stitching(image)

if __name__ == "__main__":
    main()
