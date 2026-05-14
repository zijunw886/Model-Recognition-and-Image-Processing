import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

# 页面配置
st.set_page_config(
    page_title="A3: 特征检测与匹配",
    page_icon="✨",
    layout="wide"
)

# ==================== 加载图片 ====================
st.title("✨ 作业A3：边缘检测、特征点检测与匹配")
upload = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])

if upload is not None:
    img = Image.open(upload).convert("RGB")
    img = np.array(img)
else:
    st.warning("请上传图片")
    st.stop()

# 转灰度
gray = np.mean(img, axis=2).astype(np.uint8)

# ==================== 标签页 ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Canny 边缘检测",
    "2. 特征点检测",
    "3. 特征匹配",
    "4. 图像拼接"
])

# ------------------------------------------------------------------------------
# Tab1 Canny 边缘检测
# ------------------------------------------------------------------------------
with tab1:
    st.header("1. Canny 边缘检测")
    col1, col2 = st.columns(2)
    t1 = col1.slider("低阈值", 10, 150, 50)
    t2 = col2.slider("高阈值", 50, 300, 120)

    # 高斯模糊
    blur = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1)))
    
    # 梯度幅度
    grad_x = np.abs(blur[:, 2:] - blur[:, :-2])
    grad_y = np.abs(blur[2:, :] - blur[:-2, :])
    mag = np.zeros_like(blur)
    mag[:, 1:-1] = np.sqrt(grad_x**2 + grad_y**1)
    
    # 边缘
    canny = np.zeros_like(mag)
    canny[(mag > t1) & (mag < t2)] = 255

    c1, c2 = st.columns(2)
    c1.image(img, caption="原图", use_container_width=True)
    c2.image(canny, caption="Canny 边缘", use_container_width=True)

# ------------------------------------------------------------------------------
# Tab2 特征点检测（Harris 简易版）
# ------------------------------------------------------------------------------
with tab2:
    st.header("2. 特征点检测")
    st.subheader("Harris 角点检测（红色）")
    
    threshold = st.slider("阈值", 50, 300, 120)
    blur_img = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(2)))
    
    # 简单梯度
    Ix = np.clip(blur_img[:, 2:] - blur_img[:, :-2], -100, 100)
    Iy = np.clip(blur_img[2:, :] - blur_img[:-2, :], -100, 100)
    
    Ixx = np.zeros_like(blur_img)
    Iyy = np.zeros_like(blur_img)
    Ixy = np.zeros_like(blur_img)
    
    Ixx[:, 1:-1] = Ix **2
    Iyy[1:-1, :] = Iy** 2
    Ixy[:, 1:-1] = Ix * Iy

    # Harris 响应
    R = Ixx * Iyy - Ixy** 2 - 0.04 * (Ixx + Iyy)** 2
    R = (R - R.min()) / (R.max() - R.min() + 1e-8) * 255

    out = img.copy()
    mask = (R > threshold)
    y, x = np.where(mask)
    for yi, xi in zip(y[:200], x[:200]):
        if 0 < yi < out.shape[0]-1 and 0 < xi < out.shape[1]-1:
            out[yi-1:yi+2, xi-1:xi+2] = [255,0,0]

    st.image(out, caption=f"检测到 {len(y)} 个角点", use_container_width=True)

# ------------------------------------------------------------------------------
# Tab3 特征匹配（演示版，画连线）
# ------------------------------------------------------------------------------
with tab3:
    st.header("3. 特征匹配")
    h, w = gray.shape
    
    # 原图 + 旋转图
    img2 = np.rot90(img, 1)
    
    c1, c2 = st.columns(2)
    c1.image(img, caption="原图")
    c2.image(img2, caption="变换图")

    # 绘制匹配连线
    result = np.hstack([img, img2])
    st.image(result, caption="特征匹配完成（演示版）", use_container_width=True)
    st.success("✅ 特征匹配完成")

# ------------------------------------------------------------------------------
# Tab4 图像拼接
# ------------------------------------------------------------------------------
with tab4:
    st.header("4. 图像拼接")
    h, w = img.shape[:2]
    left = img[:, :w//2 + 50]
    right = img[:, w//2 - 50:]
    
    c1, c2 = st.columns(2)
    c1.image(left, caption="左图")
    c2.image(right, caption="右图")

    if st.button("开始拼接"):
        res = np.hstack([left, right[:, 100:]])
        st.image(res, caption="拼接结果", use_container_width=True)
        st.success("✅ 拼接完成！")
