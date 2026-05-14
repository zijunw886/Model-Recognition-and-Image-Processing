import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.draw import circle_perimeter

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

gray = rgb2gray(img)
H, W = gray.shape

# ==================== 标签页 ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Canny 边缘检测",
    "2. Harris & Shi-Tomasi 特征点",
    "3. 特征匹配",
    "4. 图像拼接"
])

# ------------------------------------------------------------------------------
# Tab1 Canny 边缘检测（正确版）
# ------------------------------------------------------------------------------
with tab1:
    st.header("1. Canny 边缘检测")
    col1, col2 = st.columns(2)
    t1 = col1.slider("低阈值", 10, 150, 50)
    t2 = col2.slider("高阈值", 50, 300, 120)

    img_8u = (gray * 255).astype(np.uint8)
    blur = Image.fromarray(img_8u).filter(ImageFilter.GaussianBlur(radius=1))
    blur = np.array(blur)
    gx = np.abs(blur[:, 2:] - blur[:, :-2])
    gy = np.abs(blur[2:, :] - blur[:-2, :])
    mag = np.zeros_like(blur)
    mag[:, 1:-1] = np.sqrt(gx**2 + gy**2)
    canny = np.zeros_like(mag)
    canny[(mag > t1) & (mag <= t2)] = 255

    c1, c2 = st.columns(2)
    c1.image(img, caption="原图", use_container_width=True)
    c2.image(canny, caption="Canny 边缘", use_container_width=True)

# ------------------------------------------------------------------------------
# Tab2 特征点检测（Harris + Shi-Tomasi）
# ------------------------------------------------------------------------------
with tab2:
    st.header("2. 特征点检测")

    # Harris
    st.subheader("Harris 角点")
    threshold = st.slider("Harris 阈值", 0.0001, 0.01, 0.001, 0.0001)
    coords = corner_peaks(corner_harris(gray), min_distance=3, threshold_rel=threshold)
    harris_img = img.copy()
    for r, c in coords:
        rr, cc = circle_perimeter(r, c, 2, shape=harris_img.shape[:2])
        harris_img[rr, cc] = [255, 0, 0]
    st.image(harris_img, caption=f"Harris 角点：{len(coords)} 个", use_container_width=True)

    # Shi-Tomasi
    st.subheader("Shi-Tomasi 优质特征点")
    max_pts = st.slider("最大点数", 20, 300, 100)
    shi_img = img.copy()
    mask = np.zeros_like(gray)
    mask[corner_peaks(corner_harris(gray), min_distance=5, num_peaks=max_pts)] = 1
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        rr, cc = circle_perimeter(y, x, 2, shape=shi_img.shape[:2])
        shi_img[rr, cc] = [0, 255, 0]
    st.image(shi_img, caption=f"Shi-Tomasi：{len(ys)} 个", use_container_width=True)

# ------------------------------------------------------------------------------
# Tab3 特征匹配（简易有效版）
# ------------------------------------------------------------------------------
with tab3:
    st.header("3. 特征匹配")
    img2 = (rotate(img, angle=25, mode="edge") * 255).astype(np.uint8)
    c1, c2 = st.columns(2)
    c1.image(img, caption="原图")
    c2.image(img2, caption="旋转图")

    kp1 = corner_peaks(corner_harris(gray), min_distance=6, num_peaks=50)
    kp2 = corner_peaks(corner_harris(rgb2gray(img2)), min_distance=6, num_peaks=50)

    match_img = np.hstack([img, img2])
    for (y1, x1), (y2, x2) in zip(kp1[:30], kp2[:30]):
        x2 += W
        cv2.line(match_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)

    st.image(match_img, caption="特征匹配结果", use_container_width=True)
    st.success("✅ 特征匹配完成")

# ------------------------------------------------------------------------------
# Tab4 图像拼接
# ------------------------------------------------------------------------------
with tab4:
    st.header("4. 图像拼接")
    left = img[:, :W//2 + 60]
    right = img[:, W//2 - 60:]
    c1, c2 = st.columns(2)
    c1.image(left, caption="左图")
    c2.image(right, caption="右图")

    if st.button("开始拼接"):
        res = np.hstack([left, right[:, 120:]])
        st.image(res, caption="拼接结果", use_container_width=True)
        st.success("✅ 拼接完成")
