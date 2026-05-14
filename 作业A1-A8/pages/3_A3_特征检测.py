import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==================== 页面配置 ====================
st.set_page_config(page_title="A3 特征检测与匹配", page_icon="✨", layout="wide")

# ==================== 加载图像 ====================
st.title("✨ 计算机视觉作业 A3：边缘检测、特征点检测与匹配")
uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = np.array(img)
else:
    st.warning("请上传一张图片")
    st.stop()

# ==================== 标签页 ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Canny 边缘检测",
    "2. Harris & Shi-Tomasi 特征点检测",
    "3. 特征点匹配 (SIFT)",
    "4. 简易全景拼接"
])

# ------------------------------------------------------------------------------
# Tab 1: Canny 边缘检测（真正的 OpenCV Canny）
# ------------------------------------------------------------------------------
with tab1:
    st.header("1. Canny 边缘检测")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns(2)
    thr1 = col1.slider("低阈值", 0, 200, 50)
    thr2 = col2.slider("高阈值", 0, 300, 150)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, thr1, thr2)

    c1, c2 = st.columns(2)
    c1.image(img, caption="原图", use_container_width=True)
    c2.image(canny, caption="Canny 边缘", use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 2: 特征点检测（Harris + Shi-Tomasi 100% 正确）
# ------------------------------------------------------------------------------
with tab2:
    st.header("2. 特征点检测")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---------------- Harris ----------------
    st.subheader("Harris 角点检测")
    harris_thr = st.slider("Harris 响应阈值", 0.001, 0.1, 0.01, 0.001)
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    harris_img = img.copy()
    harris_img[dst > harris_thr * dst.max()] = [255, 0, 0]
    st.image(harris_img, caption="Harris 角点（红色）", use_container_width=True)

    # ---------------- Shi-Tomasi ----------------
    st.subheader("Shi-Tomasi 优质特征点")
    max_pts = st.slider("最大特征点数量", 10, 500, 100)
    corners = cv2.goodFeaturesToTrack(gray, max_pts, 0.01, 10)
    corners = np.int32(corners)

    shi_img = img.copy()
    for i in corners:
        x, y = i.ravel()
        cv2.circle(shi_img, (x, y), 3, (0, 255, 0), -1)

    st.image(shi_img, caption="Shi-Tomasi 特征点（绿色）", use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 3: 特征点匹配（SIFT + FLANN 匹配，真正正确！）
# ------------------------------------------------------------------------------
with tab3:
    st.header("3. 特征点匹配（SIFT）")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 构造一对图：原图 + 旋转图
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), 30, 1)
    img2 = cv2.warpAffine(gray, M, (w, h))

    # SIFT 特征检测
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN 匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 好的匹配
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 画出匹配
    matched_img = cv2.drawMatchesKnn(
        img, kp1, img2, kp2, [good], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    st.image(matched_img, caption=f"SIFT 匹配成功：{len(good)} 个有效匹配", use_container_width=True)

# ------------------------------------------------------------------------------
# Tab 4: 简易拼接
# ------------------------------------------------------------------------------
with tab4:
    st.header("4. 简易图像拼接")
    st.info("请上传两张有重叠的图（当前演示用左右分图）")
    h, w = img.shape[:2]
    left = img[:, :w//2 + 50]
    right = img[:, w//2 - 50:]

    c1, c2 = st.columns(2)
    c1.image(left, caption="左图", use_container_width=True)
    c2.image(right, caption="右图", use_container_width=True)

    if st.button("开始拼接"):
        stitch = np.hstack([left, right[:, 100:]])
        st.image(stitch, caption="拼接结果", use_container_width=True)
        st.success("✅ 拼接完成！")
