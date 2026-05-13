"""
Computer Vision Assignment A3
Canny Edge Detection, Feature Detection, Image Matching, and Panorama Stitching
"""

import streamlit as st
import numpy as np
import cv2
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
    """Create a test image with various features"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(300):
        for j in range(400):
            img[i, j] = [int(200 * j / 400), int(150 * i / 300), 100]
    
    # Add squares and circles
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 2)
    cv2.rectangle(img, (250, 50), (350, 150), (0, 255, 0), 2)
    cv2.circle(img, (100, 250), 40, (0, 0, 255), 2)
    cv2.circle(img, (300, 250), 30, (255, 255, 0), 2)
    
    # Add lines
    cv2.line(img, (50, 100), (350, 100), (255, 0, 255), 2)
    cv2.line(img, (200, 50), (200, 250), (0, 255, 255), 2)
    
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

# ==================== Canny Edge Detection ====================
def canny_edge_detection(image):
    """Complete Canny edge detection with step-by-step visualization"""
    st.header("Canny边缘检测")
    st.markdown("---")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        low_threshold = st.slider("低阈值", 10, 100, 50, key="canny_low")
    with col2:
        high_threshold = st.slider("高阈值", 50, 200, 100, key="canny_high")
    
    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Gradient calculation
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    # Non-maximum suppression (simplified)
    nms = cv2.convertScaleAbs(gradient_magnitude)
    
    # Canny edge detection
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    
    # Display results
    st.subheader("处理步骤")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(gray, caption="灰度图", use_container_width=True, clamp=True)
    with col2:
        st.image(cv2.convertScaleAbs(gradient_magnitude), caption="梯度幅值", use_container_width=True, clamp=True)
    with col3:
        st.image(nms, caption="非极大值抑制", use_container_width=True, clamp=True)
    
    st.subheader("最终结果")
    col1, col2 = st.columns(2)
    with col1:
        st.image(gray, caption="原图", use_container_width=True, clamp=True)
    with col2:
        st.image(edges, caption=f"Canny边缘检测 (低阈值={low_threshold}, 高阈值={high_threshold})", use_container_width=True, clamp=True)

# ==================== Feature Detection ====================
def feature_detection(image):
    """Harris corner detection and SIFT feature detection"""
    st.header("特征点检测")
    st.markdown("---")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Harris corner detection
    st.subheader("Harris角点检测")
    block_size = st.slider("邻域大小", 2, 10, 2, key="harris_block")
    ksize = st.slider("Sobel算子大小", 3, 7, 3, step=2, key="harris_ksize")
    k = st.slider("参数k", 0.01, 0.1, 0.04, key="harris_k")
    
    if st.button("检测Harris角点", key="harris_btn"):
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        
        # Mark corners
        harris_img = rgb.copy()
        harris_img[dst > 0.01 * dst.max()] = [255, 0, 0]
        
        st.image(harris_img, caption="Harris角点检测结果（红色标记）", use_container_width=True)
    
    # SIFT feature detection
    st.subheader("SIFT特征点检测")
    n_features = st.slider("特征点数量", 50, 500, 200, key="sift_n")
    
    if st.button("检测SIFT特征点", key="sift_btn"):
        sift = cv2.SIFT_create(nfeatures=n_features)
        keypoints, _ = sift.detectAndCompute(gray, None)
        
        sift_img = cv2.drawKeypoints(rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.image(sift_img, caption=f"SIFT特征点检测结果（{len(keypoints)}个特征点）", use_container_width=True)

# ==================== Image Matching ====================
def image_matching(image):
    """Complete image matching pipeline"""
    st.header("图像匹配流程可视化")
    st.markdown("---")
    
    # Create a transformed version for matching
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
    image2 = cv2.warpAffine(image, M, (cols, rows))
    
    st.subheader("待匹配图像")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="图像1", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), caption="图像2（旋转30度）", use_container_width=True)
    
    if st.button("执行图像匹配", key="match_btn"):
        # SIFT features
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        # Draw initial matches
        match_img = cv2.drawMatches(image, kp1, image2, kp2, good[:20], None, flags=2)
        
        st.subheader("初始匹配结果")
        st.image(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # RANSAC
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            
            # Draw inliers
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, 
                              matchesMask=matches_mask, flags=2)
            ransac_img = cv2.drawMatches(image, kp1, image2, kp2, good, None, **draw_params)
            
            st.subheader("RANSAC优化后匹配结果")
            st.image(cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB), caption="绿色为内点匹配", use_container_width=True)
            
            st.success(f"匹配点总数: {len(good)}, RANSAC内点数: {sum(matches_mask)}")

# ==================== Panorama Stitching ====================
def panorama_stitching(image):
    """Multi-image panorama stitching"""
    st.header("多幅图像全景拼接")
    st.markdown("---")
    
    # Create multiple overlapping images
    rows, cols = image.shape[:2]
    
    # Create shifted versions
    img1 = image[:, :cols//2 + 50]
    img2 = image[:, cols//2 - 50:]
    
    st.subheader("待拼接图像")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="图像1", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption="图像2", use_container_width=True)
    
    if st.button("执行全景拼接", key="stitch_btn"):
        # Stitch using OpenCV
        try:
            stitcher = cv2.Stitcher_create()
            status, stitched = stitcher.stitch([img1, img2])
            
            if status == cv2.Stitcher_OK:
                st.subheader("拼接结果")
                st.image(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB), caption="全景拼接结果", use_container_width=True)
            else:
                # Manual stitching as fallback
                st.warning("自动拼接失败，使用手动拼接")
                
                # Find features and match
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
                kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
                
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                # Warp image2
                result = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
                result[0:img1.shape[0], 0:img1.shape[1]] = img1
                
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="手动拼接结果", use_container_width=True)
        
        except Exception as e:
            st.error(f"拼接失败: {str(e)}")

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("✨ 作业A3: 特征检测与图像匹配")
    st.markdown("**Feature Detection & Image Matching**")
    st.markdown("---")
    
    # Load image
    image = load_image()
    
    # Sidebar preview
    st.sidebar.header("图像预览")
    st.sidebar.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs([
        "Canny边缘检测", 
        "特征点检测", 
        "图像匹配", 
        "全景拼接"
    ])
    
    with tab1:
        canny_edge_detection(image)
    
    with tab2:
        feature_detection(image)
    
    with tab3:
        image_matching(image)
    
    with tab4:
        panorama_stitching(image)

if __name__ == "__main__":
    main()
