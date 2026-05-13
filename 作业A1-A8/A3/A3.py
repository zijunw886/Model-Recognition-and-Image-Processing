"""
Computer Vision Assignment A3
Edge Detection, Feature Detection, Image Matching, and Panorama Stitching
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="A3: Edge Detection & Feature Matching",
    page_icon="🎯",
    layout="wide"
)

# ==================== Image Loading ====================
def load_image():
    """Load image from pic.jpg or create test image if not found"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'pic.jpg')
    
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

def create_test_image():
    """Create a test image with various features"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Gradient background
    for i in range(400):
        for j in range(600):
            img[i, j] = [int(255 * j / 600), int(255 * i / 400), 128]
    
    # Add geometric shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
    cv2.circle(img, (450, 150), 60, (0, 255, 0), -1)
    cv2.line(img, (50, 350), (550, 50), (0, 0, 255), 5)
    cv2.putText(img, "CV A3", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Add texture
    for i in range(300, 400):
        for j in range(400, 600):
            if (i + j) % 10 < 5:
                img[i, j] = [255, 255, 0]
    
    return img

# ==================== Canny Edge Detection ====================
def canny_edge_detection(image):
    """Complete Canny edge detection with step-by-step visualization"""
    st.header("Canny边缘检测")
    st.markdown("---")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        kernel_size = st.selectbox("高斯核大小", [3, 5, 7], index=1)
    with col2:
        low_threshold = st.slider("低阈值", 0, 255, 50)
    with col3:
        high_threshold = st.slider("高阈值", 0, 255, 150)
    
    if st.button("执行Canny边缘检测"):
        # Step 1: Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Step 2: Gradient calculation
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        gradient_direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        
        # Step 3: Non-maximum suppression
        gradient_magnitude_uint8 = np.uint8(gradient_magnitude)
        non_max = cv2.Canny(gradient_magnitude_uint8, low_threshold, high_threshold, L2gradient=True)
        
        # Step 4: Double thresholding
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=kernel_size)
        
        # Display results
        st.subheader("步骤可视化")
        
        # Step 1: Original and Grayscale
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True, clamp=True)
        with col2:
            st.image(gray, caption="灰度图", use_container_width=True, clamp=True)
        
        # Step 2: Gaussian Blur
        col1, col2 = st.columns(2)
        with col1:
            st.image(blurred, caption="高斯滤波后", use_container_width=True, clamp=True)
        with col2:
            fig, ax = plt.subplots()
            ax.imshow(gradient_magnitude, cmap='gray')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Gradient Magnitude')
            st.pyplot(fig)
            plt.close(fig)
        
        # Step 3: Non-maximum suppression
        col1, col2 = st.columns(2)
        with col1:
            st.image(gradient_magnitude_uint8, caption="非极大值抑制前", use_container_width=True, clamp=True)
        with col2:
            st.image(non_max, caption="非极大值抑制后", use_container_width=True, clamp=True)
        
        # Step 4: Final Canny edges
        st.subheader("最终Canny边缘检测结果")
        fig, ax = plt.subplots()
        ax.imshow(edges, cmap='gray')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Canny Edge Detection')
        st.pyplot(fig)
        plt.close(fig)

# ==================== Feature Detection ====================
def feature_detection(image):
    """Harris corner detection and SIFT feature detection"""
    st.header("特征点检测")
    st.markdown("---")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Algorithm selection
    algorithm = st.selectbox("选择检测算法", ["Harris角点检测", "SIFT特征点检测"])
    
    if algorithm == "Harris角点检测":
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            block_size = st.slider("邻域大小", 2, 10, 3)
        with col2:
            ksize = st.slider("Sobel算子大小", 3, 7, 3)
        with col3:
            k = st.slider("自由参数k", 0.01, 0.1, 0.04)
        
        if st.button("检测Harris角点"):
            # Harris corner detection
            corners = cv2.cornerHarris(gray, block_size, ksize, k)
            
            # Threshold and visualize
            corners_norm = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX)
            corners_norm = np.uint8(corners_norm)
            
            # Find corners above threshold
            threshold = st.slider("角点阈值", 0, 255, 100)
            _, corners_thresh = cv2.threshold(corners_norm, threshold, 255, cv2.THRESH_BINARY)
            
            # Draw corners on original image
            result = image.copy()
            corners_coords = np.where(corners_thresh > 0)
            for y, x in zip(corners_coords[0], corners_coords[1]):
                cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True, clamp=True)
            with col2:
                fig, ax = plt.subplots()
                ax.imshow(corners_norm, cmap='hot')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Harris Corner Response')
                st.pyplot(fig)
                plt.close(fig)
            with col3:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"检测到{len(corners_coords[0])}个角点", use_container_width=True, clamp=True)
    
    elif algorithm == "SIFT特征点检测":
        # Parameters
        n_features = st.slider("特征点数量", 0, 5000, 0)
        contrast_threshold = st.slider("对比度阈值", 0.01, 0.1, 0.04)
        
        if st.button("检测SIFT特征点"):
            # SIFT detection
            sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            # Draw keypoints
            result = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True, clamp=True)
            with col2:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"检测到{len(keypoints)}个SIFT特征点", use_container_width=True, clamp=True)
            
            st.info(f"特征描述子维度: {descriptors.shape[1] if descriptors is not None else 'N/A'}")

# ==================== Image Matching ====================
def image_matching(image):
    """Complete image matching pipeline with step-by-step visualization"""
    st.header("图像匹配流程可视化")
    st.markdown("---")
    
    # Load second image
    st.subheader("加载第二张图像")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="图像1（参考图）", use_container_width=True, clamp=True)
    
    with col2:
        uploaded_file = st.file_uploader("上传第二张图像", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image2 = np.array(Image.open(uploaded_file))
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        else:
            # Create a transformed version of image1 as image2
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1.0)
            image2 = cv2.warpAffine(image, M, (cols, rows))
            st.info("使用内置变换图像作为图像2")
        
        st.image(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), caption="图像2（待匹配）", use_container_width=True, clamp=True)
    
    # Matching parameters
    st.subheader("匹配参数设置")
    col1, col2 = st.columns(2)
    with col1:
        ratio_threshold = st.slider("比率阈值", 0.5, 1.0, 0.75)
    with col2:
        ransac_threshold = st.slider("RANSAC阈值", 1.0, 10.0, 5.0)
    
    if st.button("执行图像匹配"):
        # Convert to grayscale
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Feature detection
        st.subheader("步骤1: 特征点检测")
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        st.info(f"图像1检测到 {len(kp1)} 个特征点，图像2检测到 {len(kp2)} 个特征点")
        
        # Draw keypoints
        img_kp1 = cv2.drawKeypoints(image, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_kp2 = cv2.drawKeypoints(image2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_kp1, cv2.COLOR_BGR2RGB), caption="图像1特征点", use_container_width=True, clamp=True)
        with col2:
            st.image(cv2.cvtColor(img_kp2, cv2.COLOR_BGR2RGB), caption="图像2特征点", use_container_width=True, clamp=True)
        
        # Step 2: Initial matching
        st.subheader("步骤2: 初始特征匹配")
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        
        st.info(f"初始匹配: {len(matches)} 对，比率测试后: {len(good_matches)} 对")
        
        # Draw initial matches
        img_initial = cv2.drawMatches(image, kp1, image2, kp2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        st.image(cv2.cvtColor(img_initial, cv2.COLOR_BGR2RGB), caption="初始匹配结果（前50对）", use_container_width=True, clamp=True)
        
        # Step 3: RANSAC
        st.subheader("步骤3: RANSAC剔除误匹配")
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            matches_mask = mask.ravel().tolist()
            
            inliers = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
            st.info(f"RANSAC后保留: {len(inliers)} 对内点")
            
            # Draw inliers
            img_ransac = cv2.drawMatches(image, kp1, image2, kp2, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            st.image(cv2.cvtColor(img_ransac, cv2.COLOR_BGR2RGB), caption="RANSAC优化后的匹配结果", use_container_width=True, clamp=True)
            
            # Step 4: Image alignment
            st.subheader("步骤4: 图像对齐")
            if M is not None:
                rows, cols = image.shape[:2]
                aligned = cv2.warpPerspective(image2, M, (cols, rows))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="图像1（参考）", use_container_width=True, clamp=True)
                with col2:
                    st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), caption="图像2（对齐后）", use_container_width=True, clamp=True)
                
                st.success("图像匹配完成！单应性矩阵已计算")
            else:
                st.error("无法计算单应性矩阵")
        else:
            st.error("匹配点数量不足，无法进行RANSAC")

# ==================== Panorama Stitching ====================
def panorama_stitching(image):
    """Multi-image panorama stitching with different blending methods"""
    st.header("多幅图像全景拼接")
    st.markdown("---")
    
    # Load multiple images
    st.subheader("加载图像")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="图像1", use_container_width=True, clamp=True)
    
    with col2:
        uploaded_files = st.file_uploader("上传更多图像（至少2张）", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        images = [image]
        if uploaded_files:
            for file in uploaded_files:
                img = np.array(Image.open(file))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)
        else:
            # Create overlapping images by shifting (better for panorama)
            rows, cols = image.shape[:2]
            shift1 = int(cols * 0.3)
            shift2 = int(cols * 0.6)
            
            # Image 2: shifted right with overlap
            M1 = np.float32([[1, 0, shift1], [0, 1, 0]])
            img2 = cv2.warpAffine(image, M1, (cols + shift1, rows))
            
            # Image 3: shifted further right with overlap
            M2 = np.float32([[1, 0, shift2], [0, 1, 0]])
            img3 = cv2.warpAffine(image, M2, (cols + shift2, rows))
            
            images = [image, img2, img3]
            st.info(f"使用{len(images)}张内置重叠图像（模拟全景场景）")
        
        for i, img in enumerate(images[1:], 2):
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"图像{i}", use_container_width=True, clamp=True)
    
    # Blending method selection
    st.subheader("融合方法选择")
    blending_method = st.selectbox("选择融合方法", ["直接拼接", "加权融合"])
    
    if st.button("执行全景拼接"):
        if len(images) < 2:
            st.error("至少需要2张图像才能拼接")
            return
        
        # Step 1: Feature detection
        st.subheader("步骤1: 特征检测与匹配")
        sift = cv2.SIFT_create()
        
        # Detect features in all images
        all_kp = []
        all_des = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            all_kp.append(kp)
            all_des.append(des)
        
        st.info(f"各图像检测到的特征点数量: {[len(kp) for kp in all_kp]}")
        
        # Step 2: Match consecutive images
        st.subheader("步骤2: 图像配准")
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        homographies = []
        
        for i in range(len(images) - 1):
            matches = bf.knnMatch(all_des[i], all_des[i + 1], k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) >= 4:
                src_pts = np.float32([all_kp[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([all_kp[i + 1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                homographies.append(M)
                st.info(f"图像{i+1} -> 图像{i+2}: {len(good_matches)} 对匹配，{sum(mask.ravel())} 对内点")
            else:
                st.error(f"图像{i+1}和图像{i+2}匹配点不足")
                return
        
        # Step 3: Stitch images using manual method
        st.subheader("步骤3: 图像拼接")
        
        try:
            # Manual stitching (more robust)
            result = images[0].copy()
            
            for i in range(1, len(images)):
                # Warp current image
                h, w = images[i].shape[:2]
                
                # Calculate output size
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                warped_corners = cv2.perspectiveTransform(corners, homographies[i-1])
                
                all_corners = np.concatenate([
                    np.float32([[0, 0], [result.shape[1], 0], [result.shape[1], result.shape[0]], [0, result.shape[0]]]).reshape(-1, 1, 2),
                    warped_corners
                ], axis=0)
                
                [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
                [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
                
                # Adjust homography for translation
                translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
                H = translation.dot(homographies[i-1])
                
                # Warp image
                warped = cv2.warpPerspective(images[i], H, (x_max - x_min, y_max - y_min))
                
                # Expand result canvas
                result_expanded = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
                result_expanded[-y_min:result.shape[0]-y_min, -x_min:result.shape[1]-x_min] = result
                
                # Blend
                if blending_method == "直接拼接":
                    mask = np.zeros_like(warped)
                    mask[warped.sum(axis=2) > 0] = 1
                    mask_result = np.zeros_like(result_expanded)
                    mask_result[result_expanded.sum(axis=2) > 0] = 1
                    
                    overlap = mask * mask_result
                    result_expanded[overlap > 0] = warped[overlap > 0]
                    result_expanded[mask > 0] = np.where(mask > 0, warped, result_expanded)[mask > 0]
                    result = result_expanded
                else:
                    mask = np.zeros_like(warped, dtype=np.float32)
                    mask[warped.sum(axis=2) > 0] = 1
                    
                    mask_result = np.zeros_like(result_expanded, dtype=np.float32)
                    mask_result[result_expanded.sum(axis=2) > 0] = 1
                    
                    total_mask = mask + mask_result
                    total_mask[total_mask == 0] = 1
                    
                    result = (warped.astype(np.float32) * mask + result_expanded.astype(np.float32) * mask_result) / total_mask
                    result = np.uint8(result)
            
            # Crop black borders
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                x, y, w, h = cv2.boundingRect(contours[0])
                panorama = result[y:y+h, x:x+w]
            else:
                panorama = result
            
            st.success("拼接成功！")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Panorama ({blending_method})')
            st.pyplot(fig)
            plt.close(fig)
        
        except Exception as e:
            st.error(f"拼接过程中出错: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# ==================== Main Application ====================
def main():
    st.title("🎯 计算机视觉作业A3")
    st.markdown("**边缘检测、特征点检测、图像匹配与全景拼接**")
    st.markdown("---")
    
    # Load image
    image = load_image()
    
    # Sidebar preview
    st.sidebar.header("图像预览")
    st.sidebar.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs(["Canny边缘检测", "特征点检测", "图像匹配", "全景拼接"])
    
    with tab1:
        canny_edge_detection(image)
    
    with tab2:
        feature_detection(image)
    
    with tab3:
        image_matching(image)
    
    with tab4:
        panorama_stitching(image)
    
    # Footer
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("1. 程序自动加载内置图片 pic.jpg")
    st.markdown("2. 在左侧标签页选择不同的处理功能")
    st.markdown("3. 调整参数后点击按钮执行处理")
    st.markdown("4. 查看分步可视化的处理结果")
    
    st.markdown("---")
    st.markdown("💡 **提示**: 如果 pic.jpg 不存在，程序会自动生成测试图片")

if __name__ == "__main__":
    main()
