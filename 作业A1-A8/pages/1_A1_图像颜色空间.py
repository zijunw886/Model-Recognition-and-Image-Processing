"""
Computer Vision Assignment A1
Color Space Conversion and Image Interpolation
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="A1: Color Space & Interpolation",
    page_icon="🎨",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Built-in test image (RGB color blocks)
def create_test_image():
    """Create a built-in test image with RGB color blocks"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Red block (top-left)
    img[0:128, 0:128] = [255, 0, 0]
    # Green block (top-right)
    img[0:128, 128:256] = [0, 255, 0]
    # Blue block (bottom-left)
    img[128:256, 0:128] = [0, 0, 255]
    # Yellow block (bottom-right)
    img[128:256, 128:256] = [255, 255, 0]
    
    # Add gradient areas for better interpolation testing
    for i in range(64):
        img[64:128, i*2:(i+1)*2] = [int(255 * i / 64), int(255 * (64-i) / 64), 128]
        img[128:192, i*2:(i+1)*2] = [128, int(255 * i / 64), int(255 * (64-i) / 64)]
    
    return img

def load_default_image():
    """Load default image from root directory or create test image"""
    img_path = os.path.join(ROOT_DIR, 'pic.jpg')
    if os.path.exists(img_path):
        try:
            image = cv2.imread(img_path)
            return image
        except:
            return create_test_image()
    return create_test_image()

# ==================== Color Space Conversion ====================
def color_space_conversion(image):
    """Convert RGB to HSV and extract channels"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split RGB channels
    r_channel = rgb_image[:, :, 0]
    g_channel = rgb_image[:, :, 1]
    b_channel = rgb_image[:, :, 2]
    
    # Split HSV channels
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]
    
    return {
        'rgb': rgb_image,
        'hsv': hsv_image,
        'r': r_channel,
        'g': g_channel,
        'b': b_channel,
        'h': h_channel,
        's': s_channel,
        'v': v_channel
    }

def display_color_space_tab(image):
    """Display color space conversion results"""
    st.header("颜色空间转换")
    st.markdown("---")
    
    channels = color_space_conversion(image)
    
    # Original RGB Image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始RGB图像")
        st.image(channels['rgb'], channels='RGB', use_container_width=True)
    
    with col2:
        st.subheader("HSV图像")
        st.image(channels['hsv'], channels='RGB', use_container_width=True)
    
    st.markdown("---")
    
    # RGB Channels
    st.subheader("RGB通道")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(channels['r'], caption='Red Channel (R)', use_container_width=True, clamp=True)
    with col2:
        st.image(channels['g'], caption='Green Channel (G)', use_container_width=True, clamp=True)
    with col3:
        st.image(channels['b'], caption='Blue Channel (B)', use_container_width=True, clamp=True)
    
    st.markdown("---")
    
    # HSV Channels
    st.subheader("HSV通道")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(channels['h'], caption='Hue Channel (H)', use_container_width=True, clamp=True)
    with col2:
        st.image(channels['s'], caption='Saturation Channel (S)', use_container_width=True, clamp=True)
    with col3:
        st.image(channels['v'], caption='Value Channel (V)', use_container_width=True, clamp=True)

# ==================== Image Interpolation ====================
def nearest_neighbor_interpolation(image, scale_x, scale_y):
    """最近邻插值"""
    height, width = image.shape[:2]
    new_height = int(height * scale_y)
    new_width = int(width * scale_x)
    
    # Create new image
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            src_i = int(i / scale_y)
            src_j = int(j / scale_x)
            src_i = min(src_i, height - 1)
            src_j = min(src_j, width - 1)
            result[i, j] = image[src_i, src_j]
    
    return result

def bilinear_interpolation(image, scale_x, scale_y):
    """双线性插值"""
    height, width = image.shape[:2]
    new_height = int(height * scale_y)
    new_width = int(width * scale_x)
    
    # Create new image
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            # Calculate source coordinates
            src_i = i / scale_y
            src_j = j / scale_x
            
            # Get surrounding integer coordinates
            i0 = int(src_i)
            j0 = int(src_j)
            i1 = min(i0 + 1, height - 1)
            j1 = min(j0 + 1, width - 1)
            
            # Calculate fractional parts
            i_ratio = src_i - i0
            j_ratio = src_j - j0
            
            # Bilinear interpolation for each channel
            for c in range(3):
                top_left = image[i0, j0, c]
                top_right = image[i0, j1, c]
                bottom_left = image[i1, j0, c]
                bottom_right = image[i1, j1, c]
                
                top = top_left * (1 - j_ratio) + top_right * j_ratio
                bottom = bottom_left * (1 - j_ratio) + bottom_right * j_ratio
                result[i, j, c] = int(top * (1 - i_ratio) + bottom * i_ratio)
    
    return result

def rotate_image(image, angle, method='nearest'):
    """旋转图像"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if method == 'nearest':
        # Manual nearest neighbor rotation
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        scale_x = new_width / width
        scale_y = new_height / height
        
        for i in range(new_height):
            for j in range(new_width):
                # Inverse rotation
                y = (j - new_width/2) * np.cos(-angle * np.pi/180) - (i - new_height/2) * np.sin(-angle * np.pi/180) + center[1]
                x = (j - new_width/2) * np.sin(-angle * np.pi/180) + (i - new_height/2) * np.cos(-angle * np.pi/180) + center[0]
                
                x = int(x + 0.5)
                y = int(y + 0.5)
                
                if 0 <= x < width and 0 <= y < height:
                    result[i, j] = image[y, x]
        
        return result
    else:
        # OpenCV bilinear rotation
        return cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

def stretch_image(image, stretch_x, stretch_y, method='nearest'):
    """拉伸图像"""
    if method == 'nearest':
        return nearest_neighbor_interpolation(image, stretch_x, stretch_y)
    else:
        return bilinear_interpolation(image, stretch_x, stretch_y)

def display_interpolation_tab(image):
    """显示图像插值结果"""
    st.header("图像插值")
    st.markdown("---")
    
    # Operation selection
    operation = st.selectbox(
        "选择操作",
        ["缩放", "旋转", "拉伸"],
        key="interp_operation"
    )
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    if operation == "缩放":
        st.subheader("缩放操作")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            new_width = st.slider("新宽度 (像素)", 50, 500, 200, key="resize_width")
        with col2:
            new_height = st.slider("新高度 (像素)", 50, 500, 200, key="resize_height")
        
        method = st.selectbox("插值方法", ["最近邻", "双线性"], key="resize_method")
        
        # Perform interpolation
        if st.button("应用缩放", key="apply_resize"):
            scale_x = new_width / width
            scale_y = new_height / height
            
            with st.spinner("处理中..."):
                if method == "最近邻":
                    result = nearest_neighbor_interpolation(image, scale_x, scale_y)
                else:
                    result = bilinear_interpolation(image, scale_x, scale_y)
            
            # Display results
            st.subheader("缩放结果")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'原始图像 ({width}x{height})', use_container_width=True)
                st.caption(f"原始尺寸: {width}x{height}")
            with col2:
                st.image(result, caption=f'{method}结果 ({new_width}x{new_height})', use_container_width=True)
                st.caption(f"新尺寸: {new_width}x{new_height}")
            
            st.success(f"使用{method}方法从 {width}x{height} 缩放到 {new_width}x{new_height}")
    
    elif operation == "旋转":
        st.subheader("旋转操作")
        
        # Parameters
        angle = st.slider("旋转角度 (度)", -180, 180, 45, key="rotate_angle")
        method = st.selectbox("插值方法", ["最近邻", "双线性"], key="rotate_method")
        
        if st.button("应用旋转", key="apply_rotate"):
            with st.spinner("处理中..."):
                if method == "最近邻":
                    result = rotate_image(image, angle, method='nearest')
                else:
                    result = rotate_image(image, angle, method='bilinear')
            
            # Display results
            st.subheader("旋转结果")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'原始图像 ({width}x{height})', use_container_width=True)
            with col2:
                st.image(result, caption=f'{method}结果 ({angle}度)', use_container_width=True)
            
            st.success(f"使用{method}方法旋转 {angle} 度")
    
    elif operation == "拉伸":
        st.subheader("拉伸操作")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            stretch_x = st.slider("X轴缩放因子", 0.5, 3.0, 1.5, 0.1, key="stretch_x")
        with col2:
            stretch_y = st.slider("Y轴缩放因子", 0.5, 3.0, 1.5, 0.1, key="stretch_y")
        
        method = st.selectbox("插值方法", ["最近邻", "双线性"], key="stretch_method")
        
        if st.button("应用拉伸", key="apply_stretch"):
            with st.spinner("处理中..."):
                if method == "最近邻":
                    result = stretch_image(image, stretch_x, stretch_y, method='nearest')
                else:
                    result = stretch_image(image, stretch_x, stretch_y, method='bilinear')
            
            new_width = int(width * stretch_x)
            new_height = int(height * stretch_y)
            
            # Display results
            st.subheader("拉伸结果")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'原始图像 ({width}x{height})', use_container_width=True)
                st.caption(f"原始尺寸: {width}x{height}")
            with col2:
                st.image(result, caption=f'{method}结果 ({new_width}x{new_height})', use_container_width=True)
                st.caption(f"新尺寸: {new_width}x{new_height}")
            
            st.success(f"使用{method}方法从 {width}x{height} 拉伸到 {new_width}x{new_height}")

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("🎨 作业A1: 图像颜色空间与插值")
    st.markdown("**Color Space Conversion & Image Interpolation**")
    st.markdown("---")
    
    # Sidebar for image upload
    st.sidebar.header("图像输入")
    uploaded_file = st.sidebar.file_uploader("上传图像 (可选)", type=['jpg', 'png', 'jpeg'])
    
    # Load image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        # Use default image
        image = load_default_image()
        st.info("使用默认测试图像。从侧边栏上传您自己的图像。")
    
    # Display original image
    st.sidebar.subheader("原始图像预览")
    st.sidebar.image(image, channels='RGB', use_container_width=True)
    
    # Tab selection
    tab1, tab2 = st.tabs(["颜色空间转换", "图像插值"])
    
    with tab1:
        display_color_space_tab(image)
    
    with tab2:
        display_interpolation_tab(image)
    
    # Footer information
    st.markdown("---")
    st.markdown("### 信息说明")
    st.markdown("""
    - **内置测试图像**: RGB颜色块与渐变区域
    - **颜色空间**: RGB和HSV转换
    - **插值方法**: 最近邻和双线性
    - **操作**: 缩放、旋转、拉伸
    """)

if __name__ == "__main__":
    main()
