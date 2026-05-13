"""
Computer Vision Assignment A1
Color Space Conversion and Image Interpolation
"""

import streamlit as st
import numpy as np
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
            image = Image.open(img_path).convert("RGB")
            return np.array(image)
        except:
            return create_test_image()
    return create_test_image()

# ==================== Color Space Conversion ====================
def rgb_to_hsv(rgb):
    """Pure numpy RGB to HSV conversion (no cv2)"""
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    
    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    
    mask = maxc != minc
    d = maxc - minc
    s[mask] = d[mask] / maxc[mask]
    
    mask_r = (maxc == r) & mask
    mask_g = (maxc == g) & mask
    mask_b = (maxc == b) & mask
    
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / d[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / d[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / d[mask_b]) + 240) % 360
    
    h = (h / 2).astype(np.uint8)
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)
    
    hsv = np.stack([h, s, v], axis=-1)
    return hsv, h, s, v

def color_space_conversion(image):
    """Convert RGB to HSV and extract channels (NO CV2)"""
    rgb_image = image
    hsv_image, h, s, v = rgb_to_hsv(rgb_image)
    
    r_channel = rgb_image[:, :, 0]
    g_channel = rgb_image[:, :, 1]
    b_channel = rgb_image[:, :, 2]
    
    return {
        'rgb': rgb_image,
        'hsv': hsv_image,
        'r': r_channel,
        'g': g_channel,
        'b': b_channel,
        'h': h,
        's': s,
        'v': v
    }

def display_color_space_tab(image):
    """Display color space conversion results"""
    st.header("颜色空间转换")
    st.markdown("---")
    
    channels = color_space_conversion(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始RGB图像")
        st.image(channels['rgb'], channels='RGB', use_container_width=True)
    
    with col2:
        st.subheader("HSV图像")
        hsv_disp = cv2.cvtColor(channels['hsv'], cv2.COLOR_HSV2RGB) if 'cv2' in globals() else channels['hsv']
        st.image(channels['hsv'], channels='RGB', use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("RGB通道")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(channels['r'], caption='Red Channel (R)', use_container_width=True, clamp=True)
    with col2:
        st.image(channels['g'], caption='Green Channel (G)', use_container_width=True, clamp=True)
    with col3:
        st.image(channels['b'], caption='Blue Channel (B)', use_container_width=True, clamp=True)
    
    st.markdown("---")
    
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
    
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            src_i = i / scale_y
            src_j = j / scale_x
            
            i0 = int(src_i)
            j0 = int(src_j)
            i1 = min(i0 + 1, height - 1)
            j1 = min(j0 + 1, width - 1)
            
            i_ratio = src_i - i0
            j_ratio = src_j - j0
            
            for c in range(3):
                top_left = image[i0, j0, c]
                top_right = image[i0, j1, c]
                bottom_left = image[i1, j0, c]
                bottom_right = image[i1, j1, c]
                
                top = top_left * (1 - j_ratio) + top_right * j_ratio
                bottom = bottom_left * (1 - j_ratio) + bottom_right * j_ratio
                result[i, j, c] = int(top * (1 - i_ratio) + bottom * i_ratio)
    
    return result

def rotate_image(image, angle):
    """纯numpy旋转（无cv2）"""
    img = Image.fromarray(image)
    rotated = img.rotate(angle, expand=True, resample=Image.BILINEAR)
    return np.array(rotated)

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
    
    operation = st.selectbox(
        "选择操作",
        ["缩放", "旋转", "拉伸"],
        key="interp_operation"
    )
    
    height, width = image.shape[:2]
    
    if operation == "缩放":
        st.subheader("缩放操作")
        
        col1, col2 = st.columns(2)
        with col1:
            new_width = st.slider("新宽度 (像素)", 50, 500, 200, key="resize_width")
        with col2:
            new_height = st.slider("新高度 (像素)", 50, 500, 200, key="resize_height")
        
        method = st.selectbox("插值方法", ["最近邻", "双线性"], key="resize_method")
        
        if st.button("应用缩放", key="apply_resize"):
            scale_x = new_width / width
            scale_y = new_height / height
            
            with st.spinner("处理中..."):
                if method == "最近邻":
                    result = nearest_neighbor_interpolation(image, scale_x, scale_y)
                else:
                    result = bilinear_interpolation(image, scale_x, scale_y)
            
            st.subheader("缩放结果")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'原始图像 ({width}x{height})', use_container_width=True)
            with col2:
                st.image(result, caption=f'{method}结果 ({new_width}x{new_height})', use_container_width=True)
    
    elif operation == "旋转":
        st.subheader("旋转操作")
        angle = st.slider("旋转角度 (度)", -180, 180, 45, key="rotate_angle")
        
        if st.button("应用旋转", key="apply_rotate"):
            with st.spinner("处理中..."):
                result = rotate_image(image, angle)
            
            st.subheader("旋转结果")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'原始图像', use_container_width=True)
            with col2:
                st.image(result, caption=f'旋转 {angle} 度', use_container_width=True)
    
    elif operation == "拉伸":
        st.subheader("拉伸操作")
        
        col1, col2 = st.columns(2)
        with col1:
            stretch_x = st.slider("X轴缩放因子", 0.5, 3.0, 1.5, 0.1, key="stretch_x")
        with col2:
            stretch_y = st.slider("Y轴缩放因子", 0.5, 3.0, 1.5, 0.1, key="stretch_y")
        
        method = st.selectbox("插值方法", ["最近邻", "双线性"], key="stretch_method")
        
        if st.button("应用拉伸", key="apply_stretch"):
            with st.spinner("处理中..."):
                result = stretch_image(image, stretch_x, stretch_y, method)
            
            new_w = int(width * stretch_x)
            new_h = int(height * stretch_y)
            
            st.subheader("拉伸结果")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'原始', use_container_width=True)
            with col2:
                st.image(result, caption=f'结果 {new_w}x{new_h}', use_container_width=True)

# ==================== Main Application ====================
def main():
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    st.title("🎨 作业A1: 图像颜色空间与插值")
    st.markdown("**Color Space Conversion & Image Interpolation**")
    st.markdown("---")
    
    st.sidebar.header("图像输入")
    uploaded_file = st.sidebar.file_uploader("上传图像 (可选)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)
    else:
        image = load_default_image()
        st.info("使用默认测试图像")
    
    st.sidebar.subheader("原始图像预览")
    st.sidebar.image(image, channels='RGB', use_container_width=True)
    
    tab1, tab2 = st.tabs(["颜色空间转换", "图像插值"])
    
    with tab1:
        display_color_space_tab(image)
    with tab2:
        display_interpolation_tab(image)
    
    st.markdown("---")
    st.markdown("### 信息说明")
    st.markdown("- 内置测试图像: RGB颜色块与渐变区域\n- 颜色空间: RGB和HSV\n- 插值方法: 最近邻、双线性\n- 操作: 缩放、旋转、拉伸")

if __name__ == "__main__":
    main()
