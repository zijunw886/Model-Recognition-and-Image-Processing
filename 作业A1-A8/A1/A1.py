"""
Computer Vision Assignment A1
Color Space Conversion and Image Interpolation
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="A1: Color Space & Interpolation",
    page_icon="🎨",
    layout="wide"
)

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
    st.header("Color Space Conversion")
    st.markdown("---")
    
    channels = color_space_conversion(image)
    
    # Original RGB Image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original RGB Image")
        st.image(channels['rgb'], channels='RGB', use_container_width=True)
    
    with col2:
        st.subheader("HSV Image")
        st.image(channels['hsv'], channels='RGB', use_container_width=True)
    
    st.markdown("---")
    
    # RGB Channels
    st.subheader("RGB Channels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(channels['r'], caption='Red Channel (R)', use_container_width=True, clamp=True)
    with col2:
        st.image(channels['g'], caption='Green Channel (G)', use_container_width=True, clamp=True)
    with col3:
        st.image(channels['b'], caption='Blue Channel (B)', use_container_width=True, clamp=True)
    
    st.markdown("---")
    
    # HSV Channels
    st.subheader("HSV Channels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(channels['h'], caption='Hue Channel (H)', use_container_width=True, clamp=True)
    with col2:
        st.image(channels['s'], caption='Saturation Channel (S)', use_container_width=True, clamp=True)
    with col3:
        st.image(channels['v'], caption='Value Channel (V)', use_container_width=True, clamp=True)

# ==================== Image Interpolation ====================
def nearest_neighbor_interpolation(image, scale_x, scale_y):
    """Nearest Neighbor Interpolation"""
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
    """Bilinear Interpolation"""
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
    """Rotate image using specified interpolation method"""
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
    """Stretch image using specified interpolation method"""
    if method == 'nearest':
        return nearest_neighbor_interpolation(image, stretch_x, stretch_y)
    else:
        return bilinear_interpolation(image, stretch_x, stretch_y)

def display_interpolation_tab(image):
    """Display image interpolation results"""
    st.header("Image Interpolation")
    st.markdown("---")
    
    # Operation selection
    operation = st.selectbox(
        "Select Operation",
        ["Resize", "Rotate", "Stretch"],
        key="interp_operation"
    )
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    if operation == "Resize":
        st.subheader("Resize Operation")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            new_width = st.slider("New Width (pixels)", 50, 500, 200, key="resize_width")
        with col2:
            new_height = st.slider("New Height (pixels)", 50, 500, 200, key="resize_height")
        
        method = st.selectbox("Interpolation Method", ["Nearest Neighbor", "Bilinear"], key="resize_method")
        
        # Perform interpolation
        if st.button("Apply Resize", key="apply_resize"):
            scale_x = new_width / width
            scale_y = new_height / height
            
            with st.spinner("Processing..."):
                if method == "Nearest Neighbor":
                    result = nearest_neighbor_interpolation(image, scale_x, scale_y)
                else:
                    result = bilinear_interpolation(image, scale_x, scale_y)
            
            # Display results
            st.subheader("Resize Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'Original Image ({width}x{height})', use_container_width=True)
                st.caption(f"Original size: {width}x{height}")
            with col2:
                st.image(result, caption=f'{method} Result ({new_width}x{new_height})', use_container_width=True)
                st.caption(f"New size: {new_width}x{new_height}")
            
            st.success(f"Resized from {width}x{height} to {new_width}x{new_height} using {method}")
    
    elif operation == "Rotate":
        st.subheader("Rotate Operation")
        
        # Parameters
        angle = st.slider("Rotation Angle (degrees)", -180, 180, 45, key="rotate_angle")
        method = st.selectbox("Interpolation Method", ["Nearest Neighbor", "Bilinear"], key="rotate_method")
        
        if st.button("Apply Rotation", key="apply_rotate"):
            with st.spinner("Processing..."):
                if method == "Nearest Neighbor":
                    result = rotate_image(image, angle, method='nearest')
                else:
                    result = rotate_image(image, angle, method='bilinear')
            
            # Display results
            st.subheader("Rotation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'Original Image ({width}x{height})', use_container_width=True)
            with col2:
                st.image(result, caption=f'{method} Result ({angle} degrees)', use_container_width=True)
            
            st.success(f"Rotated by {angle} degrees using {method}")
    
    elif operation == "Stretch":
        st.subheader("Stretch Operation")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            stretch_x = st.slider("X Scale Factor", 0.5, 3.0, 1.5, 0.1, key="stretch_x")
        with col2:
            stretch_y = st.slider("Y Scale Factor", 0.5, 3.0, 1.5, 0.1, key="stretch_y")
        
        method = st.selectbox("Interpolation Method", ["Nearest Neighbor", "Bilinear"], key="stretch_method")
        
        if st.button("Apply Stretch", key="apply_stretch"):
            with st.spinner("Processing..."):
                if method == "Nearest Neighbor":
                    result = stretch_image(image, stretch_x, stretch_y, method='nearest')
                else:
                    result = stretch_image(image, stretch_x, stretch_y, method='bilinear')
            
            new_width = int(width * stretch_x)
            new_height = int(height * stretch_y)
            
            # Display results
            st.subheader("Stretch Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f'Original Image ({width}x{height})', use_container_width=True)
                st.caption(f"Original size: {width}x{height}")
            with col2:
                st.image(result, caption=f'{method} Result ({new_width}x{new_height})', use_container_width=True)
                st.caption(f"New size: {new_width}x{new_height}")
            
            st.success(f"Stretched from {width}x{height} to {new_width}x{new_height} using {method}")

# ==================== Main Application ====================
def main():
    # Title
    st.title("Computer Vision Assignment A1")
    st.markdown("**Color Space Conversion & Image Interpolation**")
    st.markdown("---")
    
    # Sidebar for image upload
    st.sidebar.header("Image Input")
    uploaded_file = st.sidebar.file_uploader("Upload an image (optional)", type=['jpg', 'png', 'jpeg'])
    
    # Load image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        # Use built-in test image
        image = create_test_image()
        st.info("Using built-in test image. Upload your own image from the sidebar.")
    
    # Display original image
    st.sidebar.subheader("Original Image Preview")
    st.sidebar.image(image, channels='RGB', use_container_width=True)
    
    # Tab selection
    tab1, tab2 = st.tabs(["Color Space Conversion", "Image Interpolation"])
    
    with tab1:
        display_color_space_tab(image)
    
    with tab2:
        display_interpolation_tab(image)
    
    # Footer information
    st.markdown("---")
    st.markdown("### Information")
    st.markdown("""
    - **Built-in Test Image**: RGB color blocks with gradient areas
    - **Color Space**: RGB and HSV conversion
    - **Interpolation Methods**: Nearest Neighbor and Bilinear
    - **Operations**: Resize, Rotate, Stretch
    """)
    
    # Algorithm explanations
    with st.expander("Algorithm Explanations"):
        st.markdown("""
        ### Color Space Conversion
        
        **RGB Color Space**: Additive color model based on Red, Green, and Blue channels.
        
        **HSV Color Space**: Hue (color type), Saturation (color purity), and Value (brightness).
        
        ### Image Interpolation
        
        **Nearest Neighbor Interpolation**: 
        - Simplest method
        - Assigns the value of the nearest pixel to the output location
        - Fast but can produce blocky results
        
        **Bilinear Interpolation**:
        - Uses 4 nearest pixels
        - Performs linear interpolation in both directions
        - Smoother results than nearest neighbor
        """)

if __name__ == "__main__":
    main()
