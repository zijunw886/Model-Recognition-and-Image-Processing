"""
Computer Vision Assignment A6
Object Detection and Segmentation
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# Try to import torch
try:
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.segmentation import fcn_resnet101, maskrcnn_resnet50_fpn
    from torchvision.transforms import ToTensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="A6: Object Detection & Segmentation",
    page_icon="🎯",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ==================== FCN Segmentation ====================
def fcn_segmentation():
    """FCN semantic segmentation"""
    st.header("FCN语义分割")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Load model
    model = fcn_resnet101(pretrained=True).to(get_device())
    model.eval()
    
    # Create test image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    if st.button("执行语义分割", key="fcn_btn"):
        transform = ToTensor()
        input_tensor = transform(img).unsqueeze(0).to(get_device())
        
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        # Get segmentation mask
        mask = output.argmax(0).cpu().numpy()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        ax2.imshow(mask, cmap='tab20')
        ax2.set_title('Segmentation Mask')
        ax2.axis('off')
        
        st.pyplot(fig)
        plt.close(fig)

# ==================== Faster R-CNN Detection ====================
def faster_rcnn_detection():
    """Faster R-CNN object detection"""
    st.header("Faster R-CNN目标检测")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(get_device())
    model.eval()
    
    conf_threshold = st.slider("置信度阈值", 0.1, 0.9, 0.5, key="rcnn_conf")
    
    # Create test image
    img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    if st.button("执行目标检测", key="rcnn_btn"):
        transform = ToTensor()
        input_tensor = transform(img).unsqueeze(0).to(get_device())
        
        with torch.no_grad():
            output = model(input_tensor)[0]
        
        # Plot with bounding boxes
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            if score > conf_threshold:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1, f'{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}', 
                        color='white', backgroundcolor='red', fontsize=8)
        
        ax.set_title('Object Detection Results')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

# ==================== Mask R-CNN Segmentation ====================
def mask_rcnn_segmentation():
    """Mask R-CNN instance segmentation"""
    st.header("Mask R-CNN实例分割")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Load model
    model = maskrcnn_resnet50_fpn(pretrained=True).to(get_device())
    model.eval()
    
    conf_threshold = st.slider("置信度阈值", 0.1, 0.9, 0.5, key="maskrcnn_conf")
    mask_threshold = st.slider("掩码阈值", 0.1, 0.9, 0.5, key="maskrcnn_mask")
    
    # Create test image
    img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    if st.button("执行实例分割", key="maskrcnn_btn"):
        transform = ToTensor()
        input_tensor = transform(img).unsqueeze(0).to(get_device())
        
        with torch.no_grad():
            output = model(input_tensor)[0]
        
        # Plot with masks
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        masks = output['masks'].cpu().numpy()
        
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
        
        for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            if score > conf_threshold:
                color = colors[i % len(colors)]
                
                # Draw mask
                mask_img = mask[0] > mask_threshold
                ax.imshow(mask_img, alpha=0.3, cmap=plt.cm.get_cmap('tab10')(i))
                
                # Draw box
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2)
                ax.add_patch(rect)
                
                ax.text(x1, y1, f'{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}', 
                        color='white', backgroundcolor=color, fontsize=8)
        
        ax.set_title('Instance Segmentation Results')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("🎯 作业A6: 目标检测与分割")
    st.markdown("**Object Detection & Segmentation**")
    st.markdown("---")
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs([
        "FCN语义分割", 
        "Faster R-CNN检测", 
        "Mask R-CNN分割"
    ])
    
    with tab1:
        fcn_segmentation()
    
    with tab2:
        faster_rcnn_detection()
    
    with tab3:
        mask_rcnn_segmentation()

if __name__ == "__main__":
    main()
