"""
Computer Vision Assignment A6
FCN Semantic Segmentation, R-CNN Object Detection, Mask R-CNN Instance Segmentation
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="A6: Segmentation & Detection",
    page_icon="🎯",
    layout="wide"
)

# ==================== Helper Functions ====================
def load_image():
    """Load image from pic.jpg or create test image"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'pic.jpg')
    
    if os.path.exists(img_path):
        return img_path
    return None

def get_device():
    """Get available device (GPU if available)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== FCN Semantic Segmentation ====================
def fcn_segmentation():
    """FCN semantic segmentation with pretrained model"""
    st.header("FCN语义分割")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Load image
    img_path = load_image()
    if img_path is None:
        st.error("未找到 pic.jpg 文件")
        return
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load FCN model
    @st.cache_resource
    def load_fcn_model():
        model = models.segmentation.fcn_resnet101(pretrained=True)
        model = model.to(get_device())
        model.eval()
        return model
    
    model = load_fcn_model()
    
    # Load and process image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(get_device())
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
    
    # Get segmentation mask
    mask = output.argmax(0).cpu().numpy()
    
    # Pascal VOC color map
    def get_pascal_color_map():
        colors = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ], dtype=np.uint8)
        return colors
    
    colors = get_pascal_color_map()
    colored_mask = colors[mask]
    
    # Visualization
    st.subheader("分割结果")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("输入图像")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img.resize((256, 256)))
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("分割掩码")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(colored_mask)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        st.pyplot(fig)
        plt.close(fig)
    
    with col3:
        st.subheader("叠加结果")
        fig, ax = plt.subplots(figsize=(6, 6))
        img_np = np.array(img.resize((256, 256)))
        blended = cv2.addWeighted(img_np, 0.7, colored_mask, 0.3, 0)
        ax.imshow(blended)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    
    # Class distribution
    st.subheader("类别分布统计")
    unique, counts = np.unique(mask, return_counts=True)
    class_names = ['背景', '飞机', '自行车', '鸟', '船', '瓶子', '公交车', '汽车', 
                   '猫', '椅子', '牛', '餐桌', '狗', '马', '摩托车', '人', 
                   '盆栽', '绵羊', '沙发', '火车', '电视']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([class_names[i] for i in unique], counts)
    ax.set_xlabel('Class')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Segmentation Class Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

# ==================== R-CNN Object Detection ====================
def rcnn_detection():
    """R-CNN series object detection demonstration"""
    st.header("R-CNN系列目标检测")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Load image
    img_path = load_image()
    if img_path is None:
        st.error("未找到 pic.jpg 文件")
        return
    
    # Model selection
    model_type = st.selectbox("选择模型", ["Faster R-CNN", "Faster R-CNN (MobileNet)"])
    
    # Load model
    @st.cache_resource
    def load_rcnn_model(model_name):
        if model_name == "Faster R-CNN":
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        model = model.to(get_device())
        model.eval()
        return model
    
    model = load_rcnn_model(model_type)
    
    # Confidence threshold
    conf_threshold = st.slider("置信度阈值", 0.1, 0.9, 0.5, key="rcnn_conf_threshold")
    
    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load and process image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(get_device())
    
    # Inference with timing
    import time
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)[0]
    inference_time = time.time() - start_time
    
    # Get detections
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
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
    
    # Visualize results
    st.subheader("检测结果")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                            fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}",
                bbox=dict(facecolor='yellow', alpha=0.8), fontsize=10)
    
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)
    
    st.info(f"检测到 {len(boxes)} 个目标")
    st.info(f"推理时间: {inference_time:.4f} 秒")
    
    # Model comparison
    st.subheader("模型对比")
    model_data = {
        '模型': ['R-CNN', 'Fast R-CNN', 'Faster R-CNN', 'Faster R-CNN (Mobile)'],
        'mAP': [66.0, 70.0, 73.2, 68.0],
        'FPS': [0.5, 5, 15, 30]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(model_data['模型'], model_data['mAP'], color='blue')
        ax.set_xlabel('Model')
        ax.set_ylabel('mAP (%)')
        ax.set_title('Model mAP Comparison')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(model_data['模型'], model_data['FPS'], color='red')
        ax.set_xlabel('Model')
        ax.set_ylabel('FPS')
        ax.set_title('Model Speed Comparison')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

# ==================== Mask R-CNN Instance Segmentation ====================
def mask_rcnn_segmentation():
    """Mask R-CNN instance segmentation"""
    st.header("Mask R-CNN实例分割")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Load image
    img_path = load_image()
    if img_path is None:
        st.error("未找到 pic.jpg 文件")
        return
    
    # Load Mask R-CNN model
    @st.cache_resource
    def load_maskrcnn_model():
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model = model.to(get_device())
        model.eval()
        return model
    
    model = load_maskrcnn_model()
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold = st.slider("置信度阈值", 0.1, 0.9, 0.5, key="maskrcnn_conf_threshold")
    with col2:
        mask_threshold = st.slider("掩码阈值", 0.1, 0.9, 0.5, key="maskrcnn_mask_threshold")
    
    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load and process image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(get_device())
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)[0]
    
    # Get results
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy()
    
    # Filter by confidence
    mask_filter = scores >= conf_threshold
    boxes = boxes[mask_filter]
    labels = labels[mask_filter]
    scores = scores[mask_filter]
    masks = masks[mask_filter]
    
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
    
    # Generate colors
    colors = plt.cm.get_cmap('tab20', len(boxes))
    
    # Visualize results
    st.subheader("实例分割结果")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("检测框")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                fill=False, color=colors(i), linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.8), fontsize=10)
        
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("实例掩码")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        for i, (mask, label) in enumerate(zip(masks, labels)):
            mask_img = mask[0] >= mask_threshold
            color = colors(i)[:3]
            ax.imshow(np.where(mask_img[..., None], color, 0), alpha=0.5)
        
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    
    # Combined visualization
    st.subheader("综合结果")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    for i, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):
        x1, y1, x2, y2 = box
        
        # Draw mask
        mask_img = mask[0] >= mask_threshold
        color = colors(i)[:3]
        ax.imshow(np.where(mask_img[..., None], color, 0), alpha=0.4)
        
        # Draw box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                            fill=False, color=colors(i), linewidth=2)
        ax.add_patch(rect)
        
        # Draw label
        ax.text(x1, y1 - 5, f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}",
                bbox=dict(facecolor='yellow', alpha=0.8), fontsize=10)
    
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

# ==================== Performance Comparison ====================
def performance_comparison():
    """Performance comparison of different models"""
    st.header("性能对比")
    st.markdown("---")
    
    # Predefined performance data
    model_data = {
        '模型': ['FCN-ResNet101', 'Faster R-CNN', 'Mask R-CNN', 'YOLOv5'],
        'mAP (%)': [68.4, 73.2, 77.0, 78.0],
        '推理时间(ms)': [230, 150, 280, 45],
        '参数量(M)': [134, 42, 47, 25]
    }
    
    st.subheader("模型性能对比表格")
    st.dataframe(model_data)
    
    # Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("mAP对比")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(model_data['模型'], model_data['mAP (%)'], color='blue')
        ax.set_xlabel('Model')
        ax.set_ylabel('mAP (%)')
        ax.set_title('Model mAP Comparison')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("推理时间对比")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(model_data['模型'], model_data['推理时间(ms)'], color='red')
        ax.set_xlabel('Model')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Model Inference Time')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    
    st.subheader("参数量对比")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(model_data['模型'], model_data['参数量(M)'], color='green')
    ax.set_xlabel('Model')
    ax.set_ylabel('Parameters (M)')
    ax.set_title('Model Parameter Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)
    
    # Radar chart
    st.subheader("综合性能雷达图")
    categories = ['mAP', 'Speed', 'Params']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    for i, model in enumerate(model_data['模型']):
        values = [
            model_data['mAP (%)'][i] / 100,
            1 - model_data['推理时间(ms)'][i] / 300,
            1 - model_data['参数量(M)'][i] / 150
        ]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ==================== Main Application ====================
def main():
    st.title("🎯 计算机视觉作业A6")
    st.markdown("**FCN语义分割、R-CNN目标检测、Mask R-CNN实例分割**")
    st.markdown("---")
    
    # Sidebar preview
    img_path = load_image()
    if img_path:
        st.sidebar.header("测试图像")
        st.sidebar.image(img_path, use_container_width=True)
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs([
        "FCN语义分割", 
        "R-CNN目标检测", 
        "Mask R-CNN实例分割", 
        "性能对比"
    ])
    
    with tab1:
        fcn_segmentation()
    
    with tab2:
        rcnn_detection()
    
    with tab3:
        mask_rcnn_segmentation()
    
    with tab4:
        performance_comparison()
    
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("1. 在上方标签页选择不同的功能模块")
    st.markdown("2. 调整参数后自动更新结果")
    st.markdown("3. 查看可视化结果")

if __name__ == "__main__":
    main()
