"""
Computer Vision Assignment A7
Self-Supervised Learning
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="A7: Self-Supervised Learning",
    page_icon="🔄",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# ==================== Rotation Prediction ====================
def rotation_prediction():
    """Rotation prediction task"""
    st.header("图像旋转预测")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Simple model
    class RotationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 4)  # 4 rotation angles
            )
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1, 64 * 7 * 7)
            x = self.fc(x)
            return x
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    epochs = st.slider("训练轮数", 1, 5, 2, key="rot_epochs")
    
    if st.button("训练旋转预测模型", key="rot_train"):
        model = RotationModel().to(get_device())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for img, _ in train_data:
                # Rotate image
                angle_idx = np.random.randint(4)
                angle = angle_idx * 90
                rotated = transforms.functional.rotate(img, angle)
                rotated = rotated.unsqueeze(0).to(get_device())
                target = torch.tensor([angle_idx]).to(get_device())
                
                optimizer.zero_grad()
                output = model(rotated)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            losses.append(avg_loss)
            st.write(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Plot loss
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Rotation Prediction Training Loss')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ==================== MAE Demo ====================
def mae_demo():
    """MAE masking demo"""
    st.header("MAE遮挡重建")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    mask_ratio = st.slider("遮挡比例", 0.2, 0.8, 0.5, key="mae_mask")
    
    # Create test image
    img = np.random.rand(28, 28)
    
    # Apply masking
    mask = np.random.rand(28, 28) < mask_ratio
    masked_img = img.copy()
    masked_img[mask] = 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(masked_img, cmap='gray')
    ax2.set_title(f'Masked Image ({mask_ratio*100:.0f}% masked)')
    ax2.axis('off')
    
    st.pyplot(fig)
    plt.close(fig)

# ==================== SimCLR Demo ====================
def simclr_demo():
    """SimCLR contrastive learning demo"""
    st.header("SimCLR对比学习")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Data augmentation options
    aug_type = st.selectbox("数据增强方式", ["随机裁剪", "颜色抖动", "高斯模糊"])
    
    # Create test image
    img = np.random.rand(32, 32, 3)
    
    # Apply augmentation
    if aug_type == "随机裁剪":
        aug_img = img[4:28, 4:28]
        aug_img = np.pad(aug_img, ((4, 4), (4, 4), (0, 0)), mode='constant')
    elif aug_type == "颜色抖动":
        aug_img = img + np.random.normal(0, 0.1, img.shape)
        aug_img = np.clip(aug_img, 0, 1)
    else:
        # Gaussian blur
        aug_img = img.copy()
        for i in range(1, 31):
            for j in range(1, 31):
                aug_img[i, j] = np.mean(img[i-1:i+2, j-1:j+2], axis=(0, 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(aug_img)
    ax2.set_title(f'{aug_type}增强')
    ax2.axis('off')
    
    st.pyplot(fig)
    plt.close(fig)

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("🔄 作业A7: 自监督学习")
    st.markdown("**Self-Supervised Learning**")
    st.markdown("---")
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs([
        "旋转预测", 
        "MAE遮挡重建", 
        "SimCLR对比学习"
    ])
    
    with tab1:
        rotation_prediction()
    
    with tab2:
        mae_demo()
    
    with tab3:
        simclr_demo()

if __name__ == "__main__":
    main()
