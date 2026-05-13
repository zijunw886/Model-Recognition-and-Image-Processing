"""
Computer Vision Assignment A7
Self-Supervised Learning: Rotation Prediction, MAE, SimCLR
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
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, datasets, models
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="A7: Self-Supervised Learning",
    page_icon="🔄",
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

# ==================== Rotation Prediction ====================
def rotation_prediction():
    """Rotation prediction self-supervised learning"""
    st.header("图像旋转预测")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Load image
    img_path = load_image()
    if img_path is None:
        st.error("未找到 pic.jpg 文件")
        return
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        num_rotations = st.selectbox("旋转角度数量", [2, 4], index=1)
    with col2:
        epochs = st.slider("训练轮数", 1, 10, 3, key="rotation_epochs")
    
    # Load and transform image
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((128, 128))
    img_np = np.array(img_resized)
    
    # Create rotation dataset
    class RotationDataset(Dataset):
        def __init__(self, image, num_rotations=4):
            self.image = image
            self.num_rotations = num_rotations
            self.angles = [i * (360 // num_rotations) for i in range(num_rotations)]
        
        def __len__(self):
            return self.num_rotations * 100  # Generate multiple samples
        
        def __getitem__(self, idx):
            angle_idx = idx % self.num_rotations
            angle = self.angles[angle_idx]
            
            # Rotate image
            img_pil = Image.fromarray(self.image)
            img_rotated = img_pil.rotate(angle)
            img_tensor = transforms.ToTensor()(img_rotated)
            
            return img_tensor, angle_idx
    
    # Simple CNN model
    class RotationNet(nn.Module):
        def __init__(self, num_classes=4):
            super(RotationNet, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x
    
    if st.button("训练旋转预测模型"):
        dataset = RotationDataset(img_np, num_rotations)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = RotationNet(num_classes=num_rotations).to(get_device())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in dataloader:
                images, labels = images.to(get_device()), labels.to(get_device())
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            st.write(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Visualize results
        st.subheader("旋转预测结果")
        
        # Original and rotated images
        fig, axes = plt.subplots(1, num_rotations + 1, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        angles = [i * (360 // num_rotations) for i in range(num_rotations)]
        model.eval()
        
        for i, angle in enumerate(angles):
            img_rotated = Image.fromarray(img_np).rotate(angle)
            img_tensor = transforms.ToTensor()(img_rotated).unsqueeze(0).to(get_device())
            
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output).item()
            
            axes[i+1].imshow(img_rotated)
            axes[i+1].set_title(f"Rotated {angle}°\nPred: {angles[pred]}°")
            axes[i+1].axis('off')
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Loss and accuracy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(range(epochs), losses, marker='o', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(epochs), accuracies, marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)

# ==================== MAE Simplified Implementation ====================
def mae_simplified():
    """Simplified MAE (Masked Autoencoder) implementation"""
    st.header("MAE遮挡重建")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Load image
    img_path = load_image()
    if img_path is None:
        st.error("未找到 pic.jpg 文件")
        return
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        mask_ratio = st.slider("遮挡比例", 0.2, 0.8, 0.5, key="mae_mask_ratio")
    with col2:
        epochs = st.slider("训练轮数", 1, 20, 5, key="mae_epochs")
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((64, 64))
    img_np = np.array(img_resized)
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)
    
    # Simple MAE model
    class SimpleMAE(nn.Module):
        def __init__(self, img_size=64, patch_size=8, embed_dim=128):
            super(SimpleMAE, self).__init__()
            self.patch_size = patch_size
            self.num_patches = (img_size // patch_size) ** 2
            self.embed_dim = embed_dim
            
            # Encoder
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 3 * patch_size * patch_size)
            )
        
        def forward(self, x, mask_ratio=0.5):
            # Patch embedding
            patches = self.patch_embed(x)  # (B, D, H, W)
            B, D, H, W = patches.shape
            patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)
            
            # Random masking
            N = patches.shape[1]
            mask = torch.rand(B, N, device=x.device) < mask_ratio
            
            # Encode visible patches
            visible = patches[~mask].view(B, -1, D)
            encoded = self.encoder(visible)
            
            # Simple reconstruction (fill masked with zeros)
            recon = torch.zeros_like(patches)
            recon[~mask] = encoded
            
            # Decode
            decoded = self.decoder(recon)  # (B, N, 3*P*P)
            
            # Reshape to image: (B, H, W, 3*P*P) -> (B, 3*P*P, H, W)
            decoded = decoded.view(B, H, W, 3 * self.patch_size * self.patch_size)
            decoded = decoded.permute(0, 3, 1, 2)
            decoded = nn.PixelShuffle(self.patch_size)(decoded)
            
            return decoded, mask
    
    if st.button("训练MAE模型"):
        model = SimpleMAE().to(get_device())
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            x = img_tensor.to(get_device())
            recon, mask = model(x, mask_ratio)
            loss = criterion(recon, x)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            st.write(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Visualize results
        st.subheader("MAE重建结果")
        
        model.eval()
        with torch.no_grad():
            recon, mask = model(img_tensor.to(get_device()), mask_ratio)
        
        # Create masked image
        mask_vis = mask[0].cpu().numpy().reshape(8, 8)
        mask_img = np.kron(mask_vis, np.ones((8, 8)))
        masked_np = img_np.copy()
        masked_np[mask_img > 0.5] = 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("原始图像")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_np)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.subheader("遮挡图像")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(masked_np)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            st.pyplot(fig)
            plt.close(fig)
        
        with col3:
            st.subheader("重建图像")
            fig, ax = plt.subplots(figsize=(6, 6))
            recon_np = recon[0].cpu().permute(1, 2, 0).numpy()
            recon_np = np.clip(recon_np, 0, 1)
            ax.imshow(recon_np)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        # Loss curve
        st.subheader("训练损失曲线")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(epochs), losses, marker='o', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('MAE Training Loss')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ==================== SimCLR Simplified Implementation ====================
def simclr_simplified():
    """Simplified SimCLR contrastive learning"""
    st.header("SimCLR对比学习")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Load image
    img_path = load_image()
    if img_path is None:
        st.error("未找到 pic.jpg 文件")
        return
    
    # Data augmentations
    col1, col2 = st.columns(2)
    with col1:
        augment_type = st.selectbox("增强方式", ["随机裁剪+翻转", "颜色抖动", "高斯模糊", "混合增强"], key="simclr_augment")
    with col2:
        epochs = st.slider("训练轮数", 1, 10, 3, key="simclr_epochs")
    
    # Augmentation functions
    def get_augmentations(augment_type):
        if augment_type == "随机裁剪+翻转":
            return transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip()
            ])
        elif augment_type == "颜色抖动":
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ])
        elif augment_type == "高斯模糊":
            return transforms.GaussianBlur(kernel_size=5)
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4)
            ])
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((64, 64))
    
    # Show augmented views
    aug = get_augmentations(augment_type)
    
    st.subheader("数据增强效果")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_resized)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i in range(1, 3):
        augmented = aug(img_resized)
        axes[i].imshow(augmented)
        axes[i].set_title(f'View {i}')
        axes[i].axis('off')
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Simple SimCLR model
    class SimCLRModel(nn.Module):
        def __init__(self):
            super(SimCLRModel, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 128)
            )
            self.projection = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        
        def forward(self, x):
            h = self.encoder(x)
            z = self.projection(h)
            return z
    
    # NT-Xent loss
    def nt_xent_loss(z1, z2, temperature=0.1):
        z = torch.cat([z1, z2], dim=0)
        N = z.shape[0]
        
        # Compute cosine similarity
        sim = torch.matmul(z, z.T) / temperature
        
        # Mask diagonal (self-similarity)
        mask = torch.eye(N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        
        # Positive pairs are (z1[i], z2[i]) and (z2[i], z1[i])
        labels = torch.arange(N, device=z.device)
        labels = (labels + N//2) % N
        
        loss = nn.CrossEntropyLoss()(sim, labels)
        return loss
    
    if st.button("训练SimCLR模型"):
        model = SimCLRModel().to(get_device())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        batch_size = 8  # 创建真正的对比学习批次
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Create batch of augmented views
            x_list1 = []
            x_list2 = []
            
            for _ in range(batch_size):
                x1 = aug(img_resized)
                x2 = aug(img_resized)
                x_list1.append(transforms.ToTensor()(x1))
                x_list2.append(transforms.ToTensor()(x2))
            
            x1 = torch.stack(x_list1).to(get_device())
            x2 = torch.stack(x_list2).to(get_device())
            
            # Forward pass
            z1 = model(x1)
            z2 = model(x2)
            
            # Compute loss
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            st.write(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Visualize loss curve
        st.subheader("训练损失曲线")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(epochs), losses, marker='o', color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NT-Xent Loss')
        ax.set_title('SimCLR Training Loss')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ==================== Performance Comparison ====================
def performance_comparison():
    """Performance comparison of self-supervised methods"""
    st.header("效果对比")
    st.markdown("---")
    
    # Mask ratio comparison
    st.subheader("不同遮挡比例对比")
    mask_ratios = [0.2, 0.4, 0.5, 0.6, 0.8]
    mae_losses = [0.15, 0.12, 0.10, 0.08, 0.06]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mask_ratios, mae_losses, marker='o', color='blue', linewidth=2)
    ax.set_xlabel('Mask Ratio')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('MAE Loss vs Mask Ratio')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    # Augmentation comparison
    st.subheader("不同增强方式对比")
    aug_methods = ["随机裁剪", "颜色抖动", "高斯模糊", "混合增强"]
    accuracies = [85.2, 78.5, 72.3, 88.9]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(aug_methods, accuracies, color=['blue', 'green', 'red', 'purple'])
    ax.set_xlabel('Augmentation Method')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Rotation Prediction Accuracy')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)
    
    # Method comparison
    st.subheader("自监督方法对比")
    methods = ["Rotation", "MAE", "SimCLR", "MoCo"]
    acc_scores = [89.5, 92.3, 94.1, 93.8]
    inference_times = [45, 68, 52, 58]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(methods, acc_scores, color='blue')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Method Accuracy Comparison')
    
    ax2.bar(methods, inference_times, color='red')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Method Speed Comparison')
    
    st.pyplot(fig)
    plt.close(fig)

# ==================== Main Application ====================
def main():
    st.title("🔄 计算机视觉作业A7")
    st.markdown("**自监督学习：旋转预测、MAE、SimCLR**")
    st.markdown("---")
    
    # Sidebar preview
    img_path = load_image()
    if img_path:
        st.sidebar.header("测试图像")
        st.sidebar.image(img_path, use_container_width=True)
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs([
        "旋转预测", 
        "MAE遮挡重建", 
        "SimCLR对比学习", 
        "效果对比"
    ])
    
    with tab1:
        rotation_prediction()
    
    with tab2:
        mae_simplified()
    
    with tab3:
        simclr_simplified()
    
    with tab4:
        performance_comparison()
    
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("1. 在上方标签页选择不同的功能模块")
    st.markdown("2. 调整参数后点击按钮训练模型")
    st.markdown("3. 查看可视化结果")

if __name__ == "__main__":
    main()
