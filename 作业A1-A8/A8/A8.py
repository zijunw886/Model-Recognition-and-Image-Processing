"""
Computer Vision Assignment A8
Autoencoder, VAE, GAN, and Diffusion Models
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
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import diffusers
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="A8: AE/VAE/GAN/Diffusion",
    page_icon="🎨",
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

def check_gpu_status():
    """Check and display GPU status"""
    if torch.cuda.is_available():
        return f"✅ GPU可用: {torch.cuda.get_device_name(0)}"
    else:
        return "⚠️ 未检测到GPU，使用CPU训练（较慢）"

# ==================== Autoencoder and VAE Comparison ====================
def ae_vae_comparison():
    """Autoencoder and VAE reconstruction comparison"""
    st.header("自编码器与VAE重构对比")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        dataset_type = st.selectbox("数据集", ["MNIST", "Fashion-MNIST"], key="ae_dataset")
    with col2:
        epochs = st.slider("训练轮数", 1, 10, 2, key="ae_epochs")  # 减少默认轮数
    
    # Load dataset - keep data in [0, 1] range for BCE loss
    transform = transforms.Compose([
        transforms.ToTensor()  # Output in [0, 1]
    ])
    
    if dataset_type == "MNIST":
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)  # 增大batch size
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    
    # Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(28*28, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 32)
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 28*28),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = x.view(-1, 28*28)
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z
    
    # VAE
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()
            self.fc1 = nn.Linear(28*28, 256)
            self.fc2_mean = nn.Linear(256, 32)
            self.fc2_logvar = nn.Linear(256, 32)
            self.fc3 = nn.Linear(32, 256)
            self.fc4 = nn.Linear(256, 28*28)
        
        def encode(self, x):
            h = torch.relu(self.fc1(x.view(-1, 28*28)))
            return self.fc2_mean(h), self.fc2_logvar(h)
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def decode(self, z):
            h = torch.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h))
        
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, z, mu, logvar
    
    # VAE loss
    def vae_loss(recon_x, x, mu, logvar):
        BCE = nn.BCELoss(reduction='sum')(recon_x, x.view(-1, 28*28))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    if st.button("训练模型", key="ae_train"):
        # Train Autoencoder
        ae = Autoencoder().to(get_device())
        ae_optimizer = optim.Adam(ae.parameters(), lr=1e-3)
        ae_criterion = nn.MSELoss()
        
        # Train VAE
        vae = VAE().to(get_device())
        vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        
        ae_losses = []
        vae_losses = []
        
        for epoch in range(epochs):
            ae.train()
            vae.train()
            ae_total_loss = 0
            vae_total_loss = 0
            
            for data, _ in train_loader:
                data = data.to(get_device())
                
                # Autoencoder training
                ae_optimizer.zero_grad()
                ae_recon, _ = ae(data)
                ae_loss = ae_criterion(ae_recon, data.view(-1, 28*28))
                ae_loss.backward()
                ae_optimizer.step()
                ae_total_loss += ae_loss.item()
                
                # VAE training
                vae_optimizer.zero_grad()
                vae_recon, _, mu, logvar = vae(data)
                loss = vae_loss(vae_recon, data, mu, logvar)
                loss.backward()
                vae_optimizer.step()
                vae_total_loss += loss.item()
            
            ae_losses.append(ae_total_loss / len(train_loader))
            vae_losses.append(vae_total_loss / len(train_loader))
            st.write(f"Epoch [{epoch+1}/{epochs}], AE Loss: {ae_losses[-1]:.4f}, VAE Loss: {vae_losses[-1]:.4f}")
        
        # Show results
        st.subheader("重构结果对比")
        
        # Get test samples
        test_samples, _ = next(iter(test_loader))
        test_samples = test_samples.to(get_device())
        
        ae.eval()
        vae.eval()
        with torch.no_grad():
            ae_recon, _ = ae(test_samples)
            vae_recon, _, _, _ = vae(test_samples)
        
        # Visualize
        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        
        for i in range(8):
            axes[0, i].imshow(test_samples[i][0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title('Input')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(ae_recon[i].cpu().numpy().reshape(28, 28), cmap='gray')
            axes[1, i].set_title('AE')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(vae_recon[i].cpu().numpy().reshape(28, 28), cmap='gray')
            axes[2, i].set_title('VAE')
            axes[2, i].axis('off')
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Loss curves
        st.subheader("训练损失曲线")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(epochs), ae_losses, marker='o', label='Autoencoder', color='blue')
        ax.plot(range(epochs), vae_losses, marker='o', label='VAE', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ==================== VAE Latent Space Visualization ====================
def vae_latent_space():
    """VAE latent space visualization and interpolation"""
    st.header("VAE潜空间可视化与交互")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch 库")
        return
    
    # Load trained VAE or train quickly
    @st.cache_resource
    def get_trained_vae():
        class VAE(nn.Module):
            def __init__(self):
                super(VAE, self).__init__()
                self.fc1 = nn.Linear(28*28, 256)
                self.fc2_mean = nn.Linear(256, 2)  # 2D latent space
                self.fc2_logvar = nn.Linear(256, 2)
                self.fc3 = nn.Linear(2, 256)
                self.fc4 = nn.Linear(256, 28*28)
            
            def encode(self, x):
                h = torch.relu(self.fc1(x.view(-1, 28*28)))
                return self.fc2_mean(h), self.fc2_logvar(h)
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                h = torch.relu(self.fc3(z))
                return torch.sigmoid(self.fc4(h))
            
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                return recon, z, mu, logvar
        
        # Quick training - keep data in [0, 1] range for BCE loss
        transform = transforms.Compose([
            transforms.ToTensor()  # Output in [0, 1]
        ])
        
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=512, shuffle=True)  # 增大batch size
        
        vae = VAE().to(get_device())
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        
        def vae_loss(recon_x, x, mu, logvar):
            BCE = nn.BCELoss(reduction='sum')(recon_x, x.view(-1, 28*28))
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD
        
        for epoch in range(3):  # 减少训练轮数
            vae.train()
            for data, _ in train_loader:
                data = data.to(get_device())
                optimizer.zero_grad()
                recon, _, mu, logvar = vae(data)
                loss = vae_loss(recon, data, mu, logvar)
                loss.backward()
                optimizer.step()
        
        vae.eval()
        return vae
    
    vae = get_trained_vae()
    
    # Get latent space points - keep data in [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor()  # Output in [0, 1]
    ])
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)
    test_samples, test_labels = next(iter(test_loader))
    test_samples = test_samples.to(get_device())
    
    with torch.no_grad():
        _, z, _, _ = vae(test_samples)
        z_np = z.cpu().numpy()
        labels_np = test_labels.numpy()
    
    # Plot latent space
    st.subheader("VAE二维潜空间")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(z_np[:, 0], z_np[:, 1], c=labels_np, cmap='tab10', alpha=0.6)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('VAE Latent Space (MNIST)')
    plt.colorbar(scatter, ticks=range(10), label='Digit')
    st.pyplot(fig)
    plt.close(fig)
    
    # Latent interpolation
    st.subheader("潜空间插值")
    col1, col2 = st.columns(2)
    
    with col1:
        z1_x = st.slider("起点X", -4.0, 4.0, -2.0, key="z1_x")
        z1_y = st.slider("起点Y", -4.0, 4.0, -2.0, key="z1_y")
    
    with col2:
        z2_x = st.slider("终点X", -4.0, 4.0, 2.0, key="z2_x")
        z2_y = st.slider("终点Y", -4.0, 4.0, 2.0, key="z2_y")
    
    num_steps = st.slider("插值步数", 5, 15, 10, key="interp_steps")
    
    if st.button("生成插值序列", key="interp_button"):
        z1 = torch.tensor([z1_x, z1_y], device=get_device()).float()
        z2 = torch.tensor([z2_x, z2_y], device=get_device()).float()
        
        interpolated_images = []
        for t in np.linspace(0, 1, num_steps):
            z_interp = (1 - t) * z1 + t * z2
            with torch.no_grad():
                img = vae.decode(z_interp.unsqueeze(0))
                img = img.cpu().numpy().reshape(28, 28)
                interpolated_images.append(img)
        
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'{i+1}')
        
        st.pyplot(fig)
        plt.close(fig)

# ==================== GAN and Diffusion Models ====================
def gan_diffusion():
    """GAN and Diffusion model experiments"""
    st.header("GAN与扩散模型实验")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["DCGAN生成", "扩散模型文生图"])
    
    with tab1:
        if not TORCH_AVAILABLE:
            st.error("需要安装 PyTorch 库")
            return
        
        # DCGAN implementation
        class Generator(nn.Module):
            def __init__(self, latent_dim=100):
                super(Generator, self).__init__()
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.main(x)
        
        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
                self.main = nn.Sequential(
                    nn.Conv2d(1, 128, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256, 1, 7, 1, 0, bias=False),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.main(x)
        
        epochs = st.slider("训练轮数", 1, 20, 5, key="gan_epochs")
        
        if st.button("训练DCGAN并生成样本", key="gan_train"):
            device = get_device()
            latent_dim = 100
            
            generator = Generator(latent_dim).to(device)
            discriminator = Discriminator().to(device)
            
            criterion = nn.BCELoss()
            optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=128, shuffle=True)  # 增大batch size
            
            for epoch in range(epochs):
                for i, (real_images, _) in enumerate(train_loader):
                    real_images = real_images.to(device)
                    batch_size = real_images.size(0)
                    
                    # Train discriminator
                    optimizer_D.zero_grad()
                    
                    # Real images
                    label = torch.full((batch_size,), 1., dtype=torch.float, device=device)
                    output = discriminator(real_images).view(-1)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    
                    # Fake images
                    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                    fake_images = generator(noise)
                    label.fill_(0.)
                    output = discriminator(fake_images.detach()).view(-1)
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    optimizer_D.step()
                    
                    # Train generator
                    optimizer_G.zero_grad()
                    label.fill_(1.)
                    output = discriminator(fake_images).view(-1)
                    errG = criterion(output, label)
                    errG.backward()
                    optimizer_G.step()
                
                st.write(f"Epoch [{epoch+1}/{epochs}] completed")
            
            # Generate samples
            generator.eval()
            with torch.no_grad():
                noise = torch.randn(25, latent_dim, 1, 1, device=device)
                fake = generator(noise).detach().cpu()
            
            # Show generated images
            st.subheader("DCGAN生成样本")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(make_grid(fake, nrow=5).permute(1, 2, 0), cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
    
    with tab2:
        if not DIFFUSERS_AVAILABLE:
            st.error("需要安装 diffusers 库")
            return
        
        # Diffusion model text-to-image
        st.subheader("扩散模型文生图")
        
        prompt = st.text_input("输入提示词", "a cat wearing sunglasses", key="diff_prompt")
        negative_prompt = st.text_input("负面提示词", "blurry, low quality", key="diff_neg_prompt")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_inference_steps = st.slider("采样步数", 10, 50, 20, key="diff_steps")
        with col2:
            guidance_scale = st.slider("引导系数", 1.0, 10.0, 7.5, key="diff_guidance")
        with col3:
            seed = st.slider("随机种子", 0, 1000, 42, key="diff_seed")
        
        if st.button("生成图像", key="diff_generate"):
            try:
                # Load lightweight model
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                pipe = pipe.to(get_device())
                
                # Generate image
                generator = torch.Generator(device=get_device()).manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
                
                # Show image
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(image)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"生成失败: {str(e)}")
                st.info("提示：如果显存不足，可以尝试使用更小的模型如 'CompVis/stable-diffusion-v1-4'")

# ==================== Main Application ====================
def main():
    st.title("🎨 计算机视觉作业A8")
    st.markdown("**自编码器、VAE、GAN与扩散模型**")
    st.markdown("---")
    
    # Sidebar info
    st.sidebar.header("系统状态")
    if TORCH_AVAILABLE:
        st.sidebar.info(check_gpu_status())
    
    # Sidebar preview
    img_path = load_image()
    if img_path:
        st.sidebar.header("图像预览")
        st.sidebar.image(img_path, use_container_width=True)
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs([
        "AE与VAE对比", 
        "VAE潜空间", 
        "GAN与扩散模型"
    ])
    
    with tab1:
        ae_vae_comparison()
    
    with tab2:
        vae_latent_space()
    
    with tab3:
        gan_diffusion()
    
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("1. 在上方标签页选择不同的功能模块")
    st.markdown("2. 调整参数后点击按钮执行")
    st.markdown("3. 查看可视化结果")

if __name__ == "__main__":
    main()
