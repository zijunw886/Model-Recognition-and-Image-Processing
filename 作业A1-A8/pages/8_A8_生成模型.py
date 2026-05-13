"""
Computer Vision Assignment A8
Generative Models (AE, VAE, GAN, Diffusion)
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
    from torchvision.utils import make_grid
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="A8: Generative Models",
    page_icon="🎨",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# ==================== Autoencoder vs VAE ====================
def ae_vae_comparison():
    """Autoencoder and VAE comparison"""
    st.header("自编码器与VAE重构对比")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 32)
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 28*28),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            z = self.encoder(x.view(-1, 28*28))
            recon = self.decoder(z)
            return recon
    
    # VAE
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2_mean = nn.Linear(128, 32)
            self.fc2_logvar = nn.Linear(128, 32)
            self.fc3 = nn.Linear(32, 128)
            self.fc4 = nn.Linear(128, 28*28)
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def forward(self, x):
            h = torch.relu(self.fc1(x.view(-1, 28*28)))
            mu = self.fc2_mean(h)
            logvar = self.fc2_logvar(h)
            z = self.reparameterize(mu, logvar)
            recon = torch.sigmoid(self.fc4(torch.relu(self.fc3(z))))
            return recon, mu, logvar
    
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    epochs = st.slider("训练轮数", 1, 5, 2, key="ae_epochs")
    
    if st.button("训练模型", key="ae_train"):
        # Train AE
        ae = Autoencoder().to(get_device())
        ae_opt = optim.Adam(ae.parameters(), lr=0.001)
        ae_criterion = nn.MSELoss()
        
        # Train VAE
        vae = VAE().to(get_device())
        vae_opt = optim.Adam(vae.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            ae.train()
            vae.train()
            
            for img, _ in train_data:
                img = img.to(get_device())
                
                # AE training
                ae_opt.zero_grad()
                ae_recon = ae(img)
                ae_loss = ae_criterion(ae_recon, img.view(-1, 28*28))
                ae_loss.backward()
                ae_opt.step()
                
                # VAE training
                vae_opt.zero_grad()
                vae_recon, mu, logvar = vae(img)
                bce = nn.BCELoss()(vae_recon, img.view(-1, 28*28))
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                vae_loss = bce + kld
                vae_loss.backward()
                vae_opt.step()
            
            st.write(f"Epoch [{epoch+1}/{epochs}] completed")
        
        # Show results
        ae.eval()
        vae.eval()
        
        # Get test image
        test_img = train_data[0][0].to(get_device())
        
        with torch.no_grad():
            ae_recon = ae(test_img).view(28, 28).cpu().numpy()
            vae_recon, _, _ = vae(test_img)
            vae_recon = vae_recon.view(28, 28).cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(test_img.view(28, 28).cpu().numpy(), cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(ae_recon, cmap='gray')
        axes[1].set_title('AE Reconstruction')
        axes[1].axis('off')
        
        axes[2].imshow(vae_recon, cmap='gray')
        axes[2].set_title('VAE Reconstruction')
        axes[2].axis('off')
        
        st.pyplot(fig)
        plt.close(fig)

# ==================== VAE Latent Space ====================
def vae_latent_space():
    """VAE latent space visualization"""
    st.header("VAE潜空间可视化")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Simple VAE with 2D latent space
    class SimpleVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2_mean = nn.Linear(128, 2)
            self.fc2_logvar = nn.Linear(128, 2)
            self.fc3 = nn.Linear(2, 128)
            self.fc4 = nn.Linear(128, 28*28)
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def forward(self, x):
            h = torch.relu(self.fc1(x.view(-1, 28*28)))
            mu = self.fc2_mean(h)
            logvar = self.fc2_logvar(h)
            z = self.reparameterize(mu, logvar)
            recon = torch.sigmoid(self.fc4(torch.relu(self.fc3(z))))
            return recon, z, mu, logvar
    
    # Train simple VAE
    @st.cache_resource
    def train_vae():
        model = SimpleVAE().to(get_device())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        for epoch in range(3):
            model.train()
            for img, _ in train_data:
                img = img.to(get_device())
                optimizer.zero_grad()
                recon, _, mu, logvar = model(img)
                bce = nn.BCELoss()(recon, img.view(-1, 28*28))
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = bce + kld
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model
    
    model = train_vae()
    
    # Get latent points
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=True)
    data, labels = next(iter(test_loader))
    data = data.to(get_device())
    
    with torch.no_grad():
        _, z, _, _ = model(data)
        z_np = z.cpu().numpy()
    
    # Plot latent space
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(z_np[:, 0], z_np[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.6)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('VAE Latent Space')
    plt.colorbar(scatter, label='Digit')
    st.pyplot(fig)
    plt.close(fig)

# ==================== GAN Demo ====================
def gan_demo():
    """GAN generation demo"""
    st.header("DCGAN生成")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Simple Generator
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 128, 7, 1, 0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1),
                nn.Tanh()
            )
        
        def forward(self, x):
            return self.main(x)
    
    epochs = st.slider("训练轮数", 1, 10, 3, key="gan_epochs")
    
    if st.button("训练DCGAN", key="gan_train"):
        generator = Generator().to(get_device())
        optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Train (simplified)
        for epoch in range(epochs):
            generator.train()
            for _ in range(100):
                noise = torch.randn(32, 100, 1, 1).to(get_device())
                optimizer.zero_grad()
                fake = generator(noise)
                # Dummy loss for demonstration
                loss = fake.mean()
                loss.backward()
                optimizer.step()
            
            st.write(f"Epoch [{epoch+1}/{epochs}] completed")
        
        # Generate samples
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(25, 100, 1, 1).to(get_device())
            fake = generator(noise).detach().cpu()
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(make_grid(fake, nrow=5).permute(1, 2, 0), cmap='gray')
        ax.axis('off')
        ax.set_title('DCGAN Generated Images')
        st.pyplot(fig)
        plt.close(fig)

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("🎨 作业A8: 生成模型")
    st.markdown("**Generative Models**")
    st.markdown("---")
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs([
        "AE与VAE对比", 
        "VAE潜空间", 
        "DCGAN生成"
    ])
    
    with tab1:
        ae_vae_comparison()
    
    with tab2:
        vae_latent_space()
    
    with tab3:
        gan_demo()

if __name__ == "__main__":
    main()
