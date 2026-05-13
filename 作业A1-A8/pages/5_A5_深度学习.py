"""
Computer Vision Assignment A5
Deep Learning Basics
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="A5: Deep Learning Basics",
    page_icon="🧠",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# ==================== Backpropagation Demo ====================
def backpropagation_demo():
    """Backpropagation visualization"""
    st.header("反向传播算法演示")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # Simple neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleNN().to(get_device())
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Generate dummy data
    X = torch.randn(32, 10).to(get_device())
    y = torch.randint(0, 2, (32,)).to(get_device())
    
    # Training loop with visualization
    losses = []
    grad_norms = []
    
    for epoch in range(50):
        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Track gradient norms
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item()
        grad_norms.append(total_grad_norm)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(grad_norms)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Magnitude')
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)

# ==================== CNN Training (LeNet-5) ====================
def cnn_training_demo():
    """LeNet-5 training demo"""
    st.header("CNN模型训练与测试")
    st.markdown("---")
    
    if not TORCH_AVAILABLE:
        st.error("需要安装 PyTorch")
        return
    
    # LeNet-5
    class LeNet5(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pool2 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool1(x)
            x = torch.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.view(-1, 16 * 4 * 4)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)
    
    epochs = st.slider("训练轮数", 1, 5, 2, key="cnn_epochs")
    
    if st.button("训练LeNet-5", key="cnn_train"):
        model = LeNet5().to(get_device())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(get_device()), target.to(get_device())
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            train_losses.append(total_loss / len(train_loader))
            
            # Evaluate
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(get_device()), target.to(get_device())
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_accuracies.append(correct / len(test_loader.dataset))
            st.write(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {test_accuracies[-1]:.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label='Training Loss', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('LeNet-5 Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        # Show sample predictions
        st.subheader("测试样本预测")
        data, target = next(iter(test_loader))
        data, target = data[:8].to(get_device()), target[:8]
        output = model(data)
        pred = output.argmax(dim=1).cpu()
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(data[i][0].cpu().numpy(), cmap='gray')
            ax.set_title(f'Pred: {pred[i]}, True: {target[i]}')
            ax.axis('off')
        
        st.pyplot(fig)
        plt.close(fig)

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("🧠 作业A5: 深度学习基础")
    st.markdown("**Deep Learning Basics**")
    st.markdown("---")
    
    # Tab selection
    tab1, tab2 = st.tabs([
        "反向传播演示", 
        "LeNet-5训练"
    ])
    
    with tab1:
        backpropagation_demo()
    
    with tab2:
        cnn_training_demo()

if __name__ == "__main__":
    main()
