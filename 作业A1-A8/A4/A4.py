"""
Computer Vision Assignment A4
Linear Regression, Classification, and Gradient Descent
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os

# Page configuration
st.set_page_config(
    page_title="A4: Regression & Classification",
    page_icon="📊",
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

# ==================== Linear Regression ====================
def linear_regression():
    """Linear regression with least squares fitting"""
    st.header("最小二乘线性回归")
    st.markdown("---")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_points = st.slider("数据点数量", 20, 200, 100)
    with col2:
        noise_level = st.slider("噪声水平", 0.1, 2.0, 0.5)
    
    if st.button("生成数据并拟合"):
        # Generate synthetic data
        np.random.seed(42)
        X = np.linspace(0, 10, n_points)
        y = 2.5 * X + 3.0 + np.random.normal(0, noise_level, n_points)
        
        # Convert to numpy arrays
        X_np = X.reshape(-1, 1)
        y_np = y
        
        # Least squares fitting
        X_design = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_design, y_np, rcond=None)[0]
        slope = beta[1]
        intercept = beta[0]
        
        # Predictions
        y_pred = slope * X + intercept
        
        # Calculate MSE
        mse = np.mean((y - y_pred) ** 2)
        
        # Plot results
        st.subheader("拟合结果")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, color='blue', alpha=0.6, label='Data points')
        ax.plot(X, y_pred, color='red', linewidth=2, label=f'Fitted line: y = {slope:.2f}x + {intercept:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Linear Regression Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        st.info(f"拟合直线: y = {slope:.4f}x + {intercept:.4f}")
        st.info(f"均方误差(MSE): {mse:.4f}")

# ==================== KNN and Linear Classifier ====================
def knn_classifier():
    """KNN and Linear classifier on CIFAR-10 data"""
    st.header("KNN与线性分类器")
    st.markdown("---")
    
    @st.cache_data
    def load_cifar10():
        """Load a small subset of CIFAR-10 for demonstration"""
        try:
            # Try to load full CIFAR-10
            X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, as_frame=False)
            # Take a subset for speed
            idx = np.random.choice(len(X), 2000, replace=False)
            X = X[idx]
            y = y[idx]
            return X, y
        except Exception:
            # Fallback: create synthetic data
            np.random.seed(42)
            X = np.random.randn(500, 3072)
            y = np.random.randint(0, 10, 500)
            return X, y
    
    # Load data
    X, y = load_cifar10()
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        k_value = st.slider("KNN的K值", 1, 20, 5)
    with col2:
        classifier_type = st.selectbox("分类器类型", ["KNN", "线性分类器"])
    
    if st.button("训练分类器"):
        if classifier_type == "KNN":
            # KNN classifier with data normalization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.subheader(f"KNN分类结果 (K={k_value})")
            st.info(f"测试集准确率: {accuracy:.4f}")
            
            # Visualize some predictions
            st.subheader("预测结果示例")
            sample_indices = np.random.choice(len(X_test), 6, replace=False)
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))
            
            for i, idx in enumerate(sample_indices):
                ax = axes[i//3, i%3]
                # Ensure image data is in correct range for visualization
                img_data = X_test[idx].reshape(32, 32, 3)
                if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                    img = img_data
                else:
                    img = img_data.astype(np.float32) / 255
                ax.imshow(img)
                ax.set_title(f"Pred: {y_pred[idx]}, True: {y_test[idx]}")
                ax.axis('off')
            
            st.pyplot(fig)
            plt.close(fig)
        
        else:
            # Linear classifier
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.subheader("线性分类器结果")
            st.info(f"测试集准确率: {accuracy:.4f}")
            
            # Visualize learned templates
            st.subheader("学习到的模板图像")
            weights = clf.coef_
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            
            for i in range(10):
                ax = axes[i//5, i%5]
                template = weights[i].reshape(32, 32, 3)
                # Normalize for visualization
                template = (template - template.min()) / (template.max() - template.min())
                ax.imshow(template)
                ax.set_title(f"Class {i}")
                ax.axis('off')
            
            st.pyplot(fig)
            plt.close(fig)
    
    # K value comparison
    st.subheader("不同K值的准确率对比")
    if st.button("对比不同K值"):
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]
        accuracies = []
        
        # Normalize data for KNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            accuracies.append(accuracy_score(y_test, y_pred))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, accuracies, marker='o', color='blue', linewidth=2)
        ax.set_xlabel('K value')
        ax.set_ylabel('Accuracy')
        ax.set_title('KNN Accuracy vs K Value')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        st.pyplot(fig)
        plt.close(fig)

# ==================== Gradient Descent Visualization ====================
def gradient_descent():
    """Gradient descent visualization with different optimizers"""
    st.header("梯度下降算法可视化")
    st.markdown("---")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        learning_rate = st.slider("学习率", 0.01, 0.5, 0.1)
    with col2:
        momentum = st.slider("动量系数", 0.0, 0.99, 0.9)
    with col3:
        n_iterations = st.slider("迭代次数", 50, 500, 200)
    
    loss_type = st.selectbox("选择损失函数", ["MSE", "交叉熵"])
    
    if st.button("运行梯度下降"):
        # Create a simple quadratic function for visualization
        def mse_loss(w):
            """Simple quadratic loss function"""
            return (w - 3) ** 2 + 2
        
        def mse_grad(w):
            """Gradient of MSE loss"""
            return 2 * (w - 3)
        
        def cross_entropy_loss(w):
            """Simplified cross-entropy-like loss"""
            return -np.log(1 / (1 + np.exp(-w))) - np.log(1 / (1 + np.exp(w - 6)))
        
        def cross_entropy_grad(w):
            """Gradient of simplified cross-entropy"""
            return -1 / (1 + np.exp(-w)) + 1 / (1 + np.exp(w - 6))
        
        if loss_type == "MSE":
            loss_func = mse_loss
            grad_func = mse_grad
        else:
            loss_func = cross_entropy_loss
            grad_func = cross_entropy_grad
        
        # SGD without momentum
        w = 0.0
        sgd_losses = []
        sgd_weights = []
        
        for _ in range(n_iterations):
            grad = grad_func(w)
            w -= learning_rate * grad
            sgd_losses.append(loss_func(w))
            sgd_weights.append(w)
        
        # SGD with momentum
        w_momentum = 0.0
        v = 0.0
        momentum_losses = []
        momentum_weights = []
        
        for _ in range(n_iterations):
            grad = grad_func(w_momentum)
            v = momentum * v + learning_rate * grad
            w_momentum -= v
            momentum_losses.append(loss_func(w_momentum))
            momentum_weights.append(w_momentum)
        
        # Plot loss curves
        st.subheader("损失曲线对比")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(n_iterations), sgd_losses, label='SGD', color='blue')
        ax.plot(range(n_iterations), momentum_losses, label='SGD with Momentum', color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'{loss_type} Loss vs Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        # Plot weight convergence
        st.subheader("参数收敛过程")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(n_iterations), sgd_weights, label='SGD', color='blue')
        ax.plot(range(n_iterations), momentum_weights, label='SGD with Momentum', color='red')
        ax.axhline(y=3, color='green', linestyle='--', label='True value')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight')
        ax.set_title('Weight Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        st.info(f"SGD最终权重: {sgd_weights[-1]:.4f}")
        st.info(f"带动量SGD最终权重: {momentum_weights[-1]:.4f}")
        st.info(f"理论最优值: 3.0")

# ==================== Main Application ====================
def main():
    st.title("📊 计算机视觉作业A4")
    st.markdown("**最小二乘回归、分类器与梯度下降**")
    st.markdown("---")
    
    # Sidebar preview
    img_path = load_image()
    if img_path:
        st.sidebar.header("图像预览")
        st.sidebar.image(img_path, use_container_width=True)
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["最小二乘线性回归", "KNN/线性分类器", "梯度下降可视化"])
    
    with tab1:
        linear_regression()
    
    with tab2:
        knn_classifier()
    
    with tab3:
        gradient_descent()
    
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("1. 在左侧标签页选择不同的功能模块")
    st.markdown("2. 调整参数后点击按钮执行")
    st.markdown("3. 查看可视化结果")

if __name__ == "__main__":
    main()
