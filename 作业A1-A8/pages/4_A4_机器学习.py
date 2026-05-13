"""
Computer Vision Assignment A4
Machine Learning Basics
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Page configuration
st.set_page_config(
    page_title="A4: Machine Learning Basics",
    page_icon="📊",
    layout="wide"
)

# Path to root directory for image loading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== Linear Regression ====================
def linear_regression_demo():
    """Linear regression with visualization"""
    st.header("最小二乘线性回归")
    st.markdown("---")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("数据点数量", 20, 100, 50, key="lr_n")
    with col2:
        noise = st.slider("噪声水平", 0.0, 2.0, 0.5, key="lr_noise")
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    true_slope = 2.5
    true_intercept = 1.0
    y = true_slope * X + true_intercept + np.random.normal(0, noise, n_samples)
    
    # Linear regression using least squares
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    slope = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    intercept = y_mean - slope * X_mean
    
    # Predictions
    y_pred = slope * X + intercept
    
    # Calculate MSE
    mse = np.mean((y - y_pred) ** 2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, label='Data points', color='blue', alpha=0.6)
    ax.plot(X, y_pred, label=f'Fitted line: y = {slope:.2f}x + {intercept:.2f}', color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Linear Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    st.subheader("拟合结果")
    st.write(f"真实参数: slope={true_slope}, intercept={true_intercept}")
    st.write(f"估计参数: slope={slope:.4f}, intercept={intercept:.4f}")
    st.write(f"均方误差 (MSE): {mse:.4f}")

# ==================== KNN and Linear Classifier ====================
def knn_classifier_demo():
    """KNN and Linear classifier on CIFAR-10"""
    st.header("KNN/线性分类器")
    st.markdown("---")
    
    @st.cache_resource
    def load_cifar10():
        try:
            X, y = fetch_openml('CIFAR_10_small', version=1, return_X_y=True, as_frame=False)
            X = X / 255.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except:
            # Generate synthetic data
            np.random.seed(42)
            X = np.random.rand(1000, 3072)
            y = np.random.randint(0, 10, 1000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = load_cifar10()
    
    # KNN Classifier
    st.subheader("KNN分类器")
    k_values = st.multiselect("选择K值", [1, 3, 5, 7, 9, 11], [3, 5], key="knn_k")
    
    if st.button("训练并评估KNN", key="knn_train"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = []
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            results.append((k, acc))
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([str(k) for k, _ in results], [acc for _, acc in results], color='skyblue')
        ax.set_xlabel('K Value')
        ax.set_ylabel('Accuracy')
        ax.set_title('KNN Accuracy vs K Value')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        for k, acc in results:
            st.write(f"K={k}: 准确率 = {acc:.4f}")
    
    # Linear Classifier
    st.subheader("线性分类器")
    if st.button("训练线性分类器", key="linear_train"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        st.write(f"线性分类器准确率: {acc:.4f}")
        
        # Visualize learned templates
        st.subheader("学到的模板图像")
        weights = clf.coef_
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        for i, ax in enumerate(axes.flat):
            template = weights[i].reshape(32, 32, 3)
            template = (template - template.min()) / (template.max() - template.min())
            ax.imshow(template)
            ax.set_title(class_names[i])
            ax.axis('off')
        
        st.pyplot(fig)
        plt.close(fig)

# ==================== Gradient Descent Visualization ====================
def gradient_descent_demo():
    """Gradient descent visualization"""
    st.header("梯度下降算法可视化")
    st.markdown("---")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider("学习率", 0.01, 0.5, 0.1, key="gd_lr")
    with col2:
        momentum = st.slider("动量系数", 0.0, 0.9, 0.0, key="gd_momentum")
    
    # Define loss function (parabola)
    def loss(w):
        return (w - 3) ** 2 + 2
    
    def gradient(w):
        return 2 * (w - 3)
    
    # Gradient descent
    w = 0.0
    v = 0.0  # momentum velocity
    losses = []
    weights = []
    
    for _ in range(20):
        losses.append(loss(w))
        weights.append(w)
        grad = gradient(w)
        v = momentum * v - learning_rate * grad
        w = w + v
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss landscape
    w_range = np.linspace(-2, 8, 100)
    ax1.plot(w_range, loss(w_range), label='Loss Function', color='blue')
    ax1.scatter(weights, losses, color='red', s=50, zorder=5)
    ax1.plot(weights, losses, 'r--', label='GD Path')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Landscape and GD Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss over iterations
    ax2.plot(range(len(losses)), losses, marker='o', color='green')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Iterations')
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)
    
    st.write(f"最终权重: {w:.4f}, 最终损失: {loss(w):.4f}")

# ==================== Main Application ====================
def main():
    # Back to home button
    if st.button("🏠 返回首页", key="back_home"):
        st.switch_page("Home.py")
    
    # Title
    st.title("📊 作业A4: 机器学习基础")
    st.markdown("**Machine Learning Basics**")
    st.markdown("---")
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs([
        "线性回归", 
        "KNN/线性分类器", 
        "梯度下降可视化"
    ])
    
    with tab1:
        linear_regression_demo()
    
    with tab2:
        knn_classifier_demo()
    
    with tab3:
        gradient_descent_demo()

if __name__ == "__main__":
    main()
