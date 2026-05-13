"""
Computer Vision Course - Assignment Portal
统一作业导航首页
"""

import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="计算机视觉作业合集",
    page_icon="🎓",
    layout="wide"
)

# Home page styling
st.markdown("""
    <style>
    .card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #007bff;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #007bff;
        margin-bottom: 10px;
    }
    .card-desc {
        color: #6c757d;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main content
st.title("🎓 计算机视觉课程作业合集")
st.subheader("模型识别与图像处理")
st.markdown("---")

# Course information
st.info("""
**课程名称**: 模型识别与图像处理  
**作业数量**: 8个  
**技术栈**: Python + Streamlit + OpenCV + PyTorch + NumPy + Matplotlib
""")

# Assignment cards
st.header("作业列表")
st.markdown("点击卡片快速跳转到对应作业页面")

assignments = [
    {
        "id": "1",
        "title": "A1 - 图像颜色空间与插值",
        "description": "RGB与HSV颜色空间转换，最近邻与双线性插值实现，支持图像缩放、旋转、拉伸操作",
        "link": "1_A1_图像颜色空间"
    },
    {
        "id": "2",
        "title": "A2 - 图像滤波与边缘检测",
        "description": "均值滤波、高斯滤波、中值滤波对比，Sobel、Prewitt、Laplacian边缘检测",
        "link": "2_A2_图像滤波"
    },
    {
        "id": "3",
        "title": "A3 - 特征检测与图像匹配",
        "description": "Canny边缘检测、Harris角点检测、SIFT特征点检测、图像匹配与全景拼接",
        "link": "3_A3_特征检测"
    },
    {
        "id": "4",
        "title": "A4 - 机器学习基础",
        "description": "最小二乘线性回归、KNN分类器、线性分类器、梯度下降算法可视化",
        "link": "4_A4_机器学习"
    },
    {
        "id": "5",
        "title": "A5 - 深度学习基础",
        "description": "HOG+词袋模型+SVM分类、反向传播算法演示、LeNet-5训练、ResNet性能对比",
        "link": "5_A5_深度学习"
    },
    {
        "id": "6",
        "title": "A6 - 目标检测与分割",
        "description": "FCN语义分割、R-CNN目标检测、Mask R-CNN实例分割、性能对比",
        "link": "6_A6_目标检测"
    },
    {
        "id": "7",
        "title": "A7 - 自监督学习",
        "description": "图像旋转预测、MAE遮挡重建、SimCLR对比学习、效果对比",
        "link": "7_A7_自监督学习"
    },
    {
        "id": "8",
        "title": "A8 - 生成模型",
        "description": "自编码器与VAE重构对比、VAE潜空间可视化、DCGAN生成、扩散模型文生图",
        "link": "8_A8_生成模型"
    }
]

# Display assignment cards in 2 columns
cols = st.columns(2)
for i, assignment in enumerate(assignments):
    with cols[i % 2]:
        # Create clickable card using markdown with link
        st.markdown(f"""
            <div class="card" onclick="window.location.href='./{assignment['link']}'">
                <div class="card-title">📌 {assignment['title']}</div>
                <div class="card-desc">{assignment['description']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add button as fallback
        st.button(f"进入 {assignment['id']}", key=f"btn_{assignment['id']}", 
                  on_click=lambda link=assignment['link']: st.switch_page(f"pages/{link}.py"))

# Footer
st.markdown("---")
st.markdown("### 📝 作业说明")
st.markdown("""
- 每个作业都是独立的Streamlit应用
- 支持参数调整和实时预览
- 所有图表坐标轴标注为英文，界面文字为中文
- 内置测试图片可直接运行，无需额外上传
""")

st.markdown("### 🚀 运行方式")
st.markdown("""
```bash
cd "C:\\Users\\claire.wei\\Desktop\\模型识别与图像处理\\作业A1-A8"
streamlit run Home.py
```
""")
