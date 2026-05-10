# 🌍 GeoDisasterAINet

<div align="center">

## An Explainable Deep Ensemble Framework for Real-Time Urban and Rural Disaster Classification and Resilience

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-WebApp-black?style=for-the-badge&logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br>

### 🌪️ Classifying Cyclones · Earthquakes · Floods · Wildfires

### 🚀 96.37% Accuracy using Deep Ensemble CNNs + XGBoost + Multiclass SVM + Explainable AI (LIME & Grad-CAM)

</div>

---

# 🔍 Overview

**GeoDisasterAINet** is an advanced multi-stage AI framework for real-time disaster classification using Deep Learning, Machine Learning, and Explainable AI.

The system classifies:

- 🌀 Cyclone
- 🏚️ Earthquake
- 🌊 Flood
- 🔥 Wildfire

from disaster images with high accuracy and explainability support.

This framework supports **United Nations Sustainable Development Goal 11 (SDG 11)** by helping build disaster-resilient urban and rural communities.

---

# ⚡ Multi-Stage Framework

| Stage | Components | Purpose |
|------|-------------|----------|
| Stage 1 | Integrated CNN Ensembles | Feature Extraction & Baseline Classification |
| Stage 2 | CNN + XGBoost | Feature Selection & Accuracy Improvement |
| Stage 3 | CNN + XGBoost + Multiclass SVM | Decision Boundary Optimization |
| Stage 4 | LIME + Grad-CAM | Explainability & Visual Interpretation |

---

# 📄 Published Research

This project is based on the IEEE Access research article:

> **GeoDisasterAINet: An Explainable Deep Ensemble Framework for Real-Time Urban and Rural Disaster Classification and Resilience**
>
> Akella S. Narasimha Raju, Seelam Sreekanth, Ranjith Kumar Gatla, et al.
>
> IEEE Access, Volume 13, 2025
>
> DOI: `10.1109/ACCESS.2025.3574451`

---

# 🏆 Key Results

## 📊 Model Performance

| Model | Training Accuracy | Testing Accuracy | Precision | Recall | F1-Score |
|------|--------------------|------------------|-----------|--------|----------|
| ERI-2025 (Stage 1) | 97% | 94% | 96% | 95% | 95% |
| ERI-2025 + XGBoost (Stage 2) | 98% | 95.37% | 97% | 96% | 96% |
| ERI-2025 + XGBoost + SVM (Stage 3) | 99% | **96.37%** | 98.5% | 97.5% | 97% |

---

## 🎯 Per-Class Performance

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 🌀 Cyclone | 98% | 95% | 96% |
| 🏚️ Earthquake | 96% | 97% | 96% |
| 🌊 Flood | 94% | 95% | 94% |
| 🔥 Wildfire | 97% | 96% | 97% |

---

# 🏗️ Architecture

```text
Input Image (224×224 RGB)
        │
        ▼
┌─────────────────────────────────────┐
│      Integrated CNN Ensemble        │
│                                     │
│  ┌─────────────┐                    │
│  │EfficientB7  │                    │
│  └─────────────┘                    │
│                                     │
│  ┌─────────────┐                    │
│  │ ResNet50    │                    │
│  └─────────────┘                    │
│                                     │
│  ┌─────────────┐                    │
│  │ InceptionV3 │                    │
│  └─────────────┘                    │
│                                     │
│  Concatenate + Flatten Features     │
└────────────────┬────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │    XGBoost     │
        │ Feature Select │
        └───────┬────────┘
                │
                ▼
        ┌────────────────┐
        │ Multiclass SVM │
        │ Classification │
        └───────┬────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Predicted Disaster Type │
    │ Confidence Score        │
    │ Grad-CAM Heatmap        │
    │ LIME Explanation        │
    └─────────────────────────┘
```

---

# 🧠 CNN Ensemble Variants

| Ensemble | Architecture Components |
|----------|--------------------------|
| ERI-2025 | EfficientNetB7 + ResNet50 + InceptionV3 |
| DRI-2025 | DenseNet201 + ResNet50 + InceptionV3 |
| DE-2025 | DenseNet201 + EfficientNetB7 |

---

# 📁 Project Structure

```text
GeoDisasterAINet/
│
├── 📂 Models/
│   └── ensemble_hybrid.h5
│
├── 📂 static/
│   └── results/
│       ├── uploads/
│       ├── gradcam/
│       ├── heatmaps/
│       └── lime/
│
├── 📂 templates/
│   ├── index.html
│   ├── signin.html
│   ├── signup.html
│   ├── home.html
│   ├── result.html
│   ├── graphs.html
│   ├── Notebook.html
│   ├── Notebook2.html
│   └── Notebook3.html
│
├── 📂 Notebooks/
│   ├── ERI_-_2025.ipynb
│   ├── DRI_-_2025.ipynb
│   └── DE.ipynb
│
├── app.py
├── signup.db
├── requirements.txt
├── Flowchart.txt
└── README.md
```

---

# ⚙️ Installation

## ✅ Prerequisites

- Python 3.10.9
- Anaconda (Recommended)
- Git

---

## 🔽 Step 1 — Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/GeoDisasterAINet.git
cd GeoDisasterAINet
```

---

## 🐍 Step 2 — Create Virtual Environment

```bash
conda create -n geodisaster python=3.10.9
conda activate geodisaster
```

---

## 📦 Step 3 — Install Dependencies

```bash
pip install tensorflow==2.10.1
pip install keras==2.10.0
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install Flask==2.2.2
pip install Werkzeug==2.2.2
pip install scikit-learn==1.0.2
pip install xgboost==2.1.2
pip install lime==0.2.0.1
pip install shap==0.46.0
pip install imbalanced-learn==0.10.1
pip install matplotlib==3.7.0
pip install scipy==1.10.1
pip install joblib==1.4.2
pip install protobuf==3.20.3
pip install jinja2==3.1.2
pip install MarkupSafe==2.1.1
pip install itsdangerous==2.0.1
pip install opencv-python
```

Or install directly:

```bash
pip install -r requirements.txt
```

---

## 🤖 Step 4 — Add Trained Model

Place the trained model file here:

```text
Models/ensemble_hybrid.h5
```

---

# 🚀 Running the Web App

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# 🌐 Application Flow

```text
Landing Page (/)
    ├── Sign In (/signin)
    ├── Sign Up (/signup)
            └── Dashboard (/home)
                    └── Upload Image
                            └── /predict
                                   ├── Predicted Class
                                   ├── Confidence Score
                                   ├── Grad-CAM Heatmap
                                   └── LIME Explanation
```

---

# 📓 Notebooks

| Notebook | Model | Description |
|----------|-------|-------------|
| ERI_-_2025.ipynb | ERI-2025 | EfficientNetB7 + ResNet50 + InceptionV3 |
| DRI_-_2025.ipynb | DRI-2025 | DenseNet201 + ResNet50 + InceptionV3 |
| DE.ipynb | DE-2025 | DenseNet201 + EfficientNetB7 |

---

# 📊 Dataset

## 🌍 Natural Disaster Image Dataset

Dataset Source:

```text
https://www.kaggle.com/datasets/alex1994/natural-disaster-image-dataset
```

---

# 🔬 Explainability (XAI)

## 🔥 Grad-CAM

- CNN Attention Visualization
- Heatmap Generation
- Disaster Region Highlighting

## 🟡 LIME

- Human-readable explanations
- Local prediction interpretation
- Positive region highlighting

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|-------------|
| Deep Learning | TensorFlow 2.10, Keras 2.10 |
| CNN Architectures | EfficientNetB7, ResNet50, InceptionV3, DenseNet201 |
| ML Models | XGBoost 2.1.2, Scikit-learn SVM |
| Data Balancing | SMOTE |
| Explainability | LIME, Grad-CAM |
| Backend | Flask |
| Image Processing | OpenCV, Pillow |
| Database | SQLite3 |
| Visualization | Matplotlib |
| Language | Python 3.10.9 |

---

# 📖 Citation

```bibtex
@article{raju2025geodisasterainet,
  title     = {GeoDisasterAINet: An Explainable Deep Ensemble Framework for Real-Time Urban and Rural Disaster Classification and Resilience},
  author    = {Narasimha Raju, Akella S. and Sreekanth, Seelam and Gatla, Ranjith Kumar and Rajababu, M. and Gireesh Kumar, Devineni and Flah, Aymen and El-Bayeh, Claude Ziad and El-Nagdy, Khaled A. and Alzaed, Ali},
  journal   = {IEEE Access},
  volume    = {13},
  pages     = {97944--97974},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2025.3574451}
}
```

---

# 👨‍💻 Author

## Shalu Pandey

### Under the Guidance of Akella S. Narasimha Raju

📧 Student Email: `shalupandey129247@gmail.com`

📧 Guide Email: `akella.raju@gmail.com`

---

# 🤝 Contributing

Contributions are welcome!

Please read `CONTRIBUTING.md` before submitting pull requests.

---

# 📜 License

This project is licensed under the MIT License.

See `LICENSE` for details.

---

# 🙏 Acknowledgements

Special thanks to:

- Institute of Aeronautical Engineering, Hyderabad
- National Disaster Management Authority (NDMA), India
- Taif University
- European Union REFRESH Project
- TensorFlow
- Keras
- Scikit-Learn
- XGBoost
- LIME
- Open-source AI research community

---

<div align="center">

# ⭐ If this project helped you, please give it a star! ⭐

## ❤️ Made with passion in support of SDG 11: Sustainable Cities and Communities

</div>
