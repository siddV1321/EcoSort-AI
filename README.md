
# ♻️ EcoSort AI
### AI-Powered Waste Classification System

> Waste Intelligence. Powered by AI.

EcoSort AI is a computer vision project that uses Artificial Intelligence to identify and classify waste images into six categories. It demonstrates how deep learning can be applied to environmental sustainability and waste segregation.

---

## 👨‍💻 About the Developer

**Siddharth Verma**  
Class 10th D | Academic Session 2026–27

I am a student interested in Artificial Intelligence, Machine Learning, Computer Vision and building technology-driven solutions to real-world problems.

EcoSort AI is one of my projects exploring the practical application of AI. Through this project, I developed an understanding of image classification, transfer learning, model training, evaluation and application development.

---

## 🌱 About the Project

Waste segregation is an important part of effective waste management. Incorrectly mixed waste makes recycling difficult and contributes to environmental pollution.

EcoSort AI explores how AI-based image recognition can help identify waste materials automatically.

The application allows users to upload an image of waste and receive an AI-generated classification.

### Waste Categories

| Category | Description |
|---|---|
| 📦 Cardboard | Cardboard packaging and boxes |
| 🍾 Glass | Glass bottles and containers |
| 🥫 Metal | Metal cans and objects |
| 📄 Paper | Paper sheets and products |
| 🧴 Plastic | Plastic bottles and packaging |
| 🗑️ Trash | General waste |

---

## ✨ Features

- AI-powered waste image classification
- Six predefined waste categories
- Image upload and prediction interface
- MobileNetV2 deep learning architecture
- Transfer learning
- PyTorch model integration
- Interactive Streamlit web application
- Separate Orange Data Mining experiment

---

## 🧠 AI Project Cycle

EcoSort AI follows the five stages of the AI Project Cycle.

### 1. Problem Scoping
Identifying the challenges of waste segregation and defining the objective of automatic waste classification.

### 2. Data Acquisition
Using the Trash Type Image Dataset available on Kaggle.

### 3. Data Exploration
Examining and preparing waste images, organising categories and preparing data for model training.

### 4. Modelling
Training a MobileNetV2-based image classification model using transfer learning and PyTorch.

### 5. Evaluation
Evaluating the model using validation data and measuring its classification performance.

---

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning and model training |
| Torchvision | Image transformations and model architecture |
| MobileNetV2 | Image classification model |
| Streamlit | Web application interface |
| Orange Data Mining | Separate ML experiment |
| GitHub | Version control and project hosting |
| Kaggle | Dataset source |

---

## 📊 Model Performance

### PyTorch Model

| Metric | Result |
|---|---|
| Architecture | MobileNetV2 |
| Learning Technique | Transfer Learning |
| Best Validation Accuracy | 84% |
| Best Epoch | 8 |
| Classes | 6 |

The model achieved its best recorded validation accuracy of 84% at Epoch 8.

**Note:** Validation accuracy is not independent test accuracy.

### Orange Data Mining Experiment

A separate experiment was conducted using Orange Data Mining.

| Algorithm | Classification Accuracy |
|---|---:|
| Logistic Regression | 78.0% |
| Random Forest | 64.4% |

This experiment helped explore machine learning model comparison and evaluation.

The Orange experiment and PyTorch model are separate workflows.

---

## 🔄 How It Works

```text
       Waste Image
            ↓
      Image Upload
            ↓
   Image Preprocessing
            ↓
    MobileNetV2 Model
            ↓
   Classification Output
            ↓
    Predicted Category
```

The uploaded image is converted into the format required by the trained model. The model processes the image and produces class probabilities. The category with the highest predicted probability is displayed in the application.

---

## 🚀 Installation & Setup

### Requirements

- Python 3.10 or compatible version
- pip
- Git
- Trained model file

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd EcoSort-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Trained Model

Place your trained MobileNetV2 model in the project directory.

Example:

```text
best_model.pth
```

Ensure the filename matches the configuration in `app.py`.

### 5. Run EcoSort AI

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📁 Project Structure

```text
EcoSort-AI/
│
├── app.py
├── best_model.pth
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── github-repository.png
│   ├── ecosort-ui.png
│   ├── training-log.png
│   └── orange-workflow.png
│
└── documentation/
```

---

## 🔮 Future Scope

- Real-time camera-based waste classification
- Mobile application integration
- Improved model accuracy
- Larger and more diverse datasets
- Smart waste bin integration
- Automated waste segregation systems
- Recycling recommendations

---

## 📚 Dataset

**Trash Type Image Dataset**

Source: Kaggle

https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset

---

## 🔗 Project Links

| Resource | Link |
|---|---|
| GitHub Repository | YOUR_GITHUB_URL |
| Live Application | YOUR_STREAMLIT_URL |
| Google Drive | YOUR_GOOGLE_DRIVE_URL |
| Presentation | YOUR_GAMMA_PRESENTATION_URL |
| Dataset | Kaggle Dataset |

---

## 🎯 Learning Outcomes

Through EcoSort AI, I explored:

- Computer Vision
- Image Classification
- Transfer Learning
- Deep Learning
- PyTorch
- Model Evaluation
- Streamlit Application Development
- Data Exploration
- AI Project Cycle
- GitHub Project Management

---

## 🌍 Vision

EcoSort AI demonstrates how Artificial Intelligence can contribute to environmental awareness and smarter waste management.

The project represents my exploration of technology, innovation and practical problem-solving.

> **A Step Towards a Cleaner Tomorrow.**

---

## 👨‍🎓 Project Information

**Project:** EcoSort AI  
**Developer:** Siddharth Verma  
**Class:** 10th D  
**Academic Session:** 2026–27  

Made with Python, AI and a vision for sustainability. ♻️
