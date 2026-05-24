# 🧮 Handwritten Math Equation Solver

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white)

An interactive web application built with Streamlit and PyTorch that uses Convolutional Neural Networks (CNNs) to recognize and solve handwritten mathematical digits and operators. Developed as a core academic project for NIT.

## ✨ Features
* **Digit Recognition:** Custom CNN model (`digit_model.pth`) trained to recognize handwritten numbers.
* **Operator Recognition:** Secondary model (`operator_model.pth`) trained to classify mathematical symbols (+, -, *, /).
* **Interactive UI:** A clean Streamlit frontend where users can upload images of equations and get real-time computed results.

## 🏗️ Project Architecture
1. **`CNN_NIT_Project.ipynb`:** The core notebook detailing the dataset preprocessing, CNN architecture, and training pipeline.
2. **`helper_functions.py`:** Utility scripts for image processing (bounding boxes, contour detection) using OpenCV before passing them to the PyTorch models.
3. **`app.py`:** The Streamlit application that handles the UI and real-time model inference.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/your-github-username/Handwritten-Equation-Solver.git](https://github.com/your-github-username/Handwritten-Equation-Solver.git)
cd Handwritten-Equation-Solver
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```