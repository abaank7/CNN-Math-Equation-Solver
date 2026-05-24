 # app.py
import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as F, ToPILImage
import streamlit as st
import matplotlib.pyplot as plt




# ------------------ MODELS ------------------

class CNN_DIGITS(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, out_features=output)
        )
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x


class CNN_OPERATORS(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, out_features=output)
        )
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x
    


# ------------------ TRANSFORMS ------------------

digit_transform = transforms.Compose([
    transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),
    transforms.Resize((28,28), antialias=True),
    transforms.Grayscale(num_output_channels=1),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Lambda(lambda x: 1.0 - x)
])

class FastOperatorTransform:
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = ToPILImage()(image)
        image = F.resize(image, size=(28,28), antialias=True)
        image = F.to_grayscale(image, 1)
        image = F.to_tensor(image)
        image = 1.0 - image
        return image

op_transform = FastOperatorTransform()




# ------------------ PREDICTION FUNCTIONS ------------------

class_names_digit = [str(i) for i in range(10)]
class_names_op = ['add', 'div', 'mul', 'subtract']
class_names_op_dict = {'add': '+', 'div': '/', 'mul': '*', 'subtract': '-'}

@st.cache_resource
def load_digit_model():
    model = CNN_DIGITS(1, 10)
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), 'digit_model.pth'),
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

@st.cache_resource
def load_operator_model():
    model = CNN_OPERATORS(1, 4)
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), 'operator_model.pth'),
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

def Predict_Digit(image_path, model, transform):
    image = read_image(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.inference_mode():
        pred = model(image)
    pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1).item()
    return pred_label

def Predict_Operator(image_path, model, transform):
    image = read_image(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.inference_mode():
        pred = model(image)
    pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1).item()
    return pred_label



# ------------------ STREAMLIT APP ------------------


st.set_page_config(page_title="Expression Evaluator", page_icon="🧮", layout="centered")

# Custom CSS for better visuals
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }

        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-size: 18px;
        padding: 0.5em 2em;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧮 Math Expression Evaluator (Arithmatic)")

st.write("Upload **two digits** and an **operator image** to evaluate an expression like `5 * 2`")

digit_model = load_digit_model()
operator_model = load_operator_model()

# Use columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    first_digit = st.file_uploader("First Digit", type=["png", "jpg", "jpeg"], key="digit1")
with col2:
    operator = st.file_uploader("Operator", type=["png", "jpg", "jpeg"], key="operator")
with col3:
    second_digit = st.file_uploader("Second Digit", type=["png", "jpg", "jpeg"], key="digit2")

st.markdown("---")

if st.button("🚀 Predict Expression"):

    if first_digit and operator and second_digit:
        with open("temp/temp_digit1.png", "wb") as f:
            f.write(first_digit.read())
        with open("temp/temp_operator.png", "wb") as f:
            f.write(operator.read())
        with open("temp/temp_digit2.png", "wb") as f:
            f.write(second_digit.read())

        num1 = Predict_Digit("temp/temp_digit1.png", digit_model, digit_transform)
        op_idx = Predict_Operator("temp/temp_operator.png", operator_model, op_transform)
        num2 = Predict_Digit("temp/temp_digit2.png", digit_model, digit_transform)

        op_symbol = class_names_op_dict[class_names_op[op_idx]]
        result = eval(f"{num1}{op_symbol}{num2}")

        st.success(f"✅ **Prediction:** `{num1} {op_symbol} {num2} = {result}`")

        st.markdown("### 📷 Uploaded Images")
        img_col1, img_col2, img_col3 = st.columns(3)
        with img_col1:
            st.image("temp/temp_digit1.png", caption=f"Digit: {num1}", use_container_width=True)
        with img_col2:
            st.image("temp/temp_operator.png", caption=f"Operator: {op_symbol}", use_container_width=True)
        with img_col3:
            st.image("temp/temp_digit2.png", caption=f"Digit: {num2}", use_container_width=True)

    else:
        st.warning("⚠️ Please upload **all three images** before predicting.")

