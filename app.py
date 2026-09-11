# app.py
import os
import torch
import operator
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as F, ToPILImage
import streamlit as st

# Ensure the temporary directory exists to prevent runtime crashes
os.makedirs("temp", exist_ok=True)

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

# ------------------ PREDICTION LOGIC ------------------

class_names_digit = [str(i) for i in range(10)]
class_names_op = ['add', 'div', 'mul', 'subtract']

# Safe evaluation mapping
op_execution_map = {
    'add': ('+', operator.add),
    'div': ('÷', operator.truediv),
    'mul': ('×', operator.mul),
    'subtract': ('-', operator.sub)
}

@st.cache_resource(show_spinner=False)
def load_digit_model():
    model = CNN_DIGITS(1, 10)
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), 'digit_model.pth'),
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_operator_model():
    model = CNN_OPERATORS(1, 4)
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), 'operator_model.pth'),
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

def predict_image(image_path, model, transform):
    try:
        image = read_image(image_path)
        image = transform(image).unsqueeze(0)
        with torch.inference_mode():
            pred = model(image)
        return torch.argmax(torch.softmax(pred, dim=1), dim=1).item()
    except Exception as e:
        st.error(f"Failed to process {image_path}: {e}")
        return None

# ------------------ HEAVY UI / STREAMLIT APP ------------------

st.set_page_config(
    page_title="Neural Math Vision", 
    page_icon="👁️‍🗨️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global Theming */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #020617 100%);
        font-family: 'Poppins', sans-serif;
        color: #f8fafc;
    }
    
    /* Hide default elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Typography */
    .title-glow {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        line-height: 1.2;
    }
    .subtitle {
        text-align: center;
        font-weight: 300;
        color: #94a3b8;
        font-size: 1.25rem;
        margin-bottom: 4rem;
        letter-spacing: 0.5px;
    }
    .column-header {
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
    }

    /* Outer Uploader Container Styling */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem 1rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stFileUploader"]:hover {
        transform: translateY(-8px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 0 15px rgba(56, 189, 248, 0.2);
        background: rgba(255, 255, 255, 0.04);
    }
    
    /* FIX: Inner Dropzone box */
    div[data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 2px dashed #94a3b8 !important;
    }
    
    /* Main text and limit text inside dropzone */
    div[data-testid="stFileUploadDropzone"] > div > div > span,
    div[data-testid="stFileUploadDropzone"] > div > div > small {
        color: #1e293b !important; 
        font-weight: 600 !important;
    }
    
    /* CRITICAL FIX: The actual 'Browse Files'/'Upload' button and all its nested text/icons */
    div[data-testid="stFileUploadDropzone"] button,
    div[data-testid="stFileUploadDropzone"] button * {
        background-color: #f1f5f9 !important; /* Light gray button so it looks clickable */
        color: #0f172a !important; /* Pure dark slate for text */
        border-color: #64748b !important;
        font-weight: 800 !important;
        fill: #0f172a !important; /* Forces the SVG icon to be dark */
    }
    
    /* Hover state for 'Browse Files'/'Upload' button */
    div[data-testid="stFileUploadDropzone"] button:hover,
    div[data-testid="stFileUploadDropzone"] button:hover * {
        background-color: #e2e8f0 !important;
        color: #000000 !important;
        fill: #000000 !important;
        border-color: #0f172a !important;
    }

    /* Primary Execute Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: #020617 !important;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.25rem;
        font-weight: 800;
        border-radius: 50px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.3);
        margin-top: 2rem;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(139, 92, 246, 0.5), 0 0 20px rgba(59, 130, 246, 0.5);
        color: #020617 !important;
    }

    /* Output Glass Card */
    .result-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 30px;
        padding: 4rem 2rem;
        text-align: center;
        margin-top: 4rem;
        position: relative;
        overflow: hidden;
        animation: pulse-glow 4s infinite alternate;
    }
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.1); }
        100% { box-shadow: 0 0 40px rgba(139, 92, 246, 0.3); }
    }
    
    .math-output {
        font-family: 'JetBrains Mono', monospace;
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(to right, #ffffff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255,255,255,0.2);
        line-height: 1;
    }
    
    hr {
        border-color: rgba(255,255,255,0.05);
        margin: 3rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR CONFIG ------------------
with st.sidebar:
    st.markdown("### 🧠 Neural Engine")
    st.markdown("Status: **Online** 🟢")
    st.markdown("---")
    st.markdown("#### Architectures")
    st.caption("Digit Pipeline: CNN (3 Conv Layers, ReLU, MaxPool)")
    st.caption("Operator Pipeline: CNN (3 Conv Layers, BatchNorm, MaxPool)")
    st.markdown("---")
    st.markdown("#### Transforms")
    st.caption("- Grayscale Conversion")
    st.caption("- Anti-aliased 28x28 Resizing")
    st.caption("- Float32 Tensor Inversion")

# ------------------ MAIN INTERFACE ------------------
st.markdown("<div class='title-glow'>Neural Vision Evaluator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload segmented digit and operator images to compute via deep learning pipelines.</div>", unsafe_allow_html=True)

# Load Models
with st.spinner("Initializing Deep Learning Pipelines..."):
    digit_model = load_digit_model()
    operator_model = load_operator_model()

# Input Layout (Labels included to fix Streamlit console warnings)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("<div class='column-header'>Left Operand</div>", unsafe_allow_html=True)
    first_digit = st.file_uploader("Upload Left Operand", type=["png", "jpg", "jpeg"], key="digit1", label_visibility="collapsed")

with col2:
    st.markdown("<div class='column-header'>Operation</div>", unsafe_allow_html=True)
    operator_img = st.file_uploader("Upload Operator", type=["png", "jpg", "jpeg"], key="operator", label_visibility="collapsed")

with col3:
    st.markdown("<div class='column-header'>Right Operand</div>", unsafe_allow_html=True)
    second_digit = st.file_uploader("Upload Right Operand", type=["png", "jpg", "jpeg"], key="digit2", label_visibility="collapsed")


# Execution Trigger
if st.button("EXECUTE TENSOR PIPELINE"):
    if first_digit and operator_img and second_digit:
        
        # Save temp files for vision processing
        files_map = {
            "temp/temp_digit1.png": first_digit,
            "temp/temp_operator.png": operator_img,
            "temp/temp_digit2.png": second_digit
        }
        
        for path, file_obj in files_map.items():
            with open(path, "wb") as f:
                f.write(file_obj.read())

        with st.spinner("Extracting features and running inference..."):
            # Inference Calls
            num1 = predict_image("temp/temp_digit1.png", digit_model, digit_transform)
            op_idx = predict_image("temp/temp_operator.png", operator_model, op_transform)
            num2 = predict_image("temp/temp_digit2.png", digit_model, digit_transform)
            
        if None not in (num1, op_idx, num2):
            op_name = class_names_op[op_idx]
            op_symbol, op_func = op_execution_map[op_name]
            
            try:
                # Mathematical Evaluation
                result = op_func(num1, num2)
                formatted_result = int(result) if result == int(result) else round(result, 4)
                
                # Visual Celebration
                st.balloons()
                
                # Render Massive Glass UI Result
                st.markdown(
                    f"""
                    <div class='result-card'>
                        <div style="color: #94a3b8; font-size: 1.2rem; font-weight: 600; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 1rem;">Computed Output</div>
                        <div class='math-output'>{num1} {op_symbol} {num2} = {formatted_result}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Post-Execution Vision Analysis
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center; color: #cbd5e1; font-weight: 300;'>Vision Analysis Log</h3>", unsafe_allow_html=True)
                
                v_col1, v_col2, v_col3 = st.columns(3)
                v_col1.image("temp/temp_digit1.png", caption=f"Classified as: {num1}", use_container_width=True)
                v_col2.image("temp/temp_operator.png", caption=f"Classified as: {op_symbol}", use_container_width=True)
                v_col3.image("temp/temp_digit2.png", caption=f"Classified as: {num2}", use_container_width=True)
                
            except ZeroDivisionError:
                st.error("⚠️ Exception: Neural network classified a division by zero. Execution halted.")
    else:
        st.warning("⚠️ INCOMPLETE TENSORS: Please provide all three image matrices before execution.")