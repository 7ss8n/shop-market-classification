# app.py
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

st.set_page_config(page_title="shop market image classification", layout="centered")
st.title("shop market image classification")

# -----------------------
# Device selection (MPS -> CUDA -> CPU)
# -----------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

st.caption(f"Using device: **{DEVICE}**")

# -----------------------
# Classes & transforms
# -----------------------
CLASSES = ["C", "B" , "A"]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# -----------------------
# Model loader (cached)
# -----------------------
@st.cache_resource
def load_model(weights_path: str):
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048,128),
        nn.ReLU(),
        nn.Linear(128,3)
    )
    state = torch.load(weights_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        st.warning(f"Loaded with non-strict mode due to: {e}")
        model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model

WEIGHTS_PATH = "catVsDog.h5"

import os
if not os.path.exists(WEIGHTS_PATH):
    st.error(f"File `{WEIGHTS_PATH}` not found in the current directory.")
    st.stop()

model = load_model(WEIGHTS_PATH)

# -----------------------
# Inference helper
# -----------------------
def predict(image: Image.Image):
    img_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().ravel()
    pred_idx = int(probs.argmax())
    return CLASSES[pred_idx], float(probs[pred_idx]), probs

# -----------------------
# UI — ONLY UPLOAD IMAGE
# -----------------------
st.subheader("Upload an image")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    pred_label, pred_conf, probs = predict(image)
    st.success(f"Prediction: **{pred_label}**  (confidence: {pred_conf:.3f})")
    st.progress(min(max(pred_conf, 0.0), 1.0))

    st.subheader("Class probabilities")
    st.write({cls: float(p) for cls, p in zip(CLASSES, probs)})

    st.caption("Note: Using ImageNet normalization. Ensure it matches your training pipeline.")
