import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 256

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("unet_isic2016.keras", compile=False)

st.title("Skin Lesion Segmentation with U-Net")
st.caption("Educational demo trained on ISIC 2016. Not a diagnostic tool.")
uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    small = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)))
    prob = load_model().predict(small[None].astype("float32") / 255.0, verbose=0)[0, ..., 0]
    mask = (prob > thr).astype(np.uint8)
    overlay = small.copy()
    overlay[mask == 1] = (0.5 * overlay[mask == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    c1, c2, c3 = st.columns(3)
    c1.image(small, caption="Input")
    c2.image(mask * 255, caption="Predicted mask")
    c3.image(overlay, caption="Overlay")
    st.write(f"Predicted lesion area: {mask.mean() * 100:.1f}% of the image")
