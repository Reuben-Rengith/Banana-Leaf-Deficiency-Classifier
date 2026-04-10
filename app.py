from __future__ import annotations

import os
import tempfile
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model

from config import IMG_SIZE_EFF, IMG_SIZE_MOB, active_classes
from feature_fusion import build_meta_features, extract_gap_batch, predict_probs_batch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

st.set_page_config(page_title="Banana Leaf Deficiency Classifier", layout="centered")
st.title("Banana Leaf Nutrient Deficiency Classifier")
st.write("Upload a banana leaf image to predict nutrient deficiency.")

@st.cache_resource
def load_models():
    model_m = tf.keras.models.load_model(
        os.path.join(RESULTS_DIR, "mobilenet_unified_final.h5"),
        compile=False,
    )
    model_e = tf.keras.models.load_model(
        os.path.join(RESULTS_DIR, "efficientnet_unified_final.h5"),
        compile=False,
    )
    gbm = joblib.load(os.path.join(RESULTS_DIR, "meta_gbm.pkl"))
    enc_m = Model(model_m.input, model_m.get_layer("gap").output)
    enc_e = Model(model_e.input, model_e.get_layer("gap").output)
    return model_m, model_e, enc_m, enc_e, gbm

def predict_image(temp_path: str):
    class_names = active_classes()
    model_m, model_e, enc_m, enc_e, gbm = load_models()

    pm = predict_probs_batch(model_m, [temp_path], IMG_SIZE_MOB)
    pe = predict_probs_batch(model_e, [temp_path], IMG_SIZE_EFF)
    gm = extract_gap_batch(enc_m, [temp_path], IMG_SIZE_MOB)
    ge = extract_gap_batch(enc_e, [temp_path], IMG_SIZE_EFF)

    X = build_meta_features(gm, ge, pm, pe)
    proba = gbm.predict_proba(X)[0]
    idx = np.argsort(proba)[::-1]
    return [(class_names[i], float(proba[i])) for i in idx]

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    with st.spinner("Predicting..."):
        preds = predict_image(temp_path)

    os.unlink(temp_path)

    st.subheader("Top 3 Predictions")
    for label, score in preds[:3]:
        st.write(f"**{label}**: {score*100:.2f}%")

    st.subheader("Confidence Scores")
    st.bar_chart({k: v for k, v in preds})