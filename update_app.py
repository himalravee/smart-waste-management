# new_app.py
import os
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

# ----------------------
# Reduce TensorFlow logging
# ----------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Waste Classifier", layout="wide")

# ----------------------
# Paths
# ----------------------
from keras.models import load_model

MODEL_PATH = "updated_model_tf20.keras"
model = load_model(MODEL_PATH)
CLASS_JSON_PATH = "class_indices.json"

# ----------------------
# Load model & classes
# ----------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_JSON_PATH):
        raise FileNotFoundError(f"Class JSON file not found: {CLASS_JSON_PATH}")
    with open(CLASS_JSON_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]

try:
    model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

try:
    class_names = load_class_names()
    st.info(f"{len(class_names)} classes loaded from JSON")
except Exception as e:
    st.error(f"Unable to load class indices: {e}")
    st.stop()

# ----------------------
# Preprocessing
# ----------------------
def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

# ----------------------
# Inference
# ----------------------
def infer(model, img_arr):
    preds = model.predict(img_arr)
    class_idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds[0]))
    return class_idx, prob

# ----------------------
# App UI
# ----------------------
st.title("Waste Classifier")
st.write("Upload an image or use your camera; the model predicts the waste class and whether it's recyclable.")

# ----------------------
# Session counters
# ----------------------
if "counts" not in st.session_state:
    st.session_state.counts = {name: 0 for name in class_names}
if "total_recyclable" not in st.session_state:
    st.session_state.total_recyclable = 0
if "total_non_recyclable" not in st.session_state:
    st.session_state.total_non_recyclable = 0

# Default recyclable heuristics
def default_recyclable(names):
    heuristics = ["plastic", "paper", "cardboard", "glass", "metal", "can", "aluminium", "tin"]
    return set([n for n in names if any(tok in n.lower() for tok in heuristics)])

if "recyclable_set" not in st.session_state:
    st.session_state.recyclable_set = default_recyclable(class_names)

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header("Session Counts & Settings")
st.sidebar.subheader("Mark Recyclable Classes")
with st.sidebar.expander("Edit Recyclable Classes"):
    cols = st.columns(2)
    for i, name in enumerate(class_names):
        col = cols[i % 2]
        checked = name in st.session_state.recyclable_set
        new = col.checkbox(name, value=checked, key=f"rc_{i}")
        if new:
            st.session_state.recyclable_set.add(name)
        elif name in st.session_state.recyclable_set:
            st.session_state.recyclable_set.remove(name)

st.sidebar.markdown("---")
st.sidebar.subheader("Current Session Counts")
counts_df = pd.DataFrame.from_dict(st.session_state.counts, orient="index", columns=["Count"]).sort_values("Count", ascending=False)
st.sidebar.write(counts_df)
st.sidebar.write(f"Total recyclable: **{st.session_state.total_recyclable}**")
st.sidebar.write(f"Total non-recyclable: **{st.session_state.total_non_recyclable}**")

if st.sidebar.button("Reset Counts"):
    st.session_state.counts = {name: 0 for name in class_names}
    st.session_state.total_recyclable = 0
    st.session_state.total_non_recyclable = 0
    st.sidebar.success("Counts reset")

# ----------------------
# Image input
# ----------------------
col_upload, col_camera = st.columns(2)
uploaded = col_upload.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if "camera_open" not in st.session_state:
    st.session_state.camera_open = False

camera_button_label = "Close Camera" if st.session_state.camera_open else "Open Camera"
if col_camera.button(camera_button_label):
    st.session_state.camera_open = not st.session_state.camera_open

camera_image = col_camera.camera_input("Take a picture with your camera") if st.session_state.camera_open else None

# ----------------------
# Prediction & display
# ----------------------
image = None
if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read()))
elif camera_image is not None:
    image = Image.open(io.BytesIO(camera_image.read()))

if image:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Selected image", use_container_width=True)

    with col2:
        with st.spinner("Predicting..."):
            arr = preprocess_image(image)
            class_idx, prob = infer(model, arr)
            predicted_name = class_names[class_idx]
            recyclable_flag = predicted_name in st.session_state.recyclable_set

        st.subheader("Prediction")
        st.markdown(f"**Class:** `{predicted_name}`")
        st.markdown(f"**Confidence:** {prob*100:.1f}%")
        st.markdown(f"**Recyclable:** {'Yes' if recyclable_flag else 'No'}")

        # Update session counters
        st.session_state.counts[predicted_name] += 1
        if recyclable_flag:
            st.session_state.total_recyclable += 1
        else:
            st.session_state.total_non_recyclable += 1

# ----------------------
# Session counts table & CSV
# ----------------------
st.write("---")
st.markdown("## Session Counts")
counts_df = pd.DataFrame.from_dict(st.session_state.counts, orient="index", columns=["Count"]).reset_index()
counts_df.columns = ["Class", "Count"]
counts_df["Recyclable"] = counts_df["Class"].apply(lambda x: "Yes" if x in st.session_state.recyclable_set else "No")

totals = pd.DataFrame({
    "Class": ["TOTAL Recyclable", "TOTAL Non-Recyclable"],
    "Count": [st.session_state.total_recyclable, st.session_state.total_non_recyclable],
    "Recyclable": ["Yes", "No"]
})

display_df = pd.concat([counts_df, totals], ignore_index=True)
st.dataframe(display_df.sort_values("Count", ascending=False), use_container_width=True)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="waste_counts.csv", mime="text/csv")

st.caption("Counts are session-only. For permanent storage, connect a database like SQLite.")



