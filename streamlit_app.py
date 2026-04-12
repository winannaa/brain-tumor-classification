import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# LOAD MODEL TFLITE
# =========================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# LABEL
# =========================
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = img / 255.0   # ⚠️ nanti kita cek ini kalau hasil aneh
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# UI
# =========================
st.title("🧠 Brain Tumor Classification")
st.write("Upload MRI image to detect brain tumor.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Predict"):
        img = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        pred = np.argmax(output)
        conf = np.max(output)

        st.success(f"Prediction: {class_names[pred]}")
        st.write(f"Confidence: {conf:.2f}")
