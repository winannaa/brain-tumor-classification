import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    return load_model("model/best_brain_tumor_model_MobileNetV2.h5")

model = load_my_model()

# =========================
# LABEL KELAS
# =========================
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# =========================
# PREPROCESSING (MobileNetV2)
# =========================
def preprocess_image(image):
    img = image.resize((224, 224))  # ukuran standar MobileNetV2
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# UI
# =========================
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload gambar MRI untuk mendeteksi jenis tumor otak.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

# =========================
# MAIN
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="MRI Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Menganalisis gambar..."):
            img = preprocess_image(image)

            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # =========================
            # HASIL
            # =========================
            st.success(f"🧾 Prediction: {class_names[predicted_class]}")
            st.info(f"📊 Confidence: {confidence:.2f}")

            # =========================
            # DETAIL PROBABILITAS
            # =========================
            st.write("### 📊 Detail Probabilities:")
            for i, label in enumerate(class_names):
                st.write(f"{label}: {prediction[0][i]:.2f}")
