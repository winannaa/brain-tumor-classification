import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

@st.cache_resource
def load_model():
    session = ort.InferenceSession("model/model.onnx")
    return session

session = load_model()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Predict"):
        img = preprocess_image(image)

        output = session.run([output_name], {input_name: img})[0]

        pred = np.argmax(output)
        conf = np.max(output)

        st.success(f"Prediction: {class_names[pred]}")
