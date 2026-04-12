import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

@st.cache_resource
def load_my_model():
    model = load_model('best_brain_tumor_model_MobileNetV2.h5')
    return model

model = load_my_model()

uploaded_file = st.file_uploader("Upload gambar MRI...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Gambar Terunggah', use_column_width=True)
    
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Klasifikasi"):
        prediction = model.predict(img_array)
