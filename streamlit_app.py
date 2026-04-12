import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# ==========================================
# 1. KONFIGURASI HALAMAN & UI
# ==========================================
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# Menambahkan gaya kustom (CSS) sederhana
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Deteksi Jenis Tumor Otak via MRI")
st.write("""
Selamat datang di aplikasi purwarupa klasifikasi tumor otak. 
Aplikasi ini dikembangkan menggunakan model **MobileNetV2** untuk membantu proses skrining awal berdasarkan citra MRI.
""")

st.info("💡 **Petunjuk:** Unggah gambar MRI dalam format JPG atau PNG untuk memulai analisis.")

# ==========================================
# 2. LOAD MODEL (DENGAN CACHING)
# ==========================================
@st.cache_resource
def load_my_model():
    # Memuat model MobileNetV2 yang sudah dilatih
    model = load_model('best_brain_tumor_model_MobileNetV2.h5')
    return model

# Memanggil fungsi load model
with st.spinner('Sedang memuat model AI...'):
    model = load_my_model()

# Daftar kelas sesuai dataset
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'Normal (No Tumor)', 'Pituitary Tumor']

# ==========================================
# 3. FITUR UNGGAH GAMBAR
# ==========================================
uploaded_file = st.file_uploader("Pilih gambar MRI...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca dan menampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar MRI Terunggah', use_column_width=True)
    
    # Tombol Prediksi
    if st.button('Mulai Analisis'):
        with st.spinner('Menganalisis citra...'):
            # --- PREPROCESSING ---
            # 1. Konversi ke RGB (karena model dilatih dengan 3 channel)
            img = image.convert('RGB')
            # 2. Resize ke 224x224 sesuai input MobileNetV2
            img = img.resize((224, 224))
            # 3. Ubah ke array dan normalisasi (skala 1/255) sesuai tahap training
            img_array = np.array(img) / 255.0
            # 4. Tambahkan dimensi batch (1, 224, 224, 3)
            img_reshape = np.expand_dims(img_array, axis=0)
            
            # --- PREDIKSI ---
            predictions = model.predict(img_reshape)
            score = tf.nn.softmax(predictions[0]) # Menggunakan softmax untuk probabilitas
            
            label_index = np.argmax(predictions)
            label_name = class_names[label_index]
            confidence = np.max(predictions) * 100 # Kepastian dalam persen

            # --- TAMPILKAN HASIL ---
            st.markdown("---")
            st.subheader("Hasil Diagnosis AI")
            
            # Logika warna berdasarkan hasil
            if label_name == 'Normal (No Tumor)':
                st.success(f"**Prediksi: {label_name}**")
            else:
                st.error(f"**Prediksi: {label_name}**")
                
            st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")

            # Visualisasi Probabilitas dengan Bar Chart
            st.write("### Detail Probabilitas:")
            chart_data = pd.DataFrame({
                'Kategori': class_names,
                'Keyakinan (%)': predictions[0] * 100
            })
            st.bar_chart(chart_data.set_index('Kategori'))

st.markdown("---")
st.caption("Dibuat untuk Curriculum Developer Intern Assessment - Technical Test | Wina Nur Annisa 🚀")
