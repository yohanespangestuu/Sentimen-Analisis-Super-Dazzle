import streamlit as st
import joblib
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Sentimen", layout="wide")

# Load model dengan error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load('naive_bayes_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

model, vectorizer = load_model()

# Judul aplikasi
st.title("📊 Analisis Sentimen Ulasan Pelanggan")
st.markdown("---")

# Input dari pengguna
teks = st.text_area(
    "Masukkan ulasan pelanggan:",
    placeholder="Contoh: 'Pelayanan sangat memuaskan...'",
    height=150
)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔍 Prediksi Sentimen", type="primary"):
        if not teks.strip():
            st.warning("Masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Menganalisis..."):
                try:
                    # Transformasi dan prediksi
                    teks_vector = vectorizer.transform([teks])
                    
                    # Prediksi label dan probabilitas
                    prediction = model.predict(teks_vector)
                    probabilities = model.predict_proba(teks_vector)[0]
                    
                    # Mapping label (sesuaikan dengan model Anda)
                    label_map = {
                        0: {"emoji": "😞", "label": "Negatif", "color": "red"},
                        1: {"emoji": "😊", "label": "Positif", "color": "green"},
                        2: {"emoji": "😐", "label": "Netral", "color": "blue"}
                    }
                    
                    label_num = int(prediction[0])
                    label_info = label_map.get(label_num, {"emoji": "❓", "label": "Tidak Dikenali", "color": "gray"})

                    # Tampilkan hasil
                    with col2:
                        st.markdown("### Hasil Analisis")
                        
                        # Tampilan visual
                        st.metric(
                            label="Sentimen Prediksi",
                            value=f"{label_info['emoji']} {label_info['label']}",
                            delta=f"Probabilitas: {probabilities[label_num]:.2%}"
                        )
                        
                        # Grafik probabilitas
                        prob_data = {
                            "Sentimen": list(label_map.keys()),
                            "Probabilitas": probabilities
                        }
                        st.bar_chart(
                            prob_data,
                            x="Sentimen",
                            y="Probabilitas",
                            color="#FFAA00"
                        )
                        
                        # Detail teknis (expandable)
                        with st.expander("Detail Teknis"):
                            st.write(f"**Label Numerik**: {label_num}")
                            st.write(f"**Fitur TF-IDF**: {teks_vector.shape[1]} dimensi")
                            st.write("**Probabilitas Tiap Kelas:**")
                            for i, prob in enumerate(probabilities):
                                st.write(f"- {label_map.get(i, {}).get('label', 'Unknown')}: {prob:.2%}")

                except Exception as e:
                    st.error(f"Terjadi error saat prediksi: {str(e)}")

# Catatan kaki
st.markdown("---")
st.caption("Aplikasi ini menggunakan model Naive Bayes yang dilatih dengan data ulasan pelanggan.")