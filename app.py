import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Pelanggan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load assets
import streamlit as st
from PIL import Image

def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Gagal memuat gambar: {str(e)}")
        return None

logo = load_image('super-banner.jpg')

if logo:
    st.image(
        logo, 
        use_container_width=True  # Parameter yang diperbarui
    )
    
else:
    st.warning("Gambar tidak ditemukan")

# CSS custom
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# local_css("style.css")  # Uncomment jika ada file CSS

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

# Sidebar untuk informasi tambahan
with st.sidebar:
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan model **Naive Bayes** untuk menganalisis sentimen dari ulasan pelanggan.
    """)
    
    # Contoh ulasan
    st.markdown("### Contoh Ulasan")
    examples = st.selectbox(
        "Pilih contoh:",
        [
            "Pilih...",
            "Pelayanan sangat memuaskan dan cepat!",
            "Produk datang terlambat dan dalam kondisi rusak.",
            "Cukup standar, tidak ada yang istimewa."
        ]
    )
    
    if examples != "Pilih...":
        st.session_state.example_text = examples
    
    st.markdown("---")
    st.markdown("### Statistik Model")
    st.metric("Akurasi Model", "85%")  # Ganti dengan nilai aktual
    st.caption("*Akurasi pada data testing")

# Header dengan layout yang lebih menarik
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("""<div style='font-size: 3rem; text-align: center;'>📊</div>""", unsafe_allow_html=True)

with col2:
    st.title("Analisis Sentimen Ulasan Pelanggan")
    st.markdown("""
    <div style='color: #666; margin-top: -10px;'>
    Analisis otomatis sentimen dari ulasan pelanggan menggunakan kecerdasan buatan
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Input area dengan layout lebih baik
st.markdown("### 🖊️ Masukkan Ulasan Pelanggan")
teks = st.text_area(
    label="",
    value=st.session_state.get('example_text', ''),
    placeholder="Contoh: 'Pelayanan sangat memuaskan...'",
    height=150,
    label_visibility="collapsed"
)

# Tombol dengan layout lebih baik
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button(
        "🔍 Analisis Sentimen",
        type="primary",
        use_container_width=True
    )

if predict_btn:
    if not teks.strip():
        st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu!")
    else:
        with st.spinner("🔄 Menganalisis sentimen..."):
            try:
                # Transformasi dan prediksi
                teks_vector = vectorizer.transform([teks])
                
                # Prediksi label dan probabilitas
                prediction = model.predict(teks_vector)
                probabilities = model.predict_proba(teks_vector)[0]
                
                # Mapping label
                label_map = {
                    0: {"emoji": "😊", "label": "Positif", "color": "#2ecc71"},
                    1: {"emoji": "😞", "label": "Negatif", "color": "#e74c3c"},
                    2: {"emoji": "😐", "label": "Netral", "color": "#3498db"}
                }
                
                label_num = int(prediction[0])
                label_info = label_map.get(label_num, {"emoji": "❓", "label": "Tidak Dikenali", "color": "#95a5a6"})

                # Hasil analisis
                st.markdown("---")
                st.markdown("## 📌 Hasil Analisis")
                
                # Container untuk hasil
                with st.container():
                    # Bagian atas - hasil utama
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        # Card hasil
                        st.markdown(f"""
                        <div style='
                            background: #f8f9fa;
                            border-radius: 10px;
                            padding: 20px;
                            border-left: 5px solid {label_info['color']};
                            margin-bottom: 20px;
                        '>
                            <h3 style='color: {label_info['color']}; margin-top: 0;'>
                                {label_info['emoji']} {label_info['label']}
                            </h3>
                            <p>Probabilitas: <strong>{probabilities[label_num]:.2%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detail teknis
                        with st.expander("🔧 Detail Teknis"):
                            st.write(f"**Label Numerik**: {label_num}")
                            st.write(f"**Fitur TF-IDF**: {teks_vector.shape[1]} dimensi")
                            st.write("**Probabilitas Tiap Kelas:**")
                            for i, prob in enumerate(probabilities):
                                st.write(f"- {label_map.get(i, {}).get('label', 'Unknown')}: {prob:.2%}")
                    
                    with res_col2:
                        # Visualisasi probabilitas
                        prob_df = pd.DataFrame({
                            "Sentimen": [label_map.get(i, {}).get("label", "Unknown") for i in label_map],
                            "Probabilitas": probabilities,
                            "Warna": [label_map.get(i, {}).get("color", "#95a5a6") for i in label_map]
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x="Sentimen",
                            y="Probabilitas",
                            color="Sentimen",
                            color_discrete_map={
                                "Positif": "#2ecc71",
                                "Negatif": "#e74c3c",
                                "Netral": "#3498db"
                            },
                            text_auto='.2%',
                            height=300
                        )
                        fig.update_layout(
                            showlegend=False,
                            yaxis_title="Probabilitas",
                            xaxis_title="",
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Bagian bawah - saran berdasarkan sentimen
                st.markdown("## 💡 Rekomendasi")
                
                feedback_suggestions = {
                    0: "✅ Pertahankan kualitas layanan! Ulasan positif menunjukkan kepuasan pelanggan.",
                    1: "❌ Perlu perbaikan! Identifikasi masalah dan siapkan solusi untuk keluhan pelanggan.",
                    2: "🔍 Tingkatkan engagement! Ulasan netral bisa dioptimalkan untuk meningkatkan kepuasan."
                }
                
                st.info(feedback_suggestions.get(label_num, "Analisis sentimen selesai."))
                
            except Exception as e:
                st.error(f"❌ Terjadi error saat prediksi: {str(e)}")

# Footer
st.markdown("---")
footer_col1, footer_col2 = st.columns([2, 1])
with footer_col1:
    st.caption("© 2025 Analisis Sentimen Pelanggan | Dibangun dengan Streamlit")
with footer_col2:
    st.caption("Versi terbaru")