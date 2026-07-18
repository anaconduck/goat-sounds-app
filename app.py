import os
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import soundfile as sf
import io

MODEL_PATH = "best_model.keras"
CLASSES    = ["lapar", "normal", "stress"]
SR         = 16000
DURATION   = 2.0
N_MFCC     = 40
N_FFT      = 512
HOP_LENGTH = 256
MAX_FRAMES = int(np.ceil(SR * DURATION / HOP_LENGTH))

CLASS_CONFIG = {
    "lapar":  {"emoji": "🍽️", "color": "#F59E0B", "icon": "😋", "desc": "Kambing butuh pakan tambahan segera."},
    "normal": {"emoji": "😊", "color": "#10B981", "icon": "🐐", "desc": "Kondisi sehat, tenang, dan sangat baik."},
    "stress": {"emoji": "😰", "color": "#EF4444", "icon": "⚠️",  "desc": "Kambing terindikasi stres! Segera periksa lingkungan."},
}

def extract_mfcc(signal, sr=SR, n_mfcc=N_MFCC, lifter=0):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH, lifter=lifter)
    features = mfcc.T  
    if features.shape[0] < MAX_FRAMES:
        features = np.pad(features, ((0, MAX_FRAMES - features.shape[0]), (0, 0)), mode='constant')
    else:
        features = features[:MAX_FRAMES]
    return features

@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def plot_full_spectrogram(y, sr, start_time=None, duration=2.0):
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
    
    # Tema Dark UI
    fig.patch.set_facecolor('#0B100E')
    ax.set_facecolor('#0B100E')
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='magma')
    
    if start_time is not None:
        ax.axvspan(start_time, start_time + duration, color='white', alpha=0.15)
        ax.axvline(start_time, color='#10B981', linestyle='-', linewidth=2.5)
        ax.axvline(start_time + duration, color='#10B981', linestyle='-', linewidth=2.5)
        
    ax.set_title('Spektrogram Frekuensi Audio', color='#E2E8F0', fontsize=13, fontweight='bold')
    ax.set_xlabel('Waktu (detik)', color='#94A3B8', fontsize=10)
    ax.set_ylabel('Frekuensi (Hz)', color='#94A3B8', fontsize=10)
    ax.tick_params(colors='#64748B')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1E293B')
    ax.spines['left'].set_color('#1E293B')
    ax.grid(color='#1E293B', alpha=0.5, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_prediction_bars_dark(probs):
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)
    fig.patch.set_facecolor('#0B100E')
    ax.set_facecolor('#0B100E')

    bar_colors = [CLASS_CONFIG[c]['color'] for c in CLASSES]
    bar_labels = [f"{CLASS_CONFIG[c]['emoji']} {c.capitalize()}" for c in CLASSES]
    percentages = probs * 100

    bars = ax.barh(bar_labels, percentages, color=bar_colors, height=0.5,
                   edgecolor='none', alpha=0.9)

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', ha='left', color='#F8FAFC',
                fontweight='bold', fontsize=13)

    ax.set_xlim(0, 110)
    ax.tick_params(colors='#94A3B8')
    ax.spines['bottom'].set_color('#1E293B')
    ax.spines['left'].set_color('#1E293B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.1, color='#F8FAFC')
    plt.tight_layout()
    return fig


# CUSTOM CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    /* Global Base */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0B100E;
        color: #E2E8F0;
    }

    /* Animasi Futuristik */
    @keyframes subtleGlow {
        0% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.15); }
        50% { box-shadow: 0 0 35px rgba(16, 185, 129, 0.35); }
        100% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.15); }
    }
    @keyframes floatUp {
        0% { transform: translateY(5px); opacity: 0.9; }
        100% { transform: translateY(0px); opacity: 1; }
    }

    /* Hero header with Glassmorphism + Glow */
    .hero-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.3) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 2rem 1.5rem; /* Dikurangi dari 3rem agar jarak atas/bawah tidak terlalu jauh */
        border-radius: 24px;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: subtleGlow 4s infinite alternate;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        background: -webkit-linear-gradient(45deg, #6EE7B7, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 4px 20px rgba(16, 185, 129, 0.4);
    }
    .hero-header p {
        color: #94A3B8;
        font-size: 1.2rem;
        margin-top: 1rem;
        font-weight: 400;
    }

    /* Result card Premium - Centered */
    .result-card {
        border-radius: 24px;
        padding: 3rem 1.5rem;
        text-align: center;
        margin: 0rem 0 1.5rem 0; /* Sweet spot: tidak sedekat -0.5rem, tapi tidak sejauh 0.5rem */
        background: #111815;
        box-shadow: 0 20px 50px -10px rgba(0,0,0,0.8);
        backdrop-filter: blur(12px);
        animation: floatUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    .result-card h2 {
        margin: 0.8rem 0 0.2rem 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        text-transform: uppercase;
    }
    .result-card .emoji-big {
        font-size: 5rem;
        display: block;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 10px 15px rgba(0,0,0,0.5));
    }
    .result-card .confidence {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-card .desc {
        font-size: 1.15rem;
        opacity: 0.9;
        margin-top: 1.2rem;
        color: #CBD5E1;
    }

    /* Section headers yang menarik */
    .section-header {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        color: #6EE7B7 !important;
        font-size: 1.0rem !important; /* Diperkecil lagi agar lebih proporsional */
        font-weight: 700;
        margin-top: 1.2rem !important; /* Didekatkan dengan elemen di atasnya (awalnya 2rem) */
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.1);
    }

    /* Streamlit overrides: Keren Button */
    .stButton>button {
        background: linear-gradient(135deg, #10B981 0%, #047857 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important; /* Disamakan dengan teks langkah */
        padding: 0.5rem 2rem !important; /* Dikurangi agar tombol tidak terlalu gemuk */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
        margin-top: 0.6rem !important; /* Sweet spot: tidak sedekat 0.2, tidak sejauh 1.0 */
        margin-bottom: -0.8rem !important; /* Dirapatkan ke bawah (Langkah 3) */
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 25px rgba(16, 185, 129, 0.5) !important;
    }

    /* Fix container padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important; /* Diberi ruang agar tidak tertutup footer */
        max-width: 800px !important;
    }

    /* Footer selalu di bawah (Fixed) */
    .footer-fixed {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(11, 16, 14, 0.95);
        backdrop-filter: blur(5px);
        text-align: center;
        padding: 12px 0;
        color: #475569;
        font-size: 0.85rem;
        z-index: 999;
        border-top: 1px solid #1E293B;
    }

    /* Mobile specific adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .hero-header {
            padding: 2rem 1rem;
            border-radius: 18px;
        }
        .hero-header h1 {
            font-size: 2rem;
        }
        .hero-header p {
            font-size: 1rem;
        }
        .result-card {
            padding: 2rem 1rem;
            border-radius: 18px;
        }
        .result-card .emoji-big {
            font-size: 4rem;
        }
        .result-card h2 {
            font-size: 1.8rem;
        }
        .result-card .confidence {
            font-size: 2.8rem;
        }
    }
</style>
"""

# MAIN APP
def main():
    st.set_page_config(
        page_title="GoatVoice Analyzer",
        page_icon="🐐",
        layout="centered"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Load Model
    model = load_keras_model()
    if model is None:
        st.error(f"❌ Model tidak ditemukan di path: {MODEL_PATH}. Pastikan Anda sudah menjalankan training.")
        return

    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>🐐 GoatVoice Analyzer</h1>
        <p>Pantau kesehatan dan kenyamanan kambing Anda secara otomatis melalui suara.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input Audio
    st.markdown('<div class="section-header">🎙️ Langkah 1: Input Suara Kambing</div>', unsafe_allow_html=True)
    
    input_method = st.radio("Pilih metode input:", ["Unggah File Audio", "Rekam Langsung (Mikrofon)"], horizontal=True)
    
    audio_data = None
    
    if input_method == "Unggah File Audio":
        uploaded_file = st.file_uploader("Upload rekaman (WAV/MP3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            audio_data = uploaded_file.getvalue()
    else:
        st.info("💡 Tekan tombol mikrofon di bawah untuk merekam suara. Usahakan suara kambing terdengar jelas.")
        recorded_audio = st.audio_input("Mulai Perekaman")
        if recorded_audio is not None:
            audio_data = recorded_audio.getvalue()

    if audio_data is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=SR)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            err_str = str(e)
            st.error(f"❌ Gagal memproses dokumen audio: {err_str}")
            return
        finally:
            os.unlink(tmp_path)

        # Pilih Interval
        st.markdown('<div class="section-header">✂️ Langkah 2: Seleksi Bagian Terbaik</div>', unsafe_allow_html=True)
        
        # Peringatan jika audio terlalu pendek atau pas 2 detik
        if duration <= DURATION + 0.05:
            st.success(f"⚡ **Durasi Pas:** Audio Anda ({duration:.1f}s) memenuhi syarat (≤ 2 detik). Sistem akan langsung memprosesnya secara utuh.")
            start_time = 0.0
        else:
            st.markdown("<p style='color: #94A3B8; font-size: 1rem;'>Geser slider untuk memilih <b>2 detik</b> suara yang paling jernih dari seluruh rekaman.</p>", unsafe_allow_html=True)
            max_val = max(0.1, duration - DURATION)
            start_time = st.slider("Titik Mulai (Start Time)", min_value=0.0, max_value=float(max_val), value=0.0, step=0.1)
        
        # Plot spectrogram
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True) # Tambah jarak atas agar tidak mepet
        fig_spec = plot_full_spectrogram(y, sr, start_time=start_time, duration=DURATION)
        st.pyplot(fig_spec)
        plt.close(fig_spec)
        
        start_sample = int(start_time * sr)
        end_sample = int((start_time + DURATION) * sr)
        y_slice = y[start_sample:end_sample]
        
        if len(y_slice) < int(DURATION * sr):
            y_slice = np.pad(y_slice, (0, int(DURATION * sr) - len(y_slice)), mode='constant')

        buffer = io.BytesIO()
        sf.write(buffer, y_slice, sr, format='wav')
        
        st.markdown("<p style='color: #94A3B8; font-size: 0.9rem; margin-top: 5px; margin-bottom: 8px;'>Dengarkan hasil potongan:</p>", unsafe_allow_html=True)
        st.audio(buffer.getvalue(), format="audio/wav")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("✨ Analisis Suara Sekarang", type="primary", use_container_width=True)

        # Hasil Prediksi
        if analyze_button:
            st.markdown('<div class="section-header">📊 Langkah 3: Hasil Analisis AI</div>', unsafe_allow_html=True)
            with st.spinner("🚀 Mengekstraksi pola suara..."):
                features = extract_mfcc(y_slice, sr=sr, n_mfcc=N_MFCC)
                input_data = np.expand_dims(features, axis=0)

                probs = model.predict(input_data, verbose=0)[0]
                pred_idx  = np.argmax(probs)
                pred_cls  = CLASSES[pred_idx]
                pred_conf = probs[pred_idx] * 100

            cfg = CLASS_CONFIG[pred_cls]
            
            st.markdown(f"""
            <div class="result-card" style="border: 2px solid {cfg['color']}55; box-shadow: 0 0 40px {cfg['color']}33;">
                <span class="emoji-big">{cfg['icon']}</span>
                <p style="color: #94A3B8; font-size: 1.1rem; margin-bottom: -5px; letter-spacing: 1px;">Kondisi Terdeteksi</p>
                <h2 style="color: {cfg['color']}; text-shadow: 0 0 20px {cfg['color']}66;">{pred_cls.upper()}</h2>
                <div class="confidence" style="color: #F8FAFC;">{pred_conf:.1f}<span style="font-size: 2rem; color: #94A3B8;">%</span></div>
                <p style="color: #64748B; font-size: 1rem; margin-top: -15px;">Akurasi Prediksi</p>
                <p class="desc">{cfg['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Bar Chart Probabilitas
            st.markdown("<p style='color: #F8FAFC; font-weight: 700; font-size: 1.2rem; margin-top: 1rem;'>Grafik Probabilitas:</p>", unsafe_allow_html=True)
            fig_bars = plot_prediction_bars_dark(probs)
            st.pyplot(fig_bars)
            plt.close(fig_bars)
            
            # Saran interaktif
            st.markdown("<br><p style='color: #F8FAFC; font-weight: 700; font-size: 1.2rem;'>💡 Saran Tindakan:</p>", unsafe_allow_html=True)
            if pred_cls == "lapar":
                st.warning("**Tindakan Prioritas:** Segera sediakan pakan segar dan pastikan ketersediaan air minum.")
            elif pred_cls == "normal":
                st.success("**Aman Terkendali:** Ternak nyaman. Cukup lanjutkan perawatan rutin Anda.")
            else:  
                st.error("**Perhatian Khusus:** Lakukan pengecekan fisik kambing atau amati kondisi lingkungan sekitar kandang (panas, predator, sakit).")

    # Footer
    st.markdown("""
    <div class="footer-fixed">
        &copy; 2026 GoatVoice Analyzer. Hak Cipta Dilindungi.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
