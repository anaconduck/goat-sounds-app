# import os
# import tempfile
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import streamlit as st
# import tensorflow as tf

# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # KONFIGURASI
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL_PATH = "best_model_dnn_final.keras"
# CLASSES    = ["lapar", "normal", "stress"]
# SR         = 16000
# DURATION   = 2.0
# N_MFCC     = 40
# N_FFT      = 512
# HOP_LENGTH = 256
# MAX_FRAMES = int(np.ceil(SR * DURATION / HOP_LENGTH))

# # Emoji & warna per kelas
# CLASS_CONFIG = {
#     "lapar":  {"emoji": "🍽️", "color": "#FF6B6B", "icon": "😋", "desc": "Kambing sedang **lapar** dan membutuhkan pakan."},
#     "normal": {"emoji": "😊", "color": "#4ECDC4", "icon": "🐐", "desc": "Kambing dalam kondisi **normal** dan sehat."},
#     "stress": {"emoji": "😰", "color": "#FF8E53", "icon": "⚠️",  "desc": "Kambing sedang **stress**, perlu perhatian khusus!"},
# }


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # FEATURE EXTRACTION (sama persis dengan notebook training)
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def extract_mfcc_from_audio(y, sr=SR):
#     """Ekstrak MFCC-40 + delta + delta-delta, lalu flatten ke mean+std."""
#     # Pastikan panjang konsisten
#     target_len = int(sr * DURATION)
#     if len(y) < target_len:
#         y = np.pad(y, (0, target_len - len(y)), mode='reflect')
#     else:
#         y = y[:target_len]

#     # Pre-emphasis
#     y = librosa.effects.preemphasis(y)

#     # MFCC + delta + delta-delta
#     mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
#                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
#     delta  = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)

#     combined = np.vstack([mfcc, delta, delta2])  # (120, frames)

#     # Pad / truncate
#     if combined.shape[1] < MAX_FRAMES:
#         pad = MAX_FRAMES - combined.shape[1]
#         combined = np.pad(combined, ((0, 0), (0, pad)), mode='constant')
#     else:
#         combined = combined[:, :MAX_FRAMES]

#     # Flat: mean + std per feature row → (240,)
#     mean = combined.mean(axis=1)
#     std  = combined.std(axis=1)
#     flat = np.hstack([mean, std])

#     return flat, mfcc


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # LOAD MODEL (cached)
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # VISUALISASI
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def plot_waveform(y, sr):
#     fig, ax = plt.subplots(figsize=(10, 3))
#     fig.patch.set_facecolor('#0E1117')
#     ax.set_facecolor('#0E1117')
#     times = np.linspace(0, len(y) / sr, len(y))
#     ax.plot(times, y, color='#4ECDC4', linewidth=0.6, alpha=0.9)
#     ax.fill_between(times, y, alpha=0.15, color='#4ECDC4')
#     ax.set_xlabel('Waktu (detik)', color='#FAFAFA', fontsize=10)
#     ax.set_ylabel('Amplitudo', color='#FAFAFA', fontsize=10)
#     ax.set_title('Gelombang Suara', color='#FAFAFA', fontsize=12, fontweight='bold')
#     ax.tick_params(colors='#AAAAAA')
#     ax.spines['bottom'].set_color('#333333')
#     ax.spines['left'].set_color('#333333')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='y', alpha=0.15, color='#555555')
#     plt.tight_layout()
#     return fig


# def plot_mfcc(mfcc, sr):
#     fig, ax = plt.subplots(figsize=(10, 4))
#     fig.patch.set_facecolor('#0E1117')
#     ax.set_facecolor('#0E1117')
#     img = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH,
#                                     x_axis='time', ax=ax, cmap='magma')
#     ax.set_title('Spektrogram MFCC', color='#FAFAFA', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Koefisien MFCC', color='#FAFAFA', fontsize=10)
#     ax.set_xlabel('Waktu (detik)', color='#FAFAFA', fontsize=10)
#     ax.tick_params(colors='#AAAAAA')
#     cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
#     cbar.ax.yaxis.set_tick_params(color='#AAAAAA')
#     plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#AAAAAA')
#     plt.tight_layout()
#     return fig


# def plot_prediction_bars(probs):
#     fig, ax = plt.subplots(figsize=(8, 3))
#     fig.patch.set_facecolor('#0E1117')
#     ax.set_facecolor('#0E1117')

#     bar_colors = [CLASS_CONFIG[c]['color'] for c in CLASSES]
#     bar_labels = [f"{CLASS_CONFIG[c]['emoji']} {c.capitalize()}" for c in CLASSES]
#     percentages = probs * 100

#     bars = ax.barh(bar_labels, percentages, color=bar_colors, height=0.5,
#                    edgecolor='none', alpha=0.85)

#     for bar, pct in zip(bars, percentages):
#         ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
#                 f'{pct:.1f}%', va='center', ha='left', color='#FAFAFA',
#                 fontweight='bold', fontsize=12)

#     ax.set_xlim(0, 110)
#     ax.set_title('Probabilitas Prediksi', color='#FAFAFA', fontsize=12, fontweight='bold')
#     ax.tick_params(colors='#CCCCCC')
#     ax.spines['bottom'].set_color('#333333')
#     ax.spines['left'].set_color('#333333')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.invert_yaxis()
#     ax.grid(axis='x', alpha=0.15, color='#555555')
#     plt.tight_layout()
#     return fig


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # CUSTOM CSS
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM_CSS = """
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

#     /* Global */
#     .stApp {
#         font-family: 'Poppins', sans-serif;
#     }

#     /* Hero header */
#     .hero-header {
#         background: linear-gradient(135deg, #2D5016 0%, #4A7C2E 40%, #6B8F3C 70%, #8B6914 100%);
#         padding: 2.5rem 2rem;
#         border-radius: 16px;
#         text-align: center;
#         margin-bottom: 2rem;
#         box-shadow: 0 8px 32px rgba(45, 80, 22, 0.3);
#         border: 1px solid rgba(255, 255, 255, 0.1);
#     }
#     .hero-header h1 {
#         color: #FFFFFF;
#         font-size: 2.2rem;
#         margin: 0;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#         letter-spacing: 1px;
#     }
#     .hero-header p {
#         color: #E0E8D0;
#         font-size: 1.05rem;
#         margin-top: 0.6rem;
#         font-weight: 300;
#     }

#     /* Upload area */
#     .upload-section {
#         background: linear-gradient(145deg, #1a2e0a 0%, #1E3A12 100%);
#         border: 2px dashed #4A7C2E;
#         border-radius: 16px;
#         padding: 2rem;
#         text-align: center;
#         margin: 1.5rem 0;
#         transition: all 0.3s ease;
#     }
#     .upload-section:hover {
#         border-color: #6B8F3C;
#         box-shadow: 0 4px 20px rgba(74, 124, 46, 0.2);
#     }

#     /* Result card */
#     .result-card {
#         border-radius: 16px;
#         padding: 2rem;
#         text-align: center;
#         margin: 1.5rem 0;
#         box-shadow: 0 8px 32px rgba(0,0,0,0.25);
#         border: 1px solid rgba(255,255,255,0.08);
#     }
#     .result-card h2 {
#         margin: 0.5rem 0 0.3rem 0;
#         font-size: 1.8rem;
#     }
#     .result-card .emoji-big {
#         font-size: 4rem;
#         display: block;
#         margin-bottom: 0.5rem;
#     }
#     .result-card .confidence {
#         font-size: 2.4rem;
#         font-weight: 700;
#         margin: 0.5rem 0;
#     }
#     .result-card .desc {
#         font-size: 1.05rem;
#         opacity: 0.9;
#         margin-top: 0.5rem;
#     }

#     /* Info card */
#     .info-card {
#         background: linear-gradient(145deg, #1a2e0a 0%, #1E3A12 100%);
#         border-radius: 12px;
#         padding: 1.2rem 1.5rem;
#         margin: 0.8rem 0;
#         border-left: 4px solid #4A7C2E;
#     }
#     .info-card h4 {
#         color: #8BBA6B;
#         margin: 0 0 0.3rem 0;
#     }
#     .info-card p {
#         color: #CCDDBB;
#         margin: 0;
#         font-size: 0.9rem;
#     }

#     /* Feature badges */
#     .feature-row {
#         display: flex;
#         gap: 1rem;
#         justify-content: center;
#         flex-wrap: wrap;
#         margin: 1rem 0;
#     }
#     .feature-badge {
#         background: rgba(74, 124, 46, 0.2);
#         border: 1px solid rgba(74, 124, 46, 0.4);
#         border-radius: 24px;
#         padding: 0.4rem 1rem;
#         font-size: 0.85rem;
#         color: #8BBA6B;
#     }

#     /* Footer */
#     .footer {
#         text-align: center;
#         padding: 2rem 0 1rem 0;
#         color: #6B7B5E;
#         font-size: 0.85rem;
#         border-top: 1px solid #2A3A1A;
#         margin-top: 3rem;
#     }

#     /* Fix Streamlit default padding / margins */
#     .block-container {
#         padding-top: 2rem !important;
#         padding-bottom: 2rem !important;
#         padding-left: 2rem !important;
#         padding-right: 2rem !important;
#         max-width: 1200px !important;
#     }

#     /* Responsive Design for Mobile (HP) */
#     @media (max-width: 768px) {
#         .block-container {
#             padding-top: 1rem !important;
#             padding-bottom: 1rem !important;
#             padding-left: 0.5rem !important;
#             padding-right: 0.5rem !important;
#         }
#         .hero-header {
#             padding: 1.5rem 1rem;
#             margin-bottom: 1rem;
#         }
#         .hero-header h1 {
#             font-size: 1.6rem;
#         }
#         .hero-header p {
#             font-size: 0.9rem;
#         }
#         .upload-section {
#             padding: 1.5rem 1rem;
#         }
#         .result-card {
#             padding: 1.5rem 1rem;
#         }
#         .result-card h2 {
#             font-size: 1.4rem;
#         }
#         .result-card .confidence {
#             font-size: 2rem;
#         }
#         .result-card .emoji-big {
#             font-size: 3rem;
#         }
#         .feature-badge {
#             font-size: 0.75rem;
#             padding: 0.3rem 0.8rem;
#         }
#         .info-card {
#             padding: 1rem;
#         }
#     }

#     /* Hide default Streamlit elements */
#     #MainMenu {visibility: hidden;}
#     header {visibility: hidden;}
#     footer {visibility: hidden;}
# </style>
# """


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # MAIN APP
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def main():
#     st.set_page_config(
#         page_title="GoatVoice Analyzer 🐐",
#         page_icon="🐐",
#         layout="wide",
#         initial_sidebar_state="collapsed"
#     )

#     st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

#     # ── Hero Header ────────────────────────────────────────────────────
#     st.markdown("""
#     <div class="hero-header">
#         <h1>🐐 GoatVoice Analyzer</h1>
#         <p>Sistem Cerdas Klasifikasi Kondisi Kambing Berdasarkan Analisis Suara</p>
#         <div class="feature-row">
#             <span class="feature-badge">🎵 MFCC-40 Feature Extraction</span>
#             <span class="feature-badge">🧠 Deep Learning</span>
#             <span class="feature-badge">📊 3 Kelas Klasifikasi</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # ── Layout: Sidebar Info + Main ────────────────────────────────────
#     col_main, col_info = st.columns([3, 1])

#     with col_info:
#         st.markdown("### 🌾 Panduan")

#         st.markdown("""
#         <div class="info-card">
#             <h4>📁 Format File</h4>
#             <p>Upload file audio dalam format <b>WAV</b> yang berisi suara kambing.</p>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("""
#         <div class="info-card">
#             <h4>🐐 Kelas Deteksi</h4>
#             <p>
#                 🍽️ <b>Lapar</b> – perlu pakan<br>
#                 😊 <b>Normal</b> – sehat<br>
#                 😰 <b>Stress</b> – perlu perhatian
#             </p>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("""
#         <div class="info-card">
#             <h4>⚙️ Teknologi</h4>
#             <p>
#                 Fitur: MFCC-40 + Delta + Delta²<br>
#                 Model: CNN (Convolutional Neural Network)<br>
#                 Akurasi: ditampilkan setelah prediksi
#             </p>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("""
#         <div class="info-card">
#             <h4>🐾 Tips</h4>
#             <p>Pastikan rekaman audio jelas dan dominan suara kambing, hindari kebisingan latar belakang.</p>
#         </div>
#         """, unsafe_allow_html=True)

#     with col_main:
#         st.markdown("### 🎤 Upload Suara Kambing")

#         st.markdown("""
#         <div class="upload-section">
#             <p style="font-size: 2.5rem; margin: 0;">🐐🎵</p>
#             <p style="color: #8BBA6B; font-size: 1rem; margin: 0.5rem 0;">
#                 Seret & lepas file audio WAV di bawah ini
#             </p>
#         </div>
#         """, unsafe_allow_html=True)

#         uploaded_file = st.file_uploader(
#             "Pilih file audio WAV",
#             type=["wav"],
#             help="Upload file audio WAV berisi suara kambing (maks ~10 detik)",
#             label_visibility="collapsed"
#         )

#         if uploaded_file is not None:
#             # ── Audio Player ───────────────────────────────────────────
#             st.markdown("---")
#             st.markdown("#### 🔊 Putar Audio")
#             st.audio(uploaded_file, format="audio/wav")

#             # ── Load audio ─────────────────────────────────────────────
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#                 tmp.write(uploaded_file.getvalue())
#                 tmp_path = tmp.name

#             try:
#                 y, sr = librosa.load(tmp_path, sr=SR, duration=DURATION)
#             except Exception as e:
#                 st.error(f"❌ Gagal memuat file audio: {e}")
#                 return
#             finally:
#                 os.unlink(tmp_path)

#             # ── Visualisasi Waveform & MFCC ────────────────────────────
#             st.markdown("#### 📈 Analisis Audio")
#             tab_wave, tab_mfcc = st.tabs(["🌊 Gelombang Suara", "🎨 Spektrogram MFCC"])

#             with tab_wave:
#                 fig_wave = plot_waveform(y, sr)
#                 st.pyplot(fig_wave)
#                 plt.close(fig_wave)

#             with tab_mfcc:
#                 # Ekstrak fitur
#                 flat_features, mfcc_raw = extract_mfcc_from_audio(y, sr)
#                 fig_mfcc = plot_mfcc(mfcc_raw, sr)
#                 st.pyplot(fig_mfcc)
#                 plt.close(fig_mfcc)

#             # ── Prediksi ───────────────────────────────────────────────
#             st.markdown("---")
#             with st.spinner("🔍 Menganalisis suara kambing..."):
#                 flat_features, mfcc_raw = extract_mfcc_from_audio(y, sr)
#                 model = load_model()

#                 # Prediksi
#                 input_data = flat_features.reshape(1, -1)
#                 probs = model.predict(input_data, verbose=0)[0]
#                 pred_idx  = np.argmax(probs)
#                 pred_cls  = CLASSES[pred_idx]
#                 pred_conf = probs[pred_idx] * 100

#             # ── Hasil Prediksi ─────────────────────────────────────────
#             cfg = CLASS_CONFIG[pred_cls]

#             st.markdown(f"""
#             <div class="result-card" style="background: linear-gradient(135deg, 
#                 {cfg['color']}22 0%, {cfg['color']}11 100%);
#                 border: 2px solid {cfg['color']}44;">
#                 <span class="emoji-big">{cfg['icon']}</span>
#                 <h2 style="color: {cfg['color']};">Kondisi: {pred_cls.upper()}</h2>
#                 <div class="confidence" style="color: {cfg['color']};">{pred_conf:.1f}%</div>
#                 <p style="color: #AAAAAA; font-size: 0.85rem;">Tingkat Kepercayaan</p>
#                 <p class="desc" style="color: #DDDDDD;">{cfg['desc']}</p>
#             </div>
#             """, unsafe_allow_html=True)

#             # ── Bar chart probabilitas ─────────────────────────────────
#             st.markdown("#### 📊 Detail Probabilitas")
#             fig_bars = plot_prediction_bars(probs)
#             st.pyplot(fig_bars)
#             plt.close(fig_bars)

#             # ── Rekomendasi ────────────────────────────────────────────
#             st.markdown("#### 💡 Rekomendasi")
#             if pred_cls == "lapar":
#                 st.success("🌿 **Berikan pakan** yang cukup. Pastikan ketersediaan rumput, "
#                            "konsentrat, dan air minum. Periksa jadwal pemberian pakan secara teratur.")
#             elif pred_cls == "normal":
#                 st.info("✅ Kambing dalam kondisi **baik**. Lanjutkan pemeliharaan rutin. "
#                         "Pastikan kandang bersih dan nyaman.")
#             else:  # stress
#                 st.warning("⚠️ Kambing mengalami **stress**. Periksa kemungkinan penyebab: "
#                            "suhu lingkungan, kepadatan kandang, predator, atau kondisi kesehatan. "
#                            "Konsultasikan dengan dokter hewan jika perlu.")

#     # ── Footer ─────────────────────────────────────────────────────────
#     st.markdown("""
#     <div class="footer">
#         <p>🐐 GoatVoice Analyzer &nbsp;|&nbsp; Klasifikasi Suara Kambing dengan Deep Learning</p>
#         <p>MFCC-40 + Delta + Delta² &nbsp;•&nbsp; Dense Neural Network &nbsp;•&nbsp; TensorFlow/Keras</p>
#     </div>
#     """, unsafe_allow_html=True)


# if __name__ == "__main__":
#     main()
