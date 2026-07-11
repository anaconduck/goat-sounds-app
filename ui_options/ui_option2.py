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

# CLASS_CONFIG = {
#     "lapar":  {
#         "emoji": "🍽️", "color": "#FF6B6B", "bg": "#2a1010",
#         "label": "Lapar",
#         "desc": "Kambing membutuhkan pakan. Periksa jadwal dan ketersediaan pakan segera.",
#         "action": "Berikan rumput / konsentrat dan pastikan air minum tersedia.",
#     },
#     "normal": {
#         "emoji": "✅", "color": "#4ECDC4", "bg": "#0e2020",
#         "label": "Normal",
#         "desc": "Kambing dalam kondisi sehat dan tenang.",
#         "action": "Lanjutkan pemeliharaan rutin. Kandang bersih, pakan terjadwal.",
#     },
#     "stress": {
#         "emoji": "⚠️", "color": "#FF8E53", "bg": "#2a1800",
#         "label": "Stress",
#         "desc": "Kambing mengalami tekanan. Perlu penanganan segera.",
#         "action": "Periksa suhu kandang, kepadatan, predator, atau isolasi. Hubungi dokter hewan jika berlanjut.",
#     },
# }

# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # CSS  –  mobile-first, centered, compact
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# html, body, .stApp { font-family: 'Inter', sans-serif; }

# /* ── Strip bawaan Streamlit ── */
# header          { visibility: hidden; }
# #MainMenu       { visibility: hidden; }
# footer          { visibility: hidden; }

# /* ── Kurangi padding container utama ── */
# .block-container {
#     padding-top: 1.5rem !important;
#     padding-bottom: 1.5rem !important;
#     padding-left: 1rem  !important;
#     padding-right: 1rem !important;
#     max-width: 520px !important;   /* maks lebar seperti HP */
# }

# /* ── App-header bar (sticky) ── */
# .app-header {
#     display: flex;
#     align-items: center;
#     gap: 10px;
#     padding: 14px 18px;
#     border-radius: 14px;
#     background: linear-gradient(135deg, #1b3a10 0%, #2e5a1a 60%, #3d6b25 100%);
#     margin-bottom: 18px;
#     box-shadow: 0 4px 16px rgba(0,0,0,.35);
# }
# .app-header .logo { font-size: 2rem; line-height: 1; }
# .app-header .title-block { flex: 1; }
# .app-header h1 {
#     margin: 0; padding: 0;
#     font-size: 1.15rem; font-weight: 700;
#     color: #fff; letter-spacing: .3px;
# }
# .app-header p {
#     margin: 2px 0 0; padding: 0;
#     font-size: 0.72rem; color: #b0d490; font-weight: 400;
# }

# /* ── Chip badge ── */
# .chip {
#     display: inline-block;
#     padding: 3px 10px;
#     border-radius: 20px;
#     font-size: 0.7rem;
#     font-weight: 600;
#     margin: 3px 2px 0;
#     background: rgba(255,255,255,.1);
#     color: #cce8aa;
#     border: 1px solid rgba(255,255,255,.15);
# }

# /* ── Kotak hasil prediksi ── */
# .result-card {
#     border-radius: 16px;
#     padding: 22px 20px 18px;
#     text-align: center;
#     margin: 14px 0;
#     border: 1.5px solid rgba(255,255,255,.12);
#     box-shadow: 0 6px 24px rgba(0,0,0,.3);
# }
# .result-card .r-emoji  { font-size: 3rem; display: block; margin-bottom: 6px; }
# .result-card .r-label  { font-size: 1.5rem; font-weight: 700; margin: 0 0 4px; }
# .result-card .r-conf   { font-size: 2rem; font-weight: 700; }
# .result-card .r-sub    { font-size: 0.72rem; color: #999; margin: 2px 0 10px; }
# .result-card .r-desc   { font-size: 0.85rem; color: #ccc; line-height: 1.5; }

# /* ── Kotak aksi / rekomendasi ── */
# .action-box {
#     border-radius: 12px;
#     padding: 14px 16px;
#     margin: 10px 0;
#     border-left: 4px solid;
#     font-size: 0.85rem;
#     line-height: 1.55;
# }

# /* ── Stat row (3 metrik) ── */
# .stat-row {
#     display: flex;
#     gap: 8px;
#     margin: 12px 0;
# }
# .stat-item {
#     flex: 1;
#     border-radius: 12px;
#     padding: 12px 8px;
#     text-align: center;
#     background: rgba(255,255,255,.04);
#     border: 1px solid rgba(255,255,255,.08);
# }
# .stat-item .s-val {
#     font-size: 1.15rem;
#     font-weight: 700;
#     display: block;
# }
# .stat-item .s-lbl {
#     font-size: 0.68rem;
#     color: #888;
#     margin-top: 3px;
#     display: block;
# }

# /* ── Info card ringan ── */
# .info-card {
#     border-radius: 12px;
#     padding: 14px 16px;
#     margin: 8px 0;
#     background: rgba(255,255,255,.03);
#     border: 1px solid rgba(255,255,255,.08);
#     font-size: 0.85rem;
#     line-height: 1.6;
#     color: #ccc;
# }
# .info-card strong { color: #eee; }

# /* ── Tab label yang lebih besar & mudah disentuh ── */
# [data-baseweb="tab"] { font-size: 0.82rem !important; padding: 10px 14px !important; }
# </style>
# """

# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # PROCESSING FUNCTIONS
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def extract_mfcc_from_audio(y, sr=SR):
#     target_len = int(sr * DURATION)
#     y = np.pad(y, (0, max(0, target_len - len(y))), mode='reflect')[:target_len]
#     y = librosa.effects.preemphasis(y)

#     mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
#     delta  = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)
#     combined = np.vstack([mfcc, delta, delta2])

#     if combined.shape[1] < MAX_FRAMES:
#         combined = np.pad(combined, ((0, 0), (0, MAX_FRAMES - combined.shape[1])), mode='constant')
#     else:
#         combined = combined[:, :MAX_FRAMES]

#     return np.hstack([combined.mean(axis=1), combined.std(axis=1)]), mfcc


# @st.cache_resource(show_spinner=False)
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # CHART HELPERS  –  ukuran kecil / ringkas
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def _style(ax, fig):
#     fig.patch.set_alpha(0.0)
#     ax.patch.set_alpha(0.0)
#     for sp in ['top', 'right']:
#         ax.spines[sp].set_visible(False)
#     for sp in ['bottom', 'left']:
#         ax.spines[sp].set_color('#333')
#     ax.tick_params(colors='#888', labelsize=8)
#     ax.xaxis.label.set_color('#888')
#     ax.yaxis.label.set_color('#888')
#     ax.title.set_color('#aaa')
#     ax.title.set_fontsize(10)
#     ax.title.set_fontweight('bold')


# def chart_probs(probs):
#     fig, ax = plt.subplots(figsize=(5, 2.2))
#     pcts = probs * 100
#     colors = [CLASS_CONFIG[c]['color'] for c in CLASSES]
#     labels = [CLASS_CONFIG[c]['label'] for c in CLASSES]
#     bars = ax.barh(labels, pcts, color=colors, height=0.55, edgecolor='none', alpha=0.9)
#     for bar, pct in zip(bars, pcts):
#         ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
#                 f"{pct:.1f}%", va='center', ha='left', fontsize=9,
#                 fontweight='600', color='#ddd')
#     ax.set_xlim(0, 115)
#     ax.set_title("Distribusi Probabilitas")
#     ax.invert_yaxis()
#     ax.tick_params(axis='y', labelsize=9, colors='#ccc')
#     _style(ax, fig)
#     plt.tight_layout(pad=0.5)
#     return fig


# def chart_wave(y, sr):
#     fig, ax = plt.subplots(figsize=(5, 2))
#     t = np.linspace(0, len(y) / sr, len(y))
#     ax.plot(t, y, color='#4ECDC4', linewidth=0.5, alpha=0.9)
#     ax.fill_between(t, y, alpha=0.12, color='#4ECDC4')
#     ax.set_xlabel('detik', fontsize=8)
#     ax.set_title("Waveform")
#     ax.set_yticks([])
#     _style(ax, fig)
#     plt.tight_layout(pad=0.5)
#     return fig


# def chart_mfcc(mfcc, sr):
#     fig, ax = plt.subplots(figsize=(5, 2.5))
#     img = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH,
#                                    x_axis='time', ax=ax, cmap='magma')
#     ax.set_title("MFCC Spectrogram")
#     ax.set_xlabel('detik', fontsize=8)
#     ax.set_ylabel('koef.', fontsize=8)
#     _style(ax, fig)
#     cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.01)
#     cbar.ax.tick_params(labelsize=7, colors='#888')
#     plt.tight_layout(pad=0.5)
#     return fig


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # HALAMAN ANALISIS
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def tab_analisis():
#     st.markdown("#### 📂 Upload Rekaman Suara")
#     uploaded = st.file_uploader(
#         "Pilih file WAV", type=["wav"],
#         help="Rekaman suara kambing, durasi min. 2 detik, format WAV.",
#         label_visibility="collapsed"
#     )

#     if uploaded is None:
#         st.markdown("""
#         <div class="info-card" style="text-align:center; padding:28px 16px;">
#             <div style="font-size:2.8rem;">🐐🎵</div>
#             <div style="margin-top:8px; color:#888; font-size:0.82rem;">
#                 Belum ada file dipilih.<br>Tap tombol di atas untuk upload.
#             </div>
#         </div>""", unsafe_allow_html=True)
#         return

#     # Audio player
#     st.audio(uploaded, format="audio/wav")

#     # Load & predict
#     with st.spinner("Mengekstrak fitur & menganalisis..."):
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#             tmp.write(uploaded.getvalue())
#             tmp_path = tmp.name
#         try:
#             y, sr = librosa.load(tmp_path, sr=SR, duration=DURATION)
#         except Exception as e:
#             st.error(f"Gagal membaca audio: {e}")
#             return
#         finally:
#             os.unlink(tmp_path)

#         flat, mfcc_raw = extract_mfcc_from_audio(y, sr)
#         model          = load_model()
#         probs          = model.predict(flat.reshape(1, -1), verbose=0)[0]

#     pred_cls  = CLASSES[np.argmax(probs)]
#     pred_conf = probs[np.argmax(probs)] * 100
#     cfg       = CLASS_CONFIG[pred_cls]

#     # ── Kartu Hasil ──
#     st.markdown(f"""
#     <div class="result-card" style="background:{cfg['bg']};
#          border-color:{cfg['color']}55;">
#         <span class="r-emoji">{cfg['emoji']}</span>
#         <div class="r-label" style="color:{cfg['color']};">{cfg['label']}</div>
#         <div class="r-conf"  style="color:{cfg['color']};">{pred_conf:.1f}<span style="font-size:1rem;font-weight:400">%</span></div>
#         <div class="r-sub">Confidence Score</div>
#         <div class="r-desc">{cfg['desc']}</div>
#     </div>""", unsafe_allow_html=True)

#     # ── 3 metrik probabilitas ──
#     p = {c: round(probs[i] * 100, 1) for i, c in enumerate(CLASSES)}
#     st.markdown(f"""
#     <div class="stat-row">
#         <div class="stat-item">
#             <span class="s-val" style="color:{CLASS_CONFIG['lapar']['color']};">{p['lapar']}%</span>
#             <span class="s-lbl">🍽️ Lapar</span>
#         </div>
#         <div class="stat-item">
#             <span class="s-val" style="color:{CLASS_CONFIG['normal']['color']};">{p['normal']}%</span>
#             <span class="s-lbl">✅ Normal</span>
#         </div>
#         <div class="stat-item">
#             <span class="s-val" style="color:{CLASS_CONFIG['stress']['color']};">{p['stress']}%</span>
#             <span class="s-lbl">⚠️ Stress</span>
#         </div>
#     </div>""", unsafe_allow_html=True)

#     # ── Rekomendasi ──
#     action_color = cfg['color']
#     st.markdown(f"""
#     <div class="action-box" style="background:{cfg['bg']};
#          border-left-color:{action_color}; color:#ddd;">
#         <strong style="color:{action_color};">💡 Rekomendasi Tindakan</strong><br>
#         {cfg['action']}
#     </div>""", unsafe_allow_html=True)

#     # ── Chart dalam expander agar tidak penuh ──
#     with st.expander("📊 Lihat Detail Analisis Audio"):
#         t1, t2, t3 = st.tabs(["Probabilitas", "Waveform", "MFCC"])
#         with t1:
#             fig = chart_probs(probs)
#             st.pyplot(fig, use_container_width=True)
#             plt.close(fig)
#         with t2:
#             fig = chart_wave(y, sr)
#             st.pyplot(fig, use_container_width=True)
#             plt.close(fig)
#         with t3:
#             fig = chart_mfcc(mfcc_raw, sr)
#             st.pyplot(fig, use_container_width=True)
#             plt.close(fig)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # HALAMAN PANDUAN
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def tab_panduan():
#     st.markdown("#### 📋 Cara Penggunaan")
#     for num, step in [
#         ("1", "Buka tab **Analisis**."),
#         ("2", "Tap **Browse files** dan pilih file audio WAV suara kambing."),
#         ("3", "Tunggu proses ekstraksi fitur dan prediksi model selesai."),
#         ("4", "Baca hasil kondisi kambing dan rekomendasi tindakannya."),
#         ("5", "Tap **Lihat Detail Analisis Audio** untuk melihat grafik waveform dan spektrogram."),
#     ]:
#         st.markdown(f"""
#         <div class="info-card" style="display:flex;gap:12px;align-items:flex-start;">
#             <span style="background:#2e5a1a;border-radius:50%;
#                   width:26px;height:26px;display:flex;align-items:center;
#                   justify-content:center;font-size:0.75rem;font-weight:700;
#                   color:#8dcc55;flex-shrink:0;">{num}</span>
#             <span>{step}</span>
#         </div>""", unsafe_allow_html=True)

#     st.markdown("#### 🎙️ Tips Rekaman")
#     st.markdown("""
#     <div class="info-card">
#         <strong>Format:</strong> WAV (mono/stereo) &nbsp;·&nbsp; <strong>Durasi:</strong> min. 2 detik<br>
#         <strong>Sample Rate:</strong> 16 kHz (disarankan)<br>
#         <strong>Lingkungan:</strong> Minim suara latar, rekaman dekat dengan kambing<br>
#         <strong>Perangkat:</strong> Ponsel / recorder portabel sudah cukup
#     </div>""", unsafe_allow_html=True)

#     st.markdown("#### 🐐 Kelas Kondisi")
#     for cls, cfg in CLASS_CONFIG.items():
#         st.markdown(f"""
#         <div class="info-card" style="border-left:4px solid {cfg['color']};">
#             <strong style="color:{cfg['color']};">{cfg['emoji']} {cfg['label']}</strong><br>
#             {cfg['desc']}
#         </div>""", unsafe_allow_html=True)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # HALAMAN TENTANG
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def tab_tentang():
#     st.markdown("#### 🔬 Teknologi")
#     for label, val in [
#         ("Ekstraksi Fitur",  "MFCC-40 + Delta + Delta-Delta → 120 fitur/frame"),
#         ("Representasi DNN", "Mean + Std per baris → vektor 240 dimensi"),
#         ("Model",            "Deep Neural Network (DNN) — Keras / TensorFlow"),
#         ("Kelas Output",     "3 kelas: Lapar · Normal · Stress"),
#         ("Input Audio",      f"SR {SR} Hz · Durasi {DURATION} s · {MAX_FRAMES} frame"),
#     ]:
#         st.markdown(f"""
#         <div class="info-card" style="padding:10px 14px;">
#             <strong>{label}:</strong> {val}
#         </div>""", unsafe_allow_html=True)

#     st.markdown("#### 🏛️ Tentang Proyek")
#     st.markdown("""
#     <div class="info-card">
#         <strong>GoatVoice Analyzer</strong> merupakan sistem klasifikasi akustik 
#         berbasis <em>Deep Learning</em> yang dikembangkan sebagai bagian dari 
#         penelitian S2 Ilmu Komputer. Sistem ini bertujuan membantu peternak, 
#         peneliti, dan praktisi dalam mendeteksi kondisi kesejahteraan kambing 
#         secara non-invasif melalui analisis suara.<br><br>
#         <span style="color:#888;font-size:0.78rem;">Versi 1.0 · © 2026</span>
#     </div>""", unsafe_allow_html=True)


# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # MAIN
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def main():
#     st.set_page_config(
#         page_title="GoatVoice Analyzer",
#         page_icon="🐐",
#         layout="centered",          # kunci: centered = seperti layar HP
#         initial_sidebar_state="collapsed",
#     )
#     st.markdown(CSS, unsafe_allow_html=True)

#     # ── Header kompak ──
#     st.markdown("""
#     <div class="app-header">
#         <div class="logo">🐐</div>
#         <div class="title-block">
#             <h1>GoatVoice Analyzer</h1>
#             <p>Sistem Deteksi Kondisi Suara Kambing</p>
#         </div>
#     </div>
#     <div style="margin-bottom:4px;">
#         <span class="chip">🎵 MFCC-40</span>
#         <span class="chip">🧠 DNN</span>
#         <span class="chip">📊 3 Kelas</span>
#     </div>
#     """, unsafe_allow_html=True)

#     # ── Navigasi tabs (mudah di-tap di HP) ──
#     t_analisis, t_panduan, t_tentang = st.tabs(
#         ["🎙️  Analisis", "📋  Panduan", "ℹ️  Tentang"]
#     )

#     with t_analisis:
#         tab_analisis()

#     with t_panduan:
#         tab_panduan()

#     with t_tentang:
#         tab_tentang()


# if __name__ == "__main__":
#     main()
