import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import wiener
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from src.systeme_rc import systeme_rc_fusion_adaptative_v3

# --- Configuration de la page ---
st.set_page_config(page_title="Démo Interactive - Filtres Échographiques", layout="wide", page_icon="🔬")

# Custom CSS pour un look professionnel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1e3a8a; text-align: center; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.title("Démo Live : Filtrage d'Échographies Médicales")
st.markdown("---")

# --- FONCTIONS DE FILTRAGE CLASSIQUES ---
@st.cache_data
def apply_he(img): return cv2.equalizeHist(img)
@st.cache_data
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)
@st.cache_data
def apply_gauss(img): return cv2.GaussianBlur(img, (5, 5), 0)
@st.cache_data
def apply_mf(img): return cv2.medianBlur(img, 5)
@st.cache_data
def apply_bf(img): return cv2.bilateralFilter(img, 9, 75, 75)
@st.cache_data
def apply_wf(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered = wiener(np.float64(img), (5, 5))
        filtered = np.nan_to_num(filtered)
    return np.clip(filtered, 0, 255).astype(np.uint8)

# --- FONCTION POUR L'HISTOGRAMME ---
def plot_histogram(image, title="Histogramme"):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.hist(image.ravel(), bins=256, range=[0, 256], color='#3b82f6', alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_yticks([]) 
    st.pyplot(fig)

# --- INTERFACE UTILISATEUR (MODE SCIENTIFIQUE) ---
st.markdown("### Étape 1 : Chargement des images (Mode Scientifique)")
col_upload1, col_upload2, col_settings = st.columns([1, 1, 1.5])

with col_upload1:
    uploaded_noisy = st.file_uploader("1. Image Bruitée (À filtrer)", type=["png", "jpg", "jpeg", "tif"])
    
with col_upload2:
    uploaded_clean = st.file_uploader("2. Vérité Terrain (Optionnel, pour le vrai PSNR)", type=["png", "jpg", "jpeg", "tif"])

with col_settings:
    st.markdown("### Étape 2 : Stratégie")
    filter_choices = st.multiselect(
        "Sélectionnez le(s) filtre(s) à appliquer (Séquentiel) :",
        ["1. Baseline (Aucun filtre)", 
         "2. Gaussien (GF)", 
         "3. Médian (MF)", 
         "4. Wiener (WF)", 
         "5. Bilatéral (BF)", 
         "6. HE (Amélioration Contraste)", 
         "7. CLAHE (Contraste Local)",
         "8. Système RC (Logique Floue + Otsu)"],
        default=["1. Baseline (Aucun filtre)"]
    )

if uploaded_noisy is not None:
    # Lecture de l'image bruitée
    file_bytes_noisy = np.asarray(bytearray(uploaded_noisy.read()), dtype=np.uint8)
    img_noisy = cv2.imdecode(file_bytes_noisy, cv2.IMREAD_GRAYSCALE)
    
    # Lecture de la vérité terrain (Si le prof l'a uploadée)
    img_clean = None
    if uploaded_clean is not None:
        file_bytes_clean = np.asarray(bytearray(uploaded_clean.read()), dtype=np.uint8)
        img_clean = cv2.imdecode(file_bytes_clean, cv2.IMREAD_GRAYSCALE)
    
    # --- APPLICATION DU FILTRE ---
    img_filtered = img_noisy.copy()
    mask_visuel = None
    
    with st.spinner("Application du traitement en cours..."):
        for filter_choice in filter_choices:
            if "Baseline" in filter_choice:
                continue
            elif "Gaussien" in filter_choice: img_filtered = apply_gauss(img_filtered)
            elif "Médian" in filter_choice: img_filtered = apply_mf(img_filtered)
            elif "Wiener" in filter_choice: img_filtered = apply_wf(img_filtered)
            elif "Bilatéral" in filter_choice: img_filtered = apply_bf(img_filtered)
            elif "HE" in filter_choice: img_filtered = apply_he(img_filtered)
            elif "CLAHE" in filter_choice: img_filtered = apply_clahe(img_filtered)
            elif "Système RC" in filter_choice:
                # On passe img_filtered deux fois pour éviter les erreurs internes,
                # car le vrai calcul PSNR sera refait juste en dessous.
                img_filtered_new, mask_visuel, _, _ = systeme_rc_fusion_adaptative_v3(img_filtered, img_filtered)
                img_filtered = img_filtered_new

    # --- CALCUL DES MÉTRIQUES (VRAI CALCUL SCIENTIFIQUE) ---
    psnr_val, ssim_val = None, None
    if img_clean is not None:
        psnr_val = calc_psnr(img_clean, img_filtered, data_range=255)
        ssim_val = calc_ssim(img_clean, img_filtered, data_range=255)

    # --- AFFICHAGE DES RÉSULTATS ---
    st.markdown("---")
    st.markdown("### Analyse Scientifique (Performances)")
    
    if img_clean is not None:
        psnr_base = calc_psnr(img_clean, img_noisy, data_range=255)
        ssim_base = calc_ssim(img_clean, img_noisy, data_range=255)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("PSNR Initial (Bruit)", f"{psnr_base:.2f} dB")
        with col_m2:
            st.metric("SSIM Initial (Bruit)", f"{ssim_base:.4f}")
        with col_m3:
            delta_psnr = psnr_val - psnr_base
            st.metric("PSNR Final (Filtré)", f"{psnr_val:.2f} dB", delta=f"{delta_psnr:.2f} dB")
        with col_m4:
            delta_ssim = ssim_val - ssim_base
            st.metric("SSIM Final (Filtré)", f"{ssim_val:.4f}", delta=f"{delta_ssim:.4f}")
    else:
        st.warning(" Vérité terrain manquante. Uploadez l'image originale pour calculer le PSNR/SSIM.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1. Image Bruitée")
        # st.image gère très bien les tableaux numpy 2D en niveaux de gris
        st.image(img_noisy, use_container_width=True)
        plot_histogram(img_noisy, "Histogramme (Bruit)")
        
    with col2:
        filter_names = " + ".join([f.split('.')[1].strip() for f in filter_choices]) if filter_choices else "Aucun"
        st.markdown(f"#### 2. Image Filtrée ({filter_names})")
        st.image(img_filtered, use_container_width=True)
        plot_histogram(img_filtered, "Histogramme (Filtré)")
        
    with col3:
        st.markdown("#### 3. Vérité Terrain (Originale)")
        if img_clean is not None:
            st.image(img_clean, use_container_width=True)
            plot_histogram(img_clean, "Histogramme (Originale)")
        else:
            st.info("Aucune image de vérité terrain fournie.")
            
    if mask_visuel is not None:
        st.markdown("---")
        st.markdown("#### Vision de l'IA (Masque Flou de Système RC)")
        # Utilisation de la colormap 'magma' pour bien voir les contours
        st.image(mask_visuel, use_container_width=True, clamp=True)
        st.caption("En blanc : Zone protégée (Contours). En noir : Zone lissée (Fond).")