import os
from pathlib import Path

# --- CHEMINS DYNAMIQUES ---
# Path(__file__).parent.parent trouve automatiquement le dossier racine du projet
ROOT_DIR = Path(__file__).parent.parent

# Dossiers de données
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"       # Là où vous avez mis vos images propres
NOISY_DATA_DIR = DATA_DIR / "noisy"   # Là où iront les images avec du speckle/gaussien

# Dossiers de sortie (résultats)
OUTPUT_DIR = ROOT_DIR / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"          # Pour sauvegarder les scores PSNR/SSIM
FIGURES_DIR = OUTPUT_DIR / "figures"  # Pour les graphiques et le zoom
DL_OUTPUT_DIR = OUTPUT_DIR / "images_rc_dl" # Pour les images traitées par le Système RC (DL)

# --- HYPERPARAMÈTRES GLOBAUX ---
VARIANCES_BRUIT = [0.01, 0.05, 0.10]
TYPES_BRUIT = ['speckle', 'gaussian', 'salt_pepper', 'mixed']

# Paramètres du Système Expert (RC)
RC_THRESHOLD = 30
RC_K_TRANSITION = 0.15