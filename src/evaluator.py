import os
import cv2
import numpy as np
import pandas as pd
import warnings
from scipy.signal import wiener
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

# 1. FONCTIONS DE BASE (Filtres)
def apply_he(img): 
    return cv2.equalizeHist(img)

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def apply_gauss(img): 
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_mf(img): 
    return cv2.medianBlur(img, 5)

def apply_bf(img): 
    return cv2.bilateralFilter(img, 9, 75, 75)

def apply_wf(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Bloque l'avertissement "divide by zero"
        filtered = wiener(np.float64(img), (5, 5))
        filtered = np.nan_to_num(filtered) # Répare les pixels problématiques
    return np.clip(filtered, 0, 255).astype(np.uint8)

# 2. LE DICTIONNAIRE DES 13 FILTRES
FILTERS_DICT = {
    "1. Baseline": lambda img: img,
    "2. HE": apply_he,
    "3. CLAHE": apply_clahe,
    "4. BF": apply_bf,
    "5. BF+HE": lambda img: apply_he(apply_bf(img)),
    "6. BF+CLAHE": lambda img: apply_clahe(apply_bf(img)),
    "7. MF": apply_mf,
    "8. MF+HE": lambda img: apply_he(apply_mf(img)),
    "9. MF+CLAHE": lambda img: apply_clahe(apply_mf(img)),
    "10. WF": apply_wf,  
    "11. WF+HE": lambda img: apply_he(apply_wf(img)),
    "12. WF+CLAHE": lambda img: apply_clahe(apply_wf(img)),
    "13. Gaussien (GF)": apply_gauss
}

# 3. MOTEUR D'ÉVALUATION (AVEC PROGRESSION)
def run_evaluation_pipeline(dir_clean, base_dir_noisy, csv_dir, types_de_bruit, variance_test=0.05):
    results = []
    
    if not os.path.exists(dir_clean):
        print(f" Erreur : Dossier propre introuvable ({dir_clean})")
        return

    image_names = [f for f in os.listdir(dir_clean) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    total_images = len(image_names)
    print(f"Lancement de l'évaluation sur {total_images} images...")
    
    for type_bruit in types_de_bruit:
        folder_path = os.path.join(base_dir_noisy, type_bruit, f"var_{variance_test}")
        if not os.path.exists(folder_path): 
            continue
            
        print(f"\n Début du traitement pour le bruit : {type_bruit.upper()}")
        
        # AJOUT ICI : enumerate pour compter les images (i)
        for i, img_name in enumerate(image_names):
            
            # Afficher la progression toutes les 50 images
            if (i + 1) % 50 == 0 or (i + 1) == total_images:
                print(f" - Progression : {i + 1} / {total_images} images filtrées...")
                
            img_clean = cv2.imread(os.path.join(dir_clean, img_name), cv2.IMREAD_GRAYSCALE)
            img_noisy = cv2.imread(os.path.join(folder_path, img_name), cv2.IMREAD_GRAYSCALE)
            
            if img_clean is None or img_noisy is None: 
                continue
            
            # Test des 13 filtres
            for filter_name, filter_func in FILTERS_DICT.items():
                img_filtered = filter_func(img_noisy) 
                
                results.append({
                    "Image": img_name, 
                    "Type de Bruit": type_bruit,
                    "Filtre": filter_name, 
                    "PSNR": calc_psnr(img_clean, img_filtered, data_range=255),
                    "SSIM": calc_ssim(img_clean, img_filtered, data_range=255)
                })
                
    # Sauvegarde
    os.makedirs(csv_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(csv_dir, "resultats_evaluation_filtres.csv"), index=False)
    print("\nÉvaluation terminée et sauvegardée !")