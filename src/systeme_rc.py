import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

def systeme_rc_fusion_adaptative_v3(img_noisy, img_clean, k_transition=0.15):
    """
    Système RC v3.0 : Logique Floue + Seuil Adaptatif d'Otsu.
    Le système est maintenant 100% autonome et s'adapte à l'histogramme de chaque image.
    """
    median_filtered = cv2.medianBlur(img_noisy, 5)
    bilateral_filtered = cv2.bilateralFilter(img_noisy, 9, 75, 75)
    
    # 1. Extraction du Gradient (Contours)
    img_for_gradient = cv2.medianBlur(img_noisy, 3) 
    sobelx = cv2.Sobel(img_for_gradient, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_for_gradient, cv2.CV_64F, 0, 1, ksize=3)
    gradient_norm = cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. INNOVATION : Calcul du seuil optimal automatique (Méthode d'Otsu)
    # L'algorithme trouve tout seul la limite entre "zone plate" et "contour"
    otsu_thresh, _ = cv2.threshold(gradient_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Inférence Floue avec le seuil intelligent
    mask_float = 1.0 / (1.0 + np.exp(-k_transition * (gradient_norm - otsu_thresh)))
    
    # 4. Fusion
    fusion_float = (mask_float * bilateral_filtered.astype(np.float32)) + ((1.0 - mask_float) * median_filtered.astype(np.float32))
    img_rc = np.clip(fusion_float, 0, 255).astype(np.uint8)
    
    psnr_val = calc_psnr(img_clean, img_rc, data_range=255)
    ssim_val = calc_ssim(img_clean, img_rc, data_range=255)
    
    return img_rc, (mask_float * 255).astype(np.uint8), psnr_val, ssim_val

def preparer_dataset_rc(dir_clean, base_dir_noisy, dossier_sortie, csv_dir, k_trans=0.15):
    """
    Applique le système RC (Otsu) sur tous les bruits et exporte les résultats 
    pour le Deep Learning. Notez qu'on ne passe plus de seuil (thresh) en paramètre !
    """
    types_de_bruit = ['gaussian', 'mixed', 'salt_pepper', 'speckle']
    resultats = []
    
    images = [f for f in os.listdir(dir_clean) if f.endswith(('.png', '.jpg', '.tif'))]
    print(f"Lancement du RC (v3 Otsu) sur {len(images)} images pour 4 types de bruit...")
    
    for type_bruit in types_de_bruit:
        # On utilise arbitrairement la variance 0.05 pour l'évaluation standard
        dossier_bruit_actuel = os.path.join(base_dir_noisy, type_bruit, "var_0.05")
        if not os.path.exists(dossier_bruit_actuel):
            continue
            
        # Dossier de sortie pour sauvegarder les images pour votre futur projet Deep Learning
        os.makedirs(os.path.join(dossier_sortie, type_bruit), exist_ok=True)
        
        for img_name in images:
            clean_img = cv2.imread(os.path.join(dir_clean, img_name), cv2.IMREAD_GRAYSCALE)
            noisy_img = cv2.imread(os.path.join(dossier_bruit_actuel, img_name), cv2.IMREAD_GRAYSCALE)
            
            if clean_img is None or noisy_img is None: continue
            
            # Appel de l'algorithme V3
            img_resultat, _, psnr, ssim = systeme_rc_fusion_adaptative_v3(noisy_img, clean_img, k_trans)
            
            # Sauvegarde de l'image ultra-propre pour votre modèle DL
            cv2.imwrite(os.path.join(dossier_sortie, type_bruit, img_name), img_resultat)
            
            # Nommage distinctif pour bien briller sur les graphiques
            resultats.append({
                "Image": img_name, 
                "Type de Bruit": type_bruit, 
                "Filtre": "14. Système RC (Otsu)",
                "PSNR": psnr, 
                "SSIM": ssim
            })
            
    os.makedirs(csv_dir, exist_ok=True)
    df = pd.DataFrame(resultats)
    df.to_csv(os.path.join(csv_dir, "resultats_rc_finaux.csv"), index=False)
    print(" Filtrage RC terminés avec succès.")