import os
import cv2
import numpy as np
from skimage.util import random_noise

def generate_noisy_datasets(dir_clean, base_out_dir, variances):
    """
    Génère 4 types d'images bruitées (Speckle, Gaussien, Poivre & Sel, Mixte)
    à différents niveaux d'intensité pour alimenter le pipeline.
    """
    np.random.seed(42) 
    
    if not os.path.exists(dir_clean):
        print(f" Erreur : Le dossier source '{dir_clean}' est introuvable.")
        return
        
    image_names = [f for f in os.listdir(dir_clean) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print(f"Traitement de {len(image_names)} images...")

    noise_types = ['speckle', 'gaussian', 'salt_pepper', 'mixed']
    for noise in noise_types:
        for var in variances:
            os.makedirs(os.path.join(base_out_dir, noise, f"var_{var}"), exist_ok=True)

    for img_name in image_names:
        path_clean = os.path.join(dir_clean, img_name)
        img_clean = cv2.imread(path_clean, cv2.IMREAD_GRAYSCALE)
        if img_clean is None: continue

        img_float = img_clean.astype(np.float32) / 255.0

        for var in variances:
            # --- Speckle ---
            gauss_speckle = np.random.normal(0, np.sqrt(var), img_float.shape)
            speckle_float = np.clip(img_float + img_float * gauss_speckle, 0, 1)
            img_speckle = (speckle_float * 255).astype(np.uint8)
            
            # --- Gaussien (Correction ici) ---
            gauss_float = random_noise(img_clean, mode='gaussian', var=var)
            img_gauss = (gauss_float * 255).astype(np.uint8)

            # --- Sel & Poivre (Correction ici) ---
            sp_float = random_noise(img_clean, mode='s&p', amount=var)
            img_sp = (sp_float * 255).astype(np.uint8)

            # --- Mixte (Correction ici) ---
            mixed_float = random_noise(speckle_float, mode='s&p', amount=var/2)
            img_mixed = (mixed_float * 255).astype(np.uint8)

            # --- SAUVEGARDE ---
            cv2.imwrite(os.path.join(base_out_dir, 'speckle', f"var_{var}", img_name), img_speckle)
            cv2.imwrite(os.path.join(base_out_dir, 'gaussian', f"var_{var}", img_name), img_gauss)
            cv2.imwrite(os.path.join(base_out_dir, 'salt_pepper', f"var_{var}", img_name), img_sp)
            cv2.imwrite(os.path.join(base_out_dir, 'mixed', f"var_{var}", img_name), img_mixed)

    print(" Génération des datasets terminée avec succès !")