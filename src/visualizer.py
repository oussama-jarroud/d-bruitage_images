import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generer_graphiques_comparatifs(csv_dir, figures_dir):
    """Génère des Boxplots séparés pour une meilleure lisibilité."""
    path_filtres = os.path.join(csv_dir, "resultats_evaluation_filtres.csv")
    path_rc = os.path.join(csv_dir, "resultats_rc_finaux.csv")
    
    if not os.path.exists(path_filtres): 
        print(" Fichier d'évaluation introuvable.")
        return

    df_filtres = pd.read_csv(path_filtres)
    
    # Fusion avec le RC
    if os.path.exists(path_rc):
        df_rc = pd.read_csv(path_rc)
        df_combined = pd.concat([df_filtres, df_rc], ignore_index=True)
    else:
        df_combined = df_filtres

    # Nettoyage des noms de bruit
    bruit_mapping = {"gaussian": "Gaussien", "mixed": "Mixte", "salt_pepper": "Sel & Poivre", "speckle": "Speckle"}
    df_combined["Type de Bruit"] = df_combined["Type de Bruit"].replace(bruit_mapping)
    
    # SÉPARATION INTELLIGENTE DES DONNÉES
    # On sépare les filtres "purs" (pour la comparaison directe avec le RC) des combinaisons HE/CLAHE (pour l'analyse de l'impact du contraste)
    # 1. Les filtres purs (Pour comparer les vraies performances de débruitage)
    filtres_principaux = [
        "1. Baseline", "13. Gaussien (GF)", "7. MF", 
        "10. WF", "4. BF", "14. Système RC (Otsu)"
    ]
    
    df_principaux = df_combined[df_combined['Filtre'].isin(filtres_principaux)]
    df_bonus = df_combined[~df_combined['Filtre'].isin(filtres_principaux)]
    
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- FONCTION POUR TRACER LES GRAPHIQUES ---
    def tracer_boxplot(df_data, metric, titre, nom_fichier):
        plt.figure(figsize=(14, 7)) # Taille plus compacte et lisible
        
        # On trie pour que les filtres s'affichent dans le bon ordre (1, 2, 3...)
        order = sorted(df_data['Filtre'].unique(), key=lambda x: int(x.split('.')[0]))
        
        sns.boxplot(data=df_data, x='Filtre', y=metric, hue='Type de Bruit', order=order)
        
        plt.title(titre, fontsize=16, fontweight='bold')
        plt.xticks(rotation=15, fontsize=11) # Rotation légère, texte plus grand
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(figures_dir, nom_fichier), dpi=300)
        plt.close()


    # GÉNÉRATION DES 4 GRAPHIQUES SÉPARÉS
    # Graphiques pour les filtres principaux (Comparaison directe avec le RC)
    tracer_boxplot(df_principaux, "PSNR", "Performances PSNR : Filtres Principaux vs Système RC", "boxplot_psnr_principaux.png")
    tracer_boxplot(df_principaux, "SSIM", "Performances SSIM : Filtres Principaux vs Système RC", "boxplot_ssim_principaux.png")
    
    # Graphiques pour les combinaisons HE/CLAHE (Pour la discussion/analyse)
    tracer_boxplot(df_bonus, "PSNR", "Impact de l'Amélioration de Contraste (HE/CLAHE)", "boxplot_psnr_bonus.png")
    tracer_boxplot(df_bonus, "SSIM", "Impact de l'Amélioration de Contraste (HE/CLAHE)", "boxplot_ssim_bonus.png")

    print(f" Graphiques séparés générés avec succès dans : {figures_dir}")