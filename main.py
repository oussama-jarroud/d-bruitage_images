import os
import argparse
from src import config
from src.dataset_builder import generate_noisy_datasets
from src.evaluator import run_evaluation_pipeline
from src.systeme_rc import preparer_dataset_rc
from src.visualizer import generer_graphiques_comparatifs

def setup_directories():
    print(" Vérification de l'arborescence...")
    dossiers = [config.RAW_DATA_DIR, config.NOISY_DATA_DIR, config.CSV_DIR, config.FIGURES_DIR, config.DL_OUTPUT_DIR]
    for d in dossiers:
        os.makedirs(d, exist_ok=True)

def main(args):
    print(" DÉMARRAGE DU PIPELINE - PROJET 2 \n")
    setup_directories()

    if args.step in ['all', 'data']:
        print("--- 1. Génération des jeux de données ---")
        generate_noisy_datasets(config.RAW_DATA_DIR, config.NOISY_DATA_DIR, config.VARIANCES_BRUIT)
        print()

    if args.step in ['all', 'eval']:
        print("--- 2. Évaluation des filtres classiques ---")
        run_evaluation_pipeline(
            config.RAW_DATA_DIR, 
            config.NOISY_DATA_DIR, 
            config.CSV_DIR, 
            config.TYPES_BRUIT, 
            variance_test=0.05
        )
        print()

    if args.step in ['all', 'rc']:
        print("--- 3. Exécution du Système Expert (RC Otsu) ---")
        preparer_dataset_rc(
            config.RAW_DATA_DIR, 
            config.NOISY_DATA_DIR, 
            config.DL_OUTPUT_DIR, 
            config.CSV_DIR,
            config.RC_K_TRANSITION
        )
        print()

    if args.step in ['all', 'viz']:
        print("--- 4. Génération des Graphiques ---")
        generer_graphiques_comparatifs(config.CSV_DIR, config.FIGURES_DIR)
        print()

    print(" Pipeline terminé !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # On autorise bien toutes les étapes ici :
    parser.add_argument('--step', type=str, default='all', choices=['data', 'eval', 'rc', 'viz', 'all'])
    args = parser.parse_args()
    main(args)