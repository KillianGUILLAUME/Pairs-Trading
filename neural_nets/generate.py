import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

from neural_sde import GeneratorSDE # Ton fichier SDE

def load_trained_model(model_path: str, device: str):
    """Charge le modèle, la config et le scaler."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    scaler = checkpoint["scaler"]
    
    model_cfg = config["model"]
    
    # On instancie le modèle avec les paramètres de la config
    model = GeneratorSDE(
        data_dim=model_cfg.get("data_dim", 2),
        hidden_dim=model_cfg.get("hidden_dim", 32),
        target_vol=1.0, # A ajuster si tu l'as sauvegardé dans le checkpoint
        min_vol=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # Mode Inférence, très important
    
    return model, scaler

def filter_cointegrated_paths(paths_np: np.ndarray, max_pvalue: float = 0.05) -> list:
    """
    Test de Dickey-Fuller Augmenté (ADF) sur le spread de chaque chemin.
    paths_np : Array numpy de taille [n_paths, seq_len, 2]
    """
    accepted_indices = []
    
    # On itère sur chaque chemin généré
    for i in range(paths_np.shape[0]):
        path = paths_np[i]
        
        # On recrée le proxy du spread (log_a - log_b)
        # (Attention : path contient déjà les log-prix dénormalisés)
        spread = path[:, 0] - path[:, 1]
        
        try:
            # Test ADF pour vérifier si le spread est stationnaire (Mean-Reverting)
            adf_result = adfuller(spread, maxlag=1) 
            p_value = adf_result[1]
            
            # Si p-value < 0.05, on rejette l'hypothèse de non-stationnarité
            if p_value < max_pvalue:
                accepted_indices.append(i)
        except Exception:
            # Sécurité si le test ADF échoue (ex: NaN à cause du SDE)
            pass
            
    return accepted_indices

def main(price_a, price_b, seq_len=2000, n_paths_to_generate=5000, batch_size=500):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Génération sur : {device.upper()}")
    
    # --- 1. Paramètres de Génération ---
    model_path = "data/models/best_val_sde.pt" #to adapt
    output_dir = "data/synthetic_paths"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. Chargement ---
    model, scaler = load_trained_model(model_path, device)
    std = scaler["std"]
    mean = scaler["mean"]
    
    # --- 3. Point de départ (y0) ---
    log_a_init = np.log(price_a[-1]) 
    log_b_init = np.log(price_b[-1]) 
    
    # On doit le standardiser comme pendant l'entraînement !
    scaled_a_init = (log_a_init - mean[0]) / std[0]
    scaled_b_init = (log_b_init - mean[1]) / std[1]
    
    # Tenseur de départ [1, 2]
    y0 = torch.tensor([[scaled_a_init, scaled_b_init]], dtype=torch.float32, device=device)
    
    print(f"Génération de {n_paths_to_generate} chemins de {seq_len} barres...")
    
    all_accepted_paths = []
    
    # --- 4. Boucle de Batching (GPU) ---
    for batch_start in tqdm(range(0, n_paths_to_generate, batch_size)):
        current_batch_size = min(batch_size, n_paths_to_generate - batch_start)
        
        with torch.no_grad():
            # Génération SDE
            # Output: [1, batch_size, seq_len, 2] -> On vire la 1ère dim avec [0]
            generated_scaled = model.generate_synthetic_paths(
                y0=y0, 
                seq_len=seq_len, 
                n_paths=current_batch_size
            )[0].cpu().numpy() # [batch_size, seq_len, 2]
            
        # --- 5. Dénormalisation & Reverse Log ---
        # 1. On remet la moyenne et l'écart-type
        generated_log = (generated_scaled * std) + mean
        
        # 2. On passe à l'exponentielle pour avoir les VRAIS prix en $ !
        generated_prices = np.exp(generated_log)
        
        # --- 6. Filtrage Quant (Test ADF) ---
        # On fait le test sur les log-prix (generated_log) car le spread financier
        # se calcule toujours sur les logs pour la cointégration.
        accepted_idx = filter_cointegrated_paths(generated_log, max_pvalue=0.05)
        
        # On stocke les VRAIS prix des chemins acceptés
        if accepted_idx:
            all_accepted_paths.append(generated_prices[accepted_idx])
            
    # --- 7. Sauvegarde ---
    if not all_accepted_paths:
        print(f"❌ Aucun chemin n'a passé le test ADF. Le SDE diverge trop sur {seq_len} barres.")
        return
        
    final_paths = np.concatenate(all_accepted_paths, axis=0)
    acceptance_rate = (len(final_paths) / n_paths_to_generate) * 100
    print(f"✅ Génération terminée ! {len(final_paths)} chemins gardés (Taux d'acceptation : {acceptance_rate:.1f}%)")
    
    # On sauvegarde chaque chemin dans un gros fichier Parquet (format long)
    print("Sauvegarde en cours...")
    records = []
    for path_id in range(len(final_paths)):
        for t in range(seq_len):
            records.append({
                "path_id": path_id,
                "step": t,
                "price_a": final_paths[path_id, t, 0],
                "price_b": final_paths[path_id, t, 1]
            })
            
    df_synthetic = pd.DataFrame(records)
    save_path = os.path.join(output_dir, f"synthetic_dataset_{seq_len}bars.parquet")
    df_synthetic.to_parquet(save_path, engine="pyarrow")
    print(f"💾 Dataset sauvegardé : {save_path}")

if __name__ == "__main__":
    main()