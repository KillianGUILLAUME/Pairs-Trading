import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from neural_nets.train import load_real_data
from neural_nets.signature_vf import CausalVelocityField

def generate_massive_paths_dataset(n_paths=1000, horizon=336, batch_size=50):
    """
    Génère des milliers de trajectoires indépendantes.
    Chaque trajectoire contient : 
      - 60h d'historique réel (Warmup pour vos filtres Kalman)
      - `horizon` heures de futur synthétique généré par l'IA.
    """
    print(f"🚀 Génération de {n_paths} chemins de {horizon}h (Total: {n_paths * horizon} heures)...")
    
    # 1. Résolution des chemins
    current_dir = os.path.abspath(os.getcwd())
    project_root = os.path.dirname(current_dir) if current_dir.endswith("notebooks") else current_dir
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 2. Chargement du modèle
    model_dir = os.path.join(project_root, "data", "models")
    latest_model = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])[-1]
    checkpoint = torch.load(os.path.join(model_dir, latest_model), map_location=device)
    
    scaler_mean = checkpoint["scaler"]["mean"]
    scaler_std = checkpoint["scaler"]["std"]
    safe_std = np.where(scaler_std == 0, 1e-8, scaler_std) 
    
    model = CausalVelocityField(data_dim=2, time_dim=64, sig_depth=3, hidden_dim=256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Chargement des données réelles pour piocher des graines
    data_dir = os.path.join(project_root, "data", "storage", "parquet")
    price_a, price_b = load_real_data("ZEC_USDT", "XRP_USDT", "1h", data_dir)
    
    window_size = 60
    valid_indices = np.arange(window_size, len(price_a) - 1)
    
    # Tirage au sort de `n_paths` points de départ historiques
    np.random.seed(42)
    selected_seeds = np.random.choice(valid_indices, size=n_paths, replace=True)

    all_paths = []

    # 4. Génération par Batch (pour exploiter le GPU/MPS)
    with torch.no_grad():
        for i in tqdm(range(0, n_paths, batch_size), desc="Génération des batchs"):
            batch_seeds = selected_seeds[i:i+batch_size]
            current_batch_size = len(batch_seeds)
            
            seeds_a, seeds_b, scaled_windows = [], [], []
            
            # Préparation du batch
            for idx in batch_seeds:
                window_a = price_a[idx-window_size:idx]
                window_b = price_b[idx-window_size:idx]
                seeds_a.append(window_a)
                seeds_b.append(window_b)
                
                log_ret_a = np.diff(np.log(window_a), prepend=0)
                log_ret_b = np.diff(np.log(window_b), prepend=0)
                raw_returns = np.stack([log_ret_a, log_ret_b], axis=1)
                scaled_returns = (raw_returns - scaler_mean) / safe_std
                scaled_windows.append(scaled_returns)
                
            batch_tensor = torch.tensor(np.array(scaled_windows), dtype=torch.float32).to(device)
            
            # Génération SDE
            generated_scaled = model.generate_trajectory(
                initial_window=batch_tensor,
                n_future_steps=horizon,
                n_ode_steps=10,
                solver="sde",     
                sigma_sde=0.5,
                martingale=True,  # Garde la co-intégration active
                target_drift=0.0
            )
            
            # Décodage et assemblage
            gen_np = generated_scaled.cpu().numpy()
            for j in range(current_batch_size):
                real_log_returns = (gen_np[j] * safe_std) + scaler_mean
                cumulative_returns = np.cumsum(real_log_returns, axis=0)
                
                fut_price_a = seeds_a[j][-1] * np.exp(cumulative_returns[:, 0])
                fut_price_b = seeds_b[j][-1] * np.exp(cumulative_returns[:, 1])
                
                # Concaténer l'historique (warmup) et le futur généré
                full_a = np.concatenate([seeds_a[j], fut_price_a])
                full_b = np.concatenate([seeds_b[j], fut_price_b])
                
                path_id = i + j
                is_synthetic = np.concatenate([np.zeros(window_size), np.ones(horizon)]).astype(bool)
                
                df_path = pd.DataFrame({
                    "path_id": path_id,
                    "step": np.arange(len(full_a)),
                    "price_a": full_a,
                    "price_b": full_b,
                    "is_synthetic": is_synthetic
                })
                all_paths.append(df_path)

    # 5. Fusionner en un seul gros DataFrame
    final_df = pd.concat(all_paths, ignore_index=True)
    
    # Sauvegarde optionnelle en parquet
    save_path = os.path.join(project_root, "data", "storage", "massive_synthetic_paths.parquet")
    final_df.to_parquet(save_path)
    
    print(f"\n✅ Dataset généré avec succès ! Shape: {final_df.shape}")
    print(f"💾 Sauvegardé dans : {save_path}")
    
    return final_df

if __name__ == "__main__":
    # Génère 1000 trajectoires de 336 heures (2 semaines) chacune.
    # Total = 396,000 lignes.
    df = generate_massive_paths_dataset(n_paths=1000, horizon=336, batch_size=50)
    print(df.head())