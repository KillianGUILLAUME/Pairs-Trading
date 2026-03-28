import os
import yaml
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

def load_real_data(pair_a="ZEC_USDT", pair_b="XRP_USDT", timeframe="1h", data_dir="data/storage/parquet"):
    """
    Charge les données historiques Parquet, les aligne sur le timestamp 
    pour éviter tout décalage temporel, et retourne les prix de clôture.
    """
    path_a = os.path.join(data_dir, timeframe, f"{pair_a}.parquet")
    path_b = os.path.join(data_dir, timeframe, f"{pair_b}.parquet")
    
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise FileNotFoundError(f"Données introuvables : {path_a} ou {path_b}. Lancez le pipeline de téléchargement.")
        
    print(f"Chargement des historiques {timeframe} pour {pair_a} et {pair_b}...")
    
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    
    df_a = df_a[['timestamp', 'close']].rename(columns={'close': 'close_a'})
    df_b = df_b[['timestamp', 'close']].rename(columns={'close': 'close_b'})
    
    df_merged = pd.merge(df_a, df_b, on='timestamp', how='inner').sort_values('timestamp')
    
    price_a = df_merged['close_a'].to_numpy()
    price_b = df_merged['close_b'].to_numpy()
    
    return price_a, price_b

# Importer les modules locaux
from signature_vf import CausalVelocityField

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def reconstruct_prices(log_returns: np.ndarray, init_prices: tuple) -> np.ndarray:
    """
    Reconstruit les prix bruts à partir des séquences de log-returns.
    Args:
        log_returns: (N_steps, 2)
        init_prices: (P0_A, P0_B)
    Returns:
        prices: (N_steps+1, 2)
    """
    prices = np.zeros((log_returns.shape[0] + 1, 2))
    prices[0] = init_prices
    
    # cumsum des log-returns = log(P_t) - log(P_0) -> P_t = P_0 * exp(cumsum)
    cumulative_returns = np.cumsum(log_returns, axis=0)
    prices[1:] = init_prices * np.exp(cumulative_returns)
    return prices

def generate_synthetic_paths(
    model_path: str,
    n_trajectories: int = 1000,
    n_steps: int = 24 * 30, # 1 mois de données horaire (720h)
    batch_size: int = 250,
):
    """
    Génère N trajectoires synthétiques via le modèle génératif causal.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Charger le modèle et ses métadonnées
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Chargement du modèle depuis {model_path} sur {device}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    nn_config = checkpoint["config"]
    scaler = checkpoint["scaler"]
    
    mean = torch.tensor(scaler["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(scaler["std"], device=device, dtype=torch.float32)
    
    # 2. Instancier le modèle
    model_cfg = nn_config["model"]
    model = CausalVelocityField(
        data_dim=model_cfg.get("data_dim", 2),
        time_dim=model_cfg.get("time_dim", 64),
        sig_depth=model_cfg.get("sig_depth", 3),
        hidden_dim=model_cfg.get("hidden_dim", 256)
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Modèle prêt.")

    # 3. Préparer une fenêtre initiale (past_window)
    # En pratique, on pioche ça dans les VRAIES données historiques 
    # pour générer N trajectoires à partir du contexte le plus récent.
    data_dir = os.path.join(project_root, "data", "storage", "parquet")
    price_a, price_b = load_real_data(data_dir=data_dir)
    
    log_ret_a = np.diff(np.log(price_a))
    log_ret_b = np.diff(np.log(price_b))
    raw_returns = np.stack([log_ret_a, log_ret_b], axis=1)
    
    # Standardisation manuelle
    scaled_returns = (raw_returns - scaler["mean"]) / (scaler["std"] + 1e-8)
    scaled_returns = torch.tensor(scaled_returns, dtype=torch.float32, device=device)
    
    window_size = nn_config.get("training", {}).get("window_size", 60)
    
    # On prend la dernière fenêtre de N jours réels comme point de départ
    actual_history = scaled_returns[-window_size:].unsqueeze(0)
    
    # On duplique ce contexte pour chaque trajectoire du batch
    dummy_history = actual_history.expand(batch_size, -1, -1)
    
    gen_config = nn_config.get("generation", {})
    n_ode_steps = gen_config.get("n_ode_steps", 20)
    n_mart_samples = gen_config.get("n_mart_samples", 8)
    solver = gen_config.get("solver", "ode")
    sigma_sde = gen_config.get("sigma_sde", 0.5)
    
    # 4. Générer par batchs
    all_generated_paths = []
    
    print(f"Génération de {n_trajectories} trajectoires de {n_steps} pas de temps...")
    print(f"Paramètres : Solver={solver.upper()}, Martingale=True, N_ODE={n_ode_steps}")
    # On itère par batchs pour gérer la RAM GPU
    for i in range(0, n_trajectories, batch_size):
        actual_batch_size = min(batch_size, n_trajectories - i)
        
        # Extrait le bon nombre de séquences historiques
        batch_history = dummy_history[:actual_batch_size]
        
        # Generation autoregressive + PCFM Martingale
        with torch.no_grad():
            gen_scaled = model.generate_trajectory(
                initial_window=batch_history,
                n_future_steps=n_steps,
                n_ode_steps=n_ode_steps,
                martingale=True,  # Projette sur la martingale (PCFM)
                n_mart_samples=n_mart_samples, 
                target_drift=0.0, # Assumes pure martingale for synthetic stress tests
                solver=solver,
                sigma_sde=sigma_sde
            )
            
        # Inverse-transform (déstandardisation vers la vraie volatilité)
        # gen_scaled: (batch, n_steps, 2)
        gen_returns = (gen_scaled * std) + mean
        all_generated_paths.append(gen_returns.cpu().numpy())
        
        print(f"[{i + actual_batch_size}/{n_trajectories}] Trajectoires générées.")

    # Concaténer tout
    final_returns = np.concatenate(all_generated_paths, axis=0) # (1000, 720, 2)
    
    # 5. Sauvegarde
    out_dir = os.path.join(project_root, "data", "storage", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"synthetic_scenarios_{timestamp}.npy")
    
    # Pour un P&L Vectorisé, on sauvegarde souvent le tenseur Numpy (Rapide + 3D)
    # ou on flatten() en Parquet (Trajectoire_ID, Step, P_A, P_B).
    # Ici on sauvegarde en .npy direct pour l'accès Monte Carlo.
    np.save(out_file, final_returns)
    print(f"Données enregistrées dans : {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générer des scénarios de test (PCFM).")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le fichier .pt")
    parser.add_argument("--n_traj", type=int, default=1000, help="Nombre de trajectoires à générer")
    parser.add_argument("--n_steps", type=int, default=720, help="Nombre de pas (heures) par trajectoire")
    args = parser.parse_args()
    
    # Vérification fichier modèle
    if not os.path.exists(args.model):
        print(f"Erreur : le modèle {args.model} n'existe pas.")
        exit(1)
        
    generate_synthetic_paths(
        model_path=args.model,
        n_trajectories=args.n_traj,
        n_steps=args.n_steps,
    )
