import os
import yaml
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from datetime import datetime

# Importer les modules locaux
from data_loader import CryptoPairsDataset
from neural_sde import GeneratorSDE
from losses import train_sde

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

import pandas as pd

def load_real_data(pair_a="ZEC_USDT", pair_b="XRP_USDT", timeframe="1h", data_dir="data/storage/parquet"):
    """
    Charge les données historiques Parquet, les aligne sur le timestamp 
    pour éviter tout décalage temporel, et retourne les prix de clôture.
    """
    path_a = os.path.join(data_dir, timeframe, f"{pair_a}.parquet")
    path_b = os.path.join(data_dir, timeframe, f"{pair_b}.parquet")
    
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise FileNotFoundError(f"Données introuvables : {path_a} ou {path_b}. Lancez le pipeline de téléchargement d'abord.")
        
    print(f"Chargement des historiques {timeframe} pour {pair_a} et {pair_b}...")
    
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    
    # On renomme la colonne 'close' pour éviter la collision
    df_a = df_a[['timestamp', 'close']].rename(columns={'close': 'close_a'})
    df_b = df_b[['timestamp', 'close']].rename(columns={'close': 'close_b'})
    
    # On aligne strictement les deux séries temporelles (Inner Join)
    df_merged = pd.merge(df_a, df_b, on='timestamp', how='inner').sort_values('timestamp')
    
    price_a = df_merged['close_a'].to_numpy()
    price_b = df_merged['close_b'].to_numpy()
    
    print(f"Séries temporelles alignées : {len(price_a)} points de données.")
    return price_a, price_b

def main():
    # 1. Charger la configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de config introuvable : {config_path}")
    
    config = load_config(config_path)
    nn_config = config.get("neural_net", {})
    if not nn_config:
        raise ValueError("Le bloc 'neural_net' est manquant dans config.yaml.")
    
    model_cfg = nn_config["model"]
    train_cfg = nn_config["training"]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Appareil utilisé pour l'entraînement : {device.upper()}")
    
    print(f"Modèle Actif : Conditional Neural SDE (Drift + Diffusion)")

    # 2. Préparer les données
    # On charge ZEC et XRP en 1h par defaut (meilleure cointégration)
    data_dir = os.path.join(project_root, "data", "storage", "parquet")
    price_a, price_b = load_real_data(
        pair_a="ZEC_USDT", 
        pair_b="XRP_USDT", 
        timeframe="1h",
        data_dir=data_dir
    )
    
    # Train / Val Split (80% / 20%) strictement chronologique
    split_idx = int(len(price_a) * 0.8)
    train_pa, val_pa = price_a[:split_idx], price_a[split_idx:]
    train_pb, val_pb = price_b[:split_idx], price_b[split_idx:]
    
    train_dataset = CryptoPairsDataset(
        price_a=train_pa, 
        price_b=train_pb, 
        seq_len=train_cfg.get("seq_len", 128),
        sig_depth=model_cfg.get("sig_depth", 4),
        compute_signature=True
    )
    
    val_dataset = CryptoPairsDataset(
        price_a=val_pa, 
        price_b=val_pb, 
        seq_len=train_cfg.get("seq_len", 128),
        sig_depth=model_cfg.get("sig_depth", 4),
        compute_signature=True,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.get("batch_size", 512), 
        shuffle=True, 
        drop_last=True,
        num_workers=4,
        pin_memory=(device == "cuda")
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 512),
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=(device == "cuda")
    )
    
    print(f"Datasets prêts : {len(train_dataset)} Train | {len(val_dataset)} Validation")

    # 4. Initialisation de Weights & Biases (W&B)
    use_wandb = False
    if os.getenv("WANDB_API_KEY"):
        print("🌊 Initialisation de Weights & Biases...")
        wandb.init(
            project="pairs_trading",
            name=f"generative_model_neural_sde_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=nn_config
        )
        use_wandb = True
    else:
        print("⚠️ W&B désactivé (pas de variable d'environnement WANDB_API_KEY trouvée).")
    
    # Volatilité moyenne historique (target_vol)
    # On prend la moyenne des écart-types sur chaque dimension
    target_vol = float(np.std(train_dataset.scaled_returns, axis=0).mean())
    min_vol = target_vol * 0.1

    print(f"📊 Calibration SDE -> Volatilité cible: {target_vol:.5f} | Plancher: {min_vol:.5f}")

    generator_sde = GeneratorSDE(
        data_dim=model_cfg.get("data_dim", 2),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        target_vol=target_vol,
        min_vol=min_vol
    ).to(device)

    # Optimisation JIT: Les MLPs sont petits mais appelés 128x fois par le solveur SDE !
    # Le "kernel launch overhead" détruit les perfs. On fuse les kernels via torch.compile.
    # if device == "cuda" and hasattr(torch, "compile"):
    #     print("⚡ Fusing Drift & Diffusion Kernels avec torch.compile...")
    #     # On compile sélectivement les réseaux (fusion de couches) sans toucher à torchsde
    #     # qui a du dynamic control flow.
    #     generator_sde.drift_net = torch.compile(generator_sde.drift_net, mode="reduce-overhead")
    #     generator_sde.diffusion_net = torch.compile(generator_sde.diffusion_net, mode="reduce-overhead")

    # 6. Lancer l'entraînement (Neural SDE avec Signature MMD)
    print("Démarrage de l'entraînement SDE... (Appuyez sur Ctrl+C pour arrêter, sauvegarder et détruire l'instance)")
    
    try:
        trained_model = train_sde(
            generator_sde=generator_sde,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            num_epochs=train_cfg.get("epochs", 500),
            lr=train_cfg.get("learning_rate", 1e-4),
            device=device,
            sig_depth=model_cfg.get("sig_depth", 4),
            use_wandb=use_wandb
        )
    except KeyboardInterrupt:
        print("\n🛑 Interruption détectée (Ctrl+C). Arrêt de l'entraînement...")
        print("💾 L'état actuel de la SDE va être sauvegardé avant destruction...")
        trained_model = generator_sde
    
    # 7. Sauvegarder le modèle
    model_dir = os.path.join(project_root, "data", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(model_dir, f"neural_sde_{timestamp}_ZEC_USDTxXRP_USDT.pt")
    
    # Sauvegarder les poids, la configuration et la standardisation (scaler)
    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "config": nn_config,
        "scaler": {
            "mean": train_dataset.mean,
            "std": train_dataset.std
        }
    }, save_path)
    
    print(f"Modèle entraîné et sauvegardé avec succès dans : {save_path}")

    # Terminer le run WandB propement
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
