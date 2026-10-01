import os
import yaml
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from datetime import datetime

torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high')

# Importer les modules locaux
from data_loader import CryptoPairsDataset
from screened_loader import ScreenedPairsDataset
from neural_sde import GeneratorSDE
from losses import train_sde


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

import pandas as pd

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

    super_dataset_path = os.path.join(project_root, "data", "storage", "screened", "super_dataset_SDE_128.parquet")
    
    # On charge le dataset
    full_dataset = ScreenedPairsDataset(
        parquet_path=super_dataset_path,
        max_adf_pvalue=0.15, # Ton seuil choisi !
        sig_depth=model_cfg.get("sig_depth", 3)
    )
    
    # Train / Val Split (80% / 20%) strictement chronologique
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    
    # On passe num_workers=0 : le dataset tenant ENTIÈREMENT dans la RAM, 
    # lancer des process Python parallèles ne fait que saturer les IPC (pickling).
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_cfg.get("batch_size", 256), shuffle=True, drop_last=True, num_workers=0, pin_memory=(device == "cuda")
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=train_cfg.get("batch_size", 256), shuffle=False, drop_last=True, num_workers=0, pin_memory=(device == "cuda")
    )
    
    print(f"Datasets prêts : {train_size} Train | {val_size} Validation")

    # --- CALIBRATION DYNAMIQUE DE LA VOLATILITÉ (Production-Safe) ---
    print("🔍 Calibration automatique de la volatilité cible...")
    # On récupère un seul gros batch du Dataloader d'entraînement
    sample_batch = next(iter(train_dataloader))
    
    # Gestion de ta structure de tuple (real_paths, conditions, preputed)
    if isinstance(sample_batch, (list, tuple)):
        sample_paths = sample_batch[0]
    else:
        sample_paths = sample_batch
        
    # sample_paths est de dimension (Batch, Seq_Len, 2)
    # Calcul du spread exact tel qu'il sera vu par la SDE : log_pA - log_pB
    sample_spreads = sample_paths[:, :, 0] - sample_paths[:, :, 1]
    
    # Sécurité absolue : on évite un target_vol de 0.0 qui ferait exploser le log d'initialisation
    min_vol = 1e-4
    target_vols = []
    for i, batch in enumerate(train_dataloader):
        if i >= 5: break  # 5 batches suffisent
        paths = batch[0] if isinstance(batch, (list, tuple)) else batch
        spreads = paths[:, :, 0] - paths[:, :, 1]
        target_vols.append(spreads.std().item())
    target_vol = max(float(np.mean(target_vols)), min_vol)

    print(f"📊 Calibration SDE -> Volatilité cible: {target_vol:.5f} | Plancher: {min_vol:.5f}")

    auto_condition_dim = getattr(full_dataset, "condition_dim", 10)
    print(f"🧬 Dim. SDE générative dynamique : {auto_condition_dim} features.")

    use_wandb = False
    if os.getenv("WANDB_API_KEY"):
        print("🌊 Initialisation de Weights & Biases...")
        
        run_name = f"neural_sde_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project="pairs_trading",
            name=run_name,
            
            # Tags pour filtrer facilement dans l'UI
            tags=[
                "neural_sde",
                f"depth_{model_cfg.get('sig_depth', 3)}",
                f"hidden_{model_cfg.get('hidden_dim', 128)}",
                device,
            ],
            
            # Notes libres (visible dans l'UI)
            notes="Conditional Neural SDE + Kernel Signature MMD. Train sur crypto pairs screened.",
            
            # Config complète loggée (reproductibilité)
            config={
                **nn_config,  # ton yaml complet
                # Infos runtime
                "device":           device,
                "target_vol":       target_vol,
                "condition_dim":    auto_condition_dim,
                "train_size":       train_size,
                "val_size":         val_size,
                "dataset_path":     super_dataset_path,
                "torch_version":    torch.__version__,
                "cuda_available":   torch.cuda.is_available(),
                "gpu_name":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            },
        )
        
        # Définir les métriques summary (min/max dans le tableau de comparaison)
        wandb.define_metric("val/mmd_loss",  summary="min")
        wandb.define_metric("train/mmd_loss", summary="min")
        wandb.define_metric("diagnostics/vol_ratio", summary="last")
        
        # Step metric : epoch comme axe X pour tous les graphs
        wandb.define_metric("*", step_metric="epoch")
        
        use_wandb = True
    else:
        print("⚠️ W&B désactivé (pas de WANDB_API_KEY).")
    
    generator_sde = GeneratorSDE(
        data_dim=model_cfg.get("data_dim", 2),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        target_vol=target_vol,
        min_vol=min_vol,
        condition_dim=auto_condition_dim
    ).to(device)

    # Optimisation JIT: Les MLPs sont petits mais appelés 128x fois par le solveur SDE !
    # Le "kernel launch overhead" détruit les perfs. On fuse les kernels via torch.compile.
    if device == "cuda" and hasattr(torch, "compile"):
        # print("⚡ Fusing Drift & Diffusion Kernels avec torch.compile...")
        # On compile sélectivement les réseaux (fusion de couches) sans toucher à torchsde
        # qui a du dynamic control flow.
        # generator_sde.drift_net = torch.compile(generator_sde.drift_net)
        # generator_sde.diffusion_net = torch.compile(generator_sde.diffusion_net)
        print("⚡ Fusing Drift & Diffusion Kernels avec torch.compile... Desactivated")

    # 6. Lancer l'entraînement (Neural SDE avec Signature MMD)
    print("Démarrage de l'entraînement SDE... (Appuyez sur Ctrl+C pour arrêter, sauvegarder et détruire l'instance)")
    
    try:
        trained_model = train_sde(
            generator_sde=generator_sde,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            sig_mean=full_dataset.sig_mean,
            sig_std=full_dataset.sig_std,
            num_epochs=train_cfg.get("epochs", 500),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            device=device,
            sig_depth=model_cfg.get("sig_depth", 3),
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
    save_path = os.path.join(model_dir, f"neural_sde_{timestamp}_world_model.pt")
    
    # Sauvegarder les poids, la configuration et la standardisation (scaler)
    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "config": nn_config,
        "scale_factor": full_dataset.scale_factor, # LA CLÉ POUR LA PRODUCTION
        "global_std": full_dataset.global_std,     # Pour le debug
        "scaler": {
            "mean": full_dataset.mean,
            "std":  full_dataset.global_std,
        },
        "signature_scaler": {
            "mean": full_dataset.sig_mean,
            "std":  full_dataset.sig_std,
        },
    }, save_path)   
    
    print(f"Modèle entraîné et sauvegardé avec succès dans : {save_path}")

    # Terminer le run WandB propement
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()