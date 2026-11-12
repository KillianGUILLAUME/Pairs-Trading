"""
neural_nets/train.py

Entraînement de la Neural SDE conditionnelle (Kernel Signature MMD).

    python neural_nets/train.py                 # config/config.yaml
    python neural_nets/train.py --smoke         # 2 époques, 3 batches : vérifie que tout tourne

Le checkpoint final contient : poids, config, facteur d'échelle global, scalers de
signature, métadonnées de conditionnement (colonnes, z-score, spread initial),
paramètres de signature et dataset utilisé — tout ce qu'il faut pour que l'inférence
(`GenerativeSDEEngine`) reproduise exactement le prétraitement de l'entraînement.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neural_nets.losses import train_sde                      # noqa: E402
from neural_nets.neural_sde import GeneratorSDE               # noqa: E402
from neural_nets.screened_loader import ScreenedPairsDataset  # noqa: E402

torch.backends.cudnn.benchmark = True


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def chronological_split(dataset: ScreenedPairsDataset, train_frac: float, embargo_bars: int):
    """
    Split chronologique (le dataset est trié par timestamp_start) avec embargo : les
    fenêtres de validation qui chevauchent la fin du train (moins de `embargo_bars`
    barres après la dernière fenêtre de train) sont écartées.
    """
    n = len(dataset)
    train_size = int(train_frac * n)
    train_idx = list(range(0, train_size))
    if dataset.timestamps_start is None or embargo_bars <= 0:
        return train_idx, list(range(train_size, n))
    ts = dataset.timestamps_start.astype(np.int64)
    bar_ms = 3600 * 1000
    cutoff = ts[:train_size].max() + embargo_bars * bar_ms
    val_idx = [i for i in range(train_size, n) if ts[i] >= cutoff]
    return train_idx, val_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(project_root, "config", "config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Run minimal (2 époques, 3 batches) pour vérifier la chaîne")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Fichier de config introuvable : {args.config}")
    config = load_config(args.config)
    nn_config = config.get("neural_net", {})
    if not nn_config:
        raise ValueError("Le bloc 'neural_net' est manquant dans config.yaml.")

    model_cfg = nn_config["model"]
    train_cfg = nn_config["training"]
    data_cfg = nn_config.get("data", {})

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Appareil utilisé pour l'entraînement : {device.upper()} | seed={seed}")
    print("Modèle actif : Conditional Neural SDE (Drift + Diffusion), loss = Kernel Signature MMD (canal temps)")

    # 1. Données
    dataset_path = os.path.join(project_root, data_cfg.get("dataset_path", "data/storage/screened/super_dataset_SDE_128_causal.parquet"))
    sig_depth = int(model_cfg.get("sig_depth", 3))
    sig_time_augment = bool(model_cfg.get("sig_time_augment", True))
    sig_lead_lag = bool(model_cfg.get("sig_lead_lag", False))

    full_dataset = ScreenedPairsDataset(
        parquet_path=dataset_path,
        max_adf_pvalue=float(data_cfg.get("max_adf_pvalue", 0.15)),
        max_hurst=float(data_cfg.get("max_hurst", 1.0)),
        max_half_life=float(data_cfg.get("max_half_life", float("inf"))),
        filter_on=data_cfg.get("filter_on", "target"),
        sig_depth=sig_depth,
        sig_time_augment=sig_time_augment,
        sig_lead_lag=sig_lead_lag,
        device="cuda" if device == "cuda" else "cpu",
    )

    train_idx, val_idx = chronological_split(
        full_dataset, float(data_cfg.get("train_frac", 0.8)), int(data_cfg.get("embargo_bars", full_dataset.seq_len))
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    batch_size = int(train_cfg.get("batch_size", 256))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, pin_memory=(device == "cuda"))
    # Validation MÉLANGÉE (graine fixe) : le jeu est trié par date, des batches consécutifs ne couvrent qu'une période
    # étroite, ce qui gonflait le MMD de validation d'un facteur ~6 (0,19 contre 0,03 mesuré) et faussait l'arrêt anticipé.
    val_gen = torch.Generator().manual_seed(seed)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=val_gen,
                                num_workers=0, pin_memory=(device == "cuda"))
    print(f"Datasets prêts : {len(train_dataset)} Train | {len(val_dataset)} Validation (embargo {data_cfg.get('embargo_bars', full_dataset.seq_len)} barres)")
    print(f"🧬 Dim. de conditionnement : {full_dataset.condition_dim} ({', '.join(full_dataset.condition_columns)} + spread initial)")

    # 2. Modèle
    generator_sde = GeneratorSDE(
        data_dim=int(model_cfg.get("data_dim", 2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        min_vol=float(model_cfg.get("min_vol", 1e-4)),
        condition_dim=full_dataset.condition_dim,
        dt=float(model_cfg.get("dt_sde", 0.25)),
        solver=model_cfg.get("sde_solver", "euler"),
        latent_dim=int(model_cfg.get("latent_dim", 0)),
        noise=model_cfg.get("noise", "gaussian"),
        student_nu=float(model_cfg.get("student_nu", 6.0)),
        mean_reversion=bool(model_cfg.get("mean_reversion", False)),
    ).to(device)

    # 3. Run / W&B
    run_name = f"neural_sde_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    num_epochs = 2 if args.smoke else int(train_cfg.get("epochs", 500))
    max_batches = 3 if args.smoke else train_cfg.get("max_batches_per_epoch")

    checkpoint_extras = {
        "config": nn_config,
        "scale_factor": full_dataset.scale_factor,
        "global_std": full_dataset.global_std,
        "scaler": {"mean": full_dataset.mean, "std": full_dataset.global_std},
        "signature_scaler": {"mean": full_dataset.sig_mean, "std": full_dataset.sig_std},
        "signature": {"depth": sig_depth, "time_augment": sig_time_augment, "lead_lag": sig_lead_lag},
        "conditioning": full_dataset.conditioning_metadata(),
        "dataset_path": os.path.relpath(dataset_path, project_root),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_meta": full_dataset.dataset_meta,
        "seed": seed,
        "git_commit": _git_commit(),
        "run_name": run_name,
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    use_wandb = False
    if os.getenv("WANDB_API_KEY") and not args.smoke:
        import wandb
        print("🌊 Initialisation de Weights & Biases...")
        wandb.init(
            project="pairs_trading", name=run_name,
            tags=["neural_sde", f"depth_{sig_depth}", f"hidden_{model_cfg.get('hidden_dim', 128)}", device, "causal"],
            notes="Conditional Neural SDE + Kernel Signature MMD (canal temps). Conditionnement causal.",
            config={**nn_config, "device": device, "condition_dim": full_dataset.condition_dim,
                    "global_std": full_dataset.global_std, "train_size": len(train_dataset),
                    "val_size": len(val_dataset), "dataset_path": dataset_path, "seed": seed,
                    "torch_version": torch.__version__, "git_commit": checkpoint_extras["git_commit"]},
        )
        wandb.define_metric("val/mmd_loss", summary="min")
        wandb.define_metric("train/mmd_loss", summary="min")
        wandb.define_metric("*", step_metric="epoch")
        use_wandb = True
    else:
        print("⚠️ W&B désactivé (pas de WANDB_API_KEY ou mode --smoke).")

    model_dir = os.path.join(project_root, "data", "models")
    os.makedirs(model_dir, exist_ok=True)

    # 4. Entraînement
    print("Démarrage de l'entraînement SDE... (Ctrl+C pour arrêter et sauvegarder)")
    try:
        trained_model, best_path, history = train_sde(
            generator_sde=generator_sde,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            sig_mean=full_dataset.sig_mean,
            sig_std=full_dataset.sig_std,
            num_epochs=num_epochs,
            lr=float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            device=device,
            sig_depth=sig_depth,
            sig_time_augment=sig_time_augment,
            sig_lead_lag=sig_lead_lag,
            use_wandb=use_wandb,
            seed=seed,
            accum_iter=int(train_cfg.get("accum_iter", 4)),
            ema_decay=float(train_cfg.get("ema_decay", 0.999)),
            early_stop_patience=int(train_cfg.get("early_stop_patience", 70)),
            val_every=int(train_cfg.get("val_every", 1)),
            max_batches_per_epoch=max_batches,
            save_dir=model_dir,
            run_name=run_name,
            checkpoint_extras=checkpoint_extras,
        )
    except KeyboardInterrupt:
        print("\n🛑 Interruption (Ctrl+C) : sauvegarde de l'état courant...")
        trained_model, best_path, history = generator_sde, None, []

    # 5. Sauvegarde finale (même format que le meilleur checkpoint)
    save_path = os.path.join(model_dir, f"{run_name}_world_model.pt")
    torch.save({"model_state_dict": trained_model.state_dict(), **checkpoint_extras,
                "best_checkpoint": best_path, "history": history}, save_path)
    print(f"Modèle sauvegardé : {save_path}" + (f" | meilleur : {best_path}" if best_path else ""))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
