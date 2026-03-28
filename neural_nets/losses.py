import torch
import torch.nn.functional as F
# import signatory
import wandb

# ============================================
# Loss pour entrainer le velocity field
# ============================================

class ConditionalFlowMatchingLoss:
    """
    Loss pour entraîner le velocity field

    Formulation :
        À chaque pas de temps physique t, on transporte un bruit source
        Δ₀ ~ N(0, I) vers l'incrément réel Δ₁ = X_{t+1} - X_t.
        Le velocity field est conditionné par la signature du passé Sig_{t-w,t},
        ce qui garantit la causalité : aucune information future n'est utilisée.

        Le temps de diffusion τ ∈ [0, 1] est distinct du temps physique t.
        L'interpolant est : Δ_τ = (1 - τ) Δ₀ + τ Δ₁
        La target velocity est : v* = Δ₁ - Δ₀ = dΔ_τ/dτ

    La loss est identique algébriquement au CFM standard, mais la sémantique
    change : x₁ est un incrément local (le prochain log-return), pas un
    change : x₁ est un incrément local (le prochain log-return), pas un
    état terminal lointain. Cela élimine le biais anticipatif.
    """
    def __init__(self, velocity_field, interpolant):
        self.velocity_field = velocity_field
        self.interpolant = interpolant
    
    def __call__(self, x_0, x_1,precomputed_sig=None): 
        """
        Args:
            x_0: (B, data_dim) - Bruit source Δ₀ ~ N(0, I)
            x_1: (B, data_dim) - Log-return cible Δ₁ = log(P_{t+1}/P_t) (standardisé)
            precomputed_sig: (B, sig_dim) - Optionnel.
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # τ ~ U[0, 1] (temps de diffusion, distinct du temps physique)
        t = torch.rand(batch_size, 1, device=device)
        
        # Interpolation via la classe fournie (Linear, BrownianBridge, etc.)
        x_t, v_target = self.interpolant.calc_xt_ut(x_0, x_1, t)
        
        # Prédiction du réseau conditionnée par Sig(past_path)
        v_pred = self.velocity_field(x_t, t, precomputed_sig=precomputed_sig) 
        
        # Loss MSE
        loss = F.mse_loss(v_pred, v_target)
        
        return loss

# ============================================
# Bridge Matching Loss
# ============================================

class BridgeMatchingLoss:
    """
    Loss de Bridge Matching

    Formulation :
        Comme pour le CFM, on transporte un bruit Δ₀ vers l'incrément Δ₁,
        mais en échantillonnant Δ_τ le long d'un Brownian Bridge :
            Δ_τ = (1 - τ) Δ₀ + τ Δ₁ + σ √(τ(1 - τ)) Z

        La dérive analytique cible est :
            u*(Δ_τ, τ | Δ₁) = (Δ₁ - Δ_τ) / (1 - τ)

        Le σ > 0 introduit de la stochasticité dans le sampling de Δ_τ
        pendant l'entraînement, ce qui régularise l'apprentissage et
        permet de modéliser l'incertitude intrinsèque des marchés.

    Ref: Shi, De Bortoli, Campbell, Doucet — NeurIPS 2023, Section 3.2.
    """
    def __init__(self, drift_network, sigma: float = 0.5):
        self.drift_network = drift_network
        self.sigma = sigma

    def __call__(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: (B, data_dim) - Bruit source Δ₀ ~ N(0, I)
            x1: (B, data_dim) - Log-return cible Δ₁ = log(P_{t+1}/P_t)
        """
        batch_size = x0.shape[0]
        device = x0.device

        # τ ~ U[0, 1-ε] (temps de diffusion)
        eps = 1e-3
        t = torch.rand(batch_size, 1, device=device) * (1.0 - eps)

        # Brownian Bridge : Δ_τ = (1-τ)Δ₀ + τΔ₁ + σ√(τ(1-τ)) Z
        mean_t = (1.0 - t) * x0 + t * x1
        std_t = self.sigma * torch.sqrt(t * (1.0 - t) + 1e-8)
        z = torch.randn_like(x0)
        x_t = mean_t + std_t * z

        # Dérive analytique cible : u* = (Δ₁ - Δ_τ) / (1 - τ)
        denom = torch.clamp(1.0 - t, min=eps)
        u_target = (x1 - x_t) / denom

        # Prédiction du réseau conditionnée par Sig(past_path)
        u_pred = self.drift_network(x_t, t)

        # Loss MSE
        loss = F.mse_loss(u_pred, u_target)

        return loss

# ============================================
# Régularisation entropique (anti mode collapse)
# ============================================

class EntropicCFMLoss:
    """
    CFM Loss avec régularisation entropique pour prévenir le mode collapse.

    Le Flow Matching standard minimise :
        L_CFM = E[ ||v_θ(Δ_τ, τ, Sig) - v*||² ]

    Le mode collapse survient quand v_θ prédit toujours la même vélocité
    indépendamment du bruit source Δ₀, effondrant la diversité des chemins
    générés. La régularisation entropique ajoute un terme qui pénalise
    les prédictions trop concentrées (variance trop faible dans le batch).

    Loss totale :
        L = L_CFM + λ_H · L_entropy

    où L_entropy = max(0, ε_min - Var_B[v_θ])  (hinge sur la variance)

    Quand la variance des prédictions reste au-dessus de ε_min, la
    régularisation est inactive. Elle ne s'active que si le modèle
    commence à collapser.

    Ref: Tong et al. (2024) — entropic regularization for flow matching.
    """

    def __init__(
        self,
        velocity_field,
        interpolant,
        lambda_entropy: float = 0.1,
        min_variance: float = 0.01,
    ):
        self.velocity_field = velocity_field
        self.interpolant = interpolant
        self.lambda_entropy = lambda_entropy
        self.min_variance = min_variance

    def __call__(self, x_0, x_1):
        """
        Args:
            x_0: (B, data_dim) - Bruit source Δ₀ ~ N(0, I)
            x_1: (B, data_dim) - Log-return cible Δ₁ = log(P_{t+1}/P_t) (standardisé)

        Returns:
            loss: scalar — L_CFM + λ · L_entropy
            metrics: dict — {cfm_loss, entropy_loss, pred_variance} pour monitoring
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # τ ~ U[0, 1]
        t = torch.rand(batch_size, 1, 1, device=device)

        # Interpolation via la classe fournie
        x_t, v_target = self.interpolant.calc_xt_ut(x_0, x_1, t)

        # Prédiction
        v_pred = self.velocity_field(x_t, t)

        # --- L_CFM ---
        cfm_loss = F.mse_loss(v_pred, v_target)

        # --- L_entropy (hinge sur la variance des prédictions) ---
        pred_var = v_pred.var(dim=0).mean()  # variance moyenne sur le batch
        entropy_loss = F.relu(self.min_variance - pred_var)

        # --- Loss totale ---
        total_loss = cfm_loss + self.lambda_entropy * entropy_loss

        metrics = {
            "cfm_loss": cfm_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "pred_variance": pred_var.item(),
        }

        return total_loss, metrics


# ============================================
# Fonction d'entraînement (La boucle principale)
# ============================================

def train_teacher(
    velocity_field,
    data_loader,
    interpolant,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    device: str = 'cuda',
    lambda_entropy: float = 0.0,
    min_variance: float = 0.01,
    use_wandb: bool = False
):
    """
    Entraîne le velocity field causal avec Conditional Flow Matching autorégressif.

    Le data_loader fournit des paires (past_path, Δ₁) où :
        - past_path : (B, window_size, 2) — fenêtre glissante de log-returns standardisés
        - Δ₁ : (B, 2) — le prochain log-return (incrément cible)

    À chaque batch, on tire Δ₀ ~ N(0, I) comme bruit source.
    Le velocity field apprend à transporter Δ₀ → Δ₁ conditionnellement à Sig(past_path).

    Args:
        lambda_entropy: poids de la régularisation entropique (0 = désactivée).
                        Valeur recommandée : 0.1.
        min_variance: seuil minimal de variance des prédictions (hinge).
    """
    use_entropic = lambda_entropy > 0.0

    if use_entropic:
        loss_fn = EntropicCFMLoss(
            velocity_field,
            interpolant=interpolant,
            lambda_entropy=lambda_entropy,
            min_variance=min_variance,
        )
    else:
        loss_fn = ConditionalFlowMatchingLoss(velocity_field, interpolant=interpolant)

    optimizer = torch.optim.AdamW(velocity_field.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      
        factor=0.5,     
        patience=50,     
        verbose=True     
    )

    # Tensor Cores Optimization (AMP)
    is_cuda = getattr(device, 'type', str(device)) == "cuda"
    use_amp = is_cuda 

    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:
        scaler = None

    velocity_field.to(device)
    velocity_field.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cfm = 0.0
        epoch_ent = 0.0
        num_batches = 0

        for batch in data_loader:
            if isinstance(batch, list) or isinstance(batch, tuple):
                x_1 = batch[0].to(device)
            else:
                x_1 = batch.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device if device != "mps" else "cpu", enabled=False):
                # 1. Générer x₀ via l'interpolant (Data Augmentation)
                if hasattr(interpolant, 'sample_x0'):
                    x_0 = interpolant.sample_x0(x_1.shape)
                else:
                    x_0 = torch.randn_like(x_1)

                x_1 = torch.nan_to_num(x_1, nan=0.0, posinf=1e4, neginf=-1e4)

                if use_entropic:
                    loss, metrics = loss_fn(x_0, x_1)
                    epoch_cfm += metrics["cfm_loss"]
                    epoch_ent += metrics["entropy_loss"]
                else:
                    loss = loss_fn(x_0, x_1)

            if scaler is not None:
                # ==========================================
                # AWS / CUDA PATH (With AMP & Scaler)
                # ==========================================
                scaler.scale(loss).backward()
                
                # Unscale the gradients back to their normal size before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(velocity_field.parameters(), max_norm=1.0)
                
                # Step and update
                scaler.step(optimizer)
                scaler.update()
                
            else:
                # ==========================================
                # MAC / MPS / CPU PATH (Standard PyTorch)
                # ==========================================
                loss.backward()
                
                # Gradients are already at normal scale, just clip them directly
                torch.nn.utils.clip_grad_norm_(velocity_field.parameters(), max_norm=1.0)
                
                # Step the optimizer
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0:
            avg = epoch_loss / num_batches
            if use_entropic:
                avg_cfm = epoch_cfm / num_batches
                avg_ent = epoch_ent / num_batches
                print(
                    f"Epoch {epoch:04d}/{num_epochs}, "
                    f"Total: {avg:.6f}, CFM: {avg_cfm:.6f}, Entropy: {avg_ent:.6f}"
                )
                if use_wandb:
                    current_lr = optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train/total_loss": avg,
                        "train/cfm_loss": avg_cfm,
                        "train/entropy_loss": avg_ent,
                        "epoch": epoch,
                        "train/learning_rate": current_lr
                    })
            else:
                print(f"Epoch {epoch:04d}/{num_epochs}, CFM Loss: {avg:.6f}")
                if use_wandb:
                    
                    wandb.log({
                        "train/total_loss": avg,
                        "epoch": epoch
                    })

    return velocity_field


# ============================================
# Neural SDE : Signature MMD Loss
# ============================================

class SignatureMMDLoss:
    """
    Maximum Mean Discrepancy (MMD) sur les Path Signatures.
    
    Calcule la distance L2 entre l'Espérance de la signature (Moment Matching).
    Comprend désormais la projection Unitaire Relative (Relative Error) pour stabiliser les gradients.
    """
    def __init__(self, depth: int = 4):
        self.depth = depth

    def __call__(self, real_paths: torch.Tensor, generated_paths: torch.Tensor) -> torch.Tensor:

        sig_real = signatory.signature(real_paths, self.depth)
        sig_gen = signatory.signature(generated_paths, self.depth)
        
        mean_sig_real = sig_real.mean(dim=0)
        mean_sig_gen = sig_gen.mean(dim=0)
        
        raw_dist = torch.norm(mean_sig_real - mean_sig_gen, p=2)
        real_norm = torch.norm(mean_sig_real, p=2).detach().clamp(min=1e-6)
        
        return raw_dist / real_norm


class KernelSignatureMMDLoss:
    """
    Maximum Mean Discrepancy (MMD) sur les Path Signatures avec un noyau Gaussien (RBF).
    
    Contrairement à la distance L2 sur les moyennes, le Kernel MMD compare toutes 
    les paires possibles (réel-réel, gen-gen, et croisées réel-gen) via une 
    matrice de Gram. Cela force la SDE à répliquer l'intégralité de la distribution 
    (Moment Matching implicite d'ordre infini sur l'espace des signatures).
    """
    def __init__(self, depth: int = 4, sigma: float = 1.0):
        self.depth = depth
        self.sigma = sigma
        
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calcule k(x, y) = exp( - ||x - y||^2 / (2 * sigma^2) ) de manière vectorisée.
        Optimisation Extrême : Puisque `x` et `y` sont projetés sur l'hypersphère (Norme=1),
        on sait mathématiquement que ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y> = 2 - 2<x,y>.
        Cela évite de recalculer les normes et divise le coût de calcul matriciel par 2 !
        """
        # Produit scalaire <x, y>
        xy_inner = torch.mm(x, y.transpose(0, 1))
        
        # Distance au carré sur la sphère unitaire
        dist_sq = 2.0 - 2.0 * xy_inner
        
        # Clamp symbolique pour les flottants
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        return torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

    def __call__(self, real_paths: torch.Tensor, generated_paths: torch.Tensor) -> torch.Tensor:
        

        sig_real = signatory.signature(real_paths, self.depth)
        sig_gen = signatory.signature(generated_paths, self.depth)
        
        # Le Clamp interne : modifie 0.0 en 1e-8, mais laisse 1.5 intact (sans faire 1.50000001)
        norm_real = torch.sqrt(torch.clamp(torch.sum(sig_real ** 2, dim=1, keepdim=True), min=1e-8))
        sig_real_norm = sig_real / norm_real
        
        norm_gen = torch.sqrt(torch.clamp(torch.sum(sig_gen ** 2, dim=1, keepdim=True), min=1e-8))
        sig_gen_norm = sig_gen / norm_gen
        
        k_xx = self.rbf_kernel(sig_real_norm, sig_real_norm)
        k_yy = self.rbf_kernel(sig_gen_norm, sig_gen_norm)
        k_xy = self.rbf_kernel(sig_real_norm, sig_gen_norm)
        
        mmd_sq = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        
        # Même chose à la fin
        return torch.sqrt(torch.clamp(mmd_sq, min=1e-8))


# ============================================
# Fonction d'entraînement (SDE)
# ============================================

def train_sde(
    generator_sde,
    train_loader,
    val_loader,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    device: str = 'cuda',
    sig_depth: int = 4,
    use_wandb: bool = False
):
    """
    Boucle d'entraînement SDE avec Signature MMD, Validation, Plots et Early Stopping.
    """
    import os
    import matplotlib.pyplot as plt
    
    generator_sde.to(device)
    
    loss_fn = KernelSignatureMMDLoss(depth=sig_depth, sigma=1.0)
    optimizer = torch.optim.AdamW(generator_sde.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      
        factor=0.5,     
        patience=20,     
        verbose=True     
    )

    # Paramètres d'Early Stopping
    best_val_loss = float('inf')
    early_stop_patience = 70
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        generator_sde.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                # non_blocking=True est crucial avec pin_memory=True :
                # ça permet au GPU de charger le prochain batch PENDANT qu'il calcule le précédent !
                x_1 = batch[0].to(device, non_blocking=True)
            else:
                x_1 = batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # x_1 contient les log-returns (incréments).
            # La SDE intègre des chemins, on transforme les cibles en chemins cumulés :
            # shape (B, L, Dim)
            real_paths = torch.cumsum(x_1, dim=1)
            
            # Pour que le temps 0 soit exactement à 0.0
            zeros = torch.zeros(real_paths.size(0), 1, real_paths.size(2), device=device)
            real_paths = torch.cat([zeros, real_paths], dim=1) # (B, L+1, Dim)
            
            batch_size, seq_len_plus_1, data_dim = real_paths.shape

            # Génération par la SDE
            ts = torch.linspace(0.0, float(seq_len_plus_1 - 1), seq_len_plus_1, device=device)
            y0 = torch.zeros(batch_size, data_dim, device=device)

            # Résolution Differentiable de la SDE
            generated_paths = generator_sde(y0, ts)
            
            # Signature MMD
            loss = loss_fn(real_paths, generated_paths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_sde.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        # ==========================================
        # VALIDATION & EARLY STOPPING (Tous les 10 Epochs)
        # ==========================================
        if epoch % 10 == 0:
            generator_sde.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x_1_val = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    
                    real_paths_val = torch.cumsum(x_1_val, dim=1)
                    zeros_val = torch.zeros(real_paths_val.size(0), 1, real_paths_val.size(2), device=device)
                    real_paths_val = torch.cat([zeros_val, real_paths_val], dim=1)
                    
                    b_size, seq_len_val, d_dim = real_paths_val.shape
                    
                    ts_val = torch.linspace(0.0, float(seq_len_val - 1), seq_len_val, device=device)
                    y0_val = torch.zeros(b_size, d_dim, device=device)
                    
                    generated_paths_val = generator_sde(y0_val, ts_val)
                    v_loss = loss_fn(real_paths_val, generated_paths_val)
                    
                    val_loss += v_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch:04d}/{num_epochs} | Train MMD: {avg_train_loss:.4f} | Val MMD: {avg_val_loss:.4f}")

            # Plotting toutes les 20 epochs
            fig_img = None
            if epoch % 20 == 0 and use_wandb:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                # On prend un sample du dernier batch de validation
                real_sample = real_paths_val[0].cpu().numpy()
                gen_sample = generated_paths_val[0].cpu().numpy()
                
                axes[0].plot(real_sample[:, 0], label='Real Asset A')
                axes[0].plot(real_sample[:, 1], label='Real Asset B')
                axes[0].set_title('Validation: Real Path (CumSum Log-Ret)')
                axes[0].legend()
                
                axes[1].plot(gen_sample[:, 0], label='Synth Asset A', linestyle='--')
                axes[1].plot(gen_sample[:, 1], label='Synth Asset B', linestyle='--')
                axes[1].set_title('Validation: SDE Generated Path')
                axes[1].legend()
                
                plt.tight_layout()
                fig_img = wandb.Image(fig)
                plt.close(fig)

            if use_wandb:
                metrics_dict = {
                    "train/mmd_loss": avg_train_loss,
                    "val/mmd_loss": avg_val_loss,
                    "epoch": epoch,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }
                if fig_img is not None:
                    metrics_dict["val/plot"] = fig_img
                wandb.log(metrics_dict)
                
            # Early Stopping Logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Sauvegarde du meilleur modèle (Validation)
                os.makedirs("data/models", exist_ok=True)
                torch.save(generator_sde.state_dict(), "data/models/best_val_sde.pt")
            else:
                epochs_no_improve += 10
                
            if epochs_no_improve >= early_stop_patience:
                print(f"🛑 Early stopping déclenché à l'epoch {epoch} (Patience={early_stop_patience} epochs sans amélioration).")
                # Restauration des meilleurs poids
                if os.path.exists("data/models/best_val_sde.pt"):
                    generator_sde.load_state_dict(torch.load("data/models/best_val_sde.pt"))
                break

    return generator_sde
