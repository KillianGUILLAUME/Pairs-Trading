import torch
import torch.nn.functional as F
import signatory
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

class SignatureMMDLoss(torch.nn.Module):
    """
    Maximum Mean Discrepancy (MMD) sur les Path Signatures.
    
    Calcule la distance L2 entre l'Espérance de la signature (Moment Matching).
    Comprend désormais la projection Unitaire Relative (Relative Error) pour stabiliser les gradients.
    Utilise `signatory.Signature` (Module) plutôt que la fonction stateless :
    depth et options sont stockés une seule fois, le module est éligible à .to(device).
    """
    def __init__(self, depth: int = 4):
        super().__init__()
        self.depth = depth
        # Module persistant : instancié une fois, réutilisé à chaque forward
        self.sig_fn = signatory.Signature(depth=depth)

    def forward(self, real_paths: torch.Tensor, generated_paths: torch.Tensor) -> torch.Tensor:
        sig_real = self.sig_fn(real_paths)
        sig_gen  = self.sig_fn(generated_paths)
        
        mean_sig_real = sig_real.mean(dim=0)
        mean_sig_gen  = sig_gen.mean(dim=0)
        
        raw_dist  = torch.norm(mean_sig_real - mean_sig_gen, p=2)
        real_norm = torch.norm(mean_sig_real, p=2).detach().clamp(min=1e-6)
        
        return raw_dist / real_norm

    # Alias pour compatibilité avec l'ancien code non-Module
    def __call__(self, real_paths, generated_paths):
        return self.forward(real_paths, generated_paths)


class KernelSignatureMMDLoss(torch.nn.Module):
    """
    Multi-scale Kernel MMD sur Path Signatures (RBF kernel).
    Compare des distributions de chemins via leurs signatures tronquées.
    
    Optimisations :
    - signatory.Signature instancié une fois (nn.Module persistent)
    - sigmas, sig_mean, sig_std enregistrés comme buffers (suivent .to(device))
    - Kernel multi-échelle vectorisé (broadcast sur S échelles simultanément)
    """

    def __init__(
        self,
        depth: int = 4,
        sigmas: list = None,
        sig_scaler_mean: torch.Tensor = None,
        sig_scaler_std: torch.Tensor = None,
    ):
        super().__init__()
        self.depth = depth

        # Module persistent : suit .to(device) automatiquement
        self.sig_fn = signatory.Signature(depth=depth)

        # --- Buffers : vivent sur le bon device, inclus dans state_dict ---
        _sigmas = torch.tensor(
            sigmas if sigmas is not None else [0.1, 0.5, 1.0, 5.0, 10.0],
            dtype=torch.float32,
        )
        self.register_buffer("sigmas", _sigmas)

        # Scalers : toujours enregistrés comme buffers (même si None → placeholder)
        # On utilise un flag pour savoir si on doit normaliser
        self._use_scaler = sig_scaler_mean is not None and sig_scaler_std is not None

        if self._use_scaler:
            self.register_buffer("sig_mean", sig_scaler_mean.float())
            self.register_buffer("sig_std",  sig_scaler_std.float().clamp(min=1e-8))
        else:
            # Placeholders (permet un state_dict cohérent si tu save/load)
            self.register_buffer("sig_mean", torch.zeros(1))
            self.register_buffer("sig_std",  torch.ones(1))

    def _normalize(self, sig_raw: torch.Tensor) -> torch.Tensor:
        if self._use_scaler:
            return (sig_raw - self.sig_mean) / self.sig_std
        return sig_raw

    def forward(
        self,
        real_paths: torch.Tensor,
        generated_paths: torch.Tensor,
        precomputed_sig_real: torch.Tensor = None,
    ) -> torch.Tensor:

        # --- Signatures (avec cache possible sur le réel, qui ne change pas) ---
        if precomputed_sig_real is not None:
            sig_real = precomputed_sig_real
        else:
            sig_real = self._normalize(self.sig_fn(real_paths))

        sig_gen = self._normalize(self.sig_fn(generated_paths))

        # --- Distances L2² vectorisées (B x B) ---
        # Note : cdist(..., p=2).pow(2) fait une sqrt puis un carré → gaspillage.
        # On calcule directement ||x-y||² = ||x||² + ||y||² - 2 x·yᵀ
        dist_xx = self._pairwise_sq_dist(sig_real, sig_real)
        dist_yy = self._pairwise_sq_dist(sig_gen,  sig_gen)
        dist_xy = self._pairwise_sq_dist(sig_real, sig_gen)

        # --- Multi-scale RBF kernel (broadcast sur S échelles) ---
        gammas = (1.0 / (2.0 * self.sigmas.pow(2))).view(-1, 1, 1)  # (S,1,1)

        k_xx = 1.0 / (1.0 + gammas * dist_xx)
        k_yy = 1.0 / (1.0 + gammas * dist_yy)
        k_xy = 1.0 / (1.0 + gammas * dist_xy)

        mmd_per_scale = (
            k_xx.mean(dim=(1, 2))
            + k_yy.mean(dim=(1, 2))
            - 2.0 * k_xy.mean(dim=(1, 2))
        )
        mmd_sq = mmd_per_scale.sum()

        return torch.clamp(mmd_sq, min=0.0)

    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """||x_i - y_j||² vectorisé, sans passer par sqrt."""
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x_sq = (x * x).sum(dim=1, keepdim=True)           # (Bx, 1)
        y_sq = (y * y).sum(dim=1, keepdim=True).T         # (1, By)
        xy   = x @ y.T                                    # (Bx, By)
        return (x_sq + y_sq - 2.0 * xy).clamp(min=0.0)

# ============================================
# Fonction d'entraînement (SDE)
# ============================================

def _compute_extra_metrics(real_paths: torch.Tensor, generated_paths: torch.Tensor) -> dict:
    """Métriques diagnostiques pour wandb."""
    with torch.no_grad():
        real_spread = real_paths[:, :, 0] - real_paths[:, :, 1]
        gen_spread  = generated_paths[:, :, 0] - generated_paths[:, :, 1]

        real_vol   = real_spread.std(dim=1).mean().item()
        gen_vol    = gen_spread.std(dim=1).mean().item()
        real_drift = real_spread.diff(dim=1).mean().item()
        gen_drift  = gen_spread.diff(dim=1).mean().item()
        gen_norm   = generated_paths.norm(dim=-1).mean().item()

    return {
        "diagnostics/real_spread_vol": real_vol,
        "diagnostics/gen_spread_vol":  gen_vol,
        "diagnostics/vol_ratio":       gen_vol / (real_vol + 1e-8),
        "diagnostics/real_drift":      real_drift,
        "diagnostics/gen_drift":       gen_drift,
        "diagnostics/gen_path_norm":   gen_norm,
    }




def train_sde(
    generator_sde,
    train_loader,
    val_loader,
    sig_mean: torch.Tensor, 
    sig_std: torch.Tensor,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = 'cuda',
    sig_depth: int = 4,
    use_wandb: bool = False
):
    """
    Boucle d'entraînement SDE avec Kernel Signature MMD, Validation, Plots et Early Stopping.
    """
    import os
    import matplotlib.pyplot as plt
    
    generator_sde.to(device)
    
    loss_fn = KernelSignatureMMDLoss(
        depth=sig_depth,
        sigmas=[1.0, 5.0, 10.0, 20.0, 40.0],
        sig_scaler_mean=sig_mean,
        sig_scaler_std=sig_std,
    ).to(device)
        
    optimizer = torch.optim.AdamW(generator_sde.parameters(), lr=lr, weight_decay=weight_decay, fused=True)

    from torch.optim.swa_utils import AveragedModel

    ema_sde = AveragedModel(
        generator_sde,
        avg_fn=lambda avg, p, n: 0.999 * avg + 0.001 * p,
    )

    accum_iter = 4
    steps_per_epoch = (len(train_loader) + accum_iter - 1) // accum_iter
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-7
    )

    # Paramètres d'Early Stopping
    best_val_loss = float('inf')
    early_stop_patience = 70
    epochs_no_improve = 0

    ts_cache = None

    for epoch in range(num_epochs):
        generator_sde.train()
        epoch_loss, num_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)


        last_grad_norm = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                # non_blocking=True est crucial avec pin_memory=True :
                # ça permet au GPU de charger le prochain batch PENDANT qu'il calcule le précédent !
                real_paths = batch[0].to(device, non_blocking=True) # Shape (B, L, 2)
                conditions = batch[1].to(device, non_blocking=True)
                preputed = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            else:
                 real_paths = batch.to(device, non_blocking=True)
                 conditions = None
                 preputed = None
            
            B, seq_len, _ = real_paths.shape
            if ts_cache is None or ts_cache.shape[0] != seq_len:
                ts_cache = torch.linspace(0., float(seq_len - 1), seq_len, device=device)

            y0 = real_paths[:, 0, :]

            # Résolution Differentiable de la SDE
            generated_paths = generator_sde(y0, ts_cache, conditions)
            generated_paths = torch.clamp(generated_paths, min=-30.0, max=30.0)
            
            # Signature MMD bypass avec Accumulation
            loss = loss_fn(real_paths, generated_paths, preputed)
            scaled_loss = loss / accum_iter
                
            scaled_loss.backward()

            
            # Application du gradient tous les 'accum_iter' pas ou à la toute fin
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(generator_sde.parameters(), max_norm=1.0)
                total_norm = 0.0
                for p in generator_sde.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                last_grad_norm = total_norm ** 0.5

                optimizer.step()
                scheduler.step()
                ema_sde.update_parameters(generator_sde)
                optimizer.zero_grad(set_to_none=True)


            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # ==========================================
        # VALIDATION & EARLY STOPPING (Tous les 10 Epochs)
        # ==========================================
        if epoch % 1 == 0: #to have better metrics (time compute)
            ema_sde.module.eval()
            val_loss = 0.0
            val_batches = 0
            ts_val_cache = None
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        real_paths_val = batch[0].to(device, non_blocking=True)
                        conditions_val = batch[1].to(device, non_blocking=True)
                        preputed_val = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
                    else:
                        real_paths_val = batch.to(device, non_blocking=True)
                        conditions_val = None
                        preputed_val = None
                    b_size, seq_len_val, d_dim = real_paths_val.shape

                    if ts_val_cache is None or ts_val_cache.shape[0] != seq_len_val:
                        ts_val_cache = torch.linspace(0.0, float(seq_len_val - 1), seq_len_val, device=device)
                    
                    y0_val = real_paths_val[:, 0, :]
                    
                    generated_paths_val = ema_sde.module(y0_val, ts_val_cache, conditions_val)
                    v_loss = loss_fn(real_paths_val, generated_paths_val, preputed_val)
                    
                    val_loss += v_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)

            extra_metrics = _compute_extra_metrics(real_paths_val, generated_paths_val)
            
            print(f"Epoch {epoch:04d}/{num_epochs} | Train MMD: {avg_train_loss:.4f} | Val MMD: {avg_val_loss:.4f}")


            # Plotting toutes les 5 epochs
            fig_img = None
            if epoch % 5 == 0 and use_wandb:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                # On prend un sample du dernier batch de validation
                real_sample = real_paths_val[0].cpu().numpy()
                gen_sample = generated_paths_val[0].cpu().numpy()
                
                axes[0].plot(real_sample[:, 0], label='Real Asset A')
                axes[0].plot(real_sample[:, 1], label='Real Asset B')
                axes[0].set_title('Validation: Scaled Log-Prices (Real)')
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
                    # Losses principales
                    "train/mmd_loss": avg_train_loss,
                    "val/mmd_loss": avg_val_loss,
                    
                    # Overfitting gap (doit rester proche de 0)
                    "val/overfit_gap": avg_val_loss - avg_train_loss,
                    
                    # Optimiseur
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/grad_norm": last_grad_norm,          # calculé juste avant
                    
                    # Early stopping (utile pour voir quand tu es proche du stop)
                    "train/epochs_no_improve": epochs_no_improve,
                    
                    # Diagnostics SDE
                    **extra_metrics,            # le dict calculé par _compute_extra_metrics
                    
                    # Axe X
                    "epoch": epoch,
                }
                
                if fig_img is not None:
                    metrics_dict["val/generated_paths"] = fig_img
                
                wandb.log(metrics_dict)

                
            # Early Stopping Logic (Validation sur modèle EMA)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Sauvegarde du meilleur modèle (Validation)
                os.makedirs("data/models", exist_ok=True)
                torch.save(ema_sde.module.state_dict(), "data/models/best_val_sde.pt")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= early_stop_patience:
                print(f"🛑 Early stopping déclenché à l'epoch {epoch} (Patience={early_stop_patience} epochs sans amélioration).")
                # Restauration des meilleurs poids
                if os.path.exists("data/models/best_val_sde.pt"):
                    ema_sde.module.load_state_dict(
                        torch.load("data/models/best_val_sde.pt", weights_only=True)
                    )
                break

    return ema_sde.module
