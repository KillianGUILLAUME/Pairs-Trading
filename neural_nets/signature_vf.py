import math
import torch
import torch.nn as nn
try:
    import signatory
    HAS_SIGNATORY = True
    if torch.cuda.is_available():
        print("⚡ Signatory GPU backend loaded successfully.")
    else:
        print("⚡ Signatory loaded (CPU mode).")
except ImportError:
    import iisignature
    HAS_SIGNATORY = False
    print("⚠️ Signatory not found. Falling back to iisignature (CPU only) for Mac compatibility.")

def compute_path_signature(path_tensor: torch.Tensor, depth: int) -> torch.Tensor:
    """
    Calcule la signature du chemin de manière agnostique à l'environnement.
    - AWS (Linux/CUDA) : Utilise 'signatory' directement sur le GPU.
    - Mac (Local) : Fait le pont vers 'iisignature' sur le CPU en toute sécurité.
    """
    if HAS_SIGNATORY:
        # Exécution native sur GPU, aucune copie mémoire
        return signatory.signature(path_tensor, depth)
    else:
        # Fallback Mac : On sécurise le transfert vers le CPU et Numpy
        device = path_tensor.device
        path_np = path_tensor.detach().cpu().numpy()
        sig_raw = iisignature.sig(path_np, depth)
        return torch.tensor(sig_raw, dtype=torch.float32, device=device)

def compute_sig_channels(dim: int, depth: int) -> int:
    return sum(dim ** i for i in range(1, depth + 1))

class TimeEmbedding(nn.Module):
    """
        Sinusoidal Positional Encoding (Time Embedding).
        
        Convertit un scalaire temporel t (ex: 0.45) en un vecteur haute dimension.
        Nécessaire car les réseaux de neurones ont du mal à apprendre à partir 
        d'un seul nombre brut pour le temps.
        
        Référence : 
        Inspiré de "Attention Is All You Need" (Vaswani et al., 2017) pour les Transformers.
        
        Mathématiques :
        Le papier original définit les fréquences comme :
        ω_i = 1 / 10000^(i / (half_dim - 1))
        
        Pour des raisons de stabilité numérique et de rapidité sur GPU, on calcule 
        cela via le domaine logarithmique :
        exp(-i * (ln(10000) / (half_dim - 1)))
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class Seq2SeqVelocityField(nn.Module):
    """
    Velocity field Seq2Seq pour la Data Augmentation (Flow Matching complet).
    
    Transforme une séquence entière de bruit (ou un processus OU) de 
    taille (B, L, 2) vers une séquence réelle de même taille.
    
    Architecture :
        - 1D CNN : Pour extraire les features locales (micro-structure temporelle).
        - Path Signature : Pour capturer la géométrie globale de la séquence.
        - Time Embedding : Pour conditionner sur le temps de diffusion (Flow-time τ).
    """
    def __init__(self, data_dim=2, time_dim=64, sig_depth=4, hidden_dim=512):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = TimeEmbedding(time_dim)
        self.sig_depth = sig_depth
        
        # Dimension de la signature globale (data_dim + 1 pour l'axe temporel monotone)
        self.sig_channels = compute_sig_channels(data_dim + 1, sig_depth)
        
        # Le Feature Extractor Local (CNN 1D)
        # Input : (Batch, Channels=2, Length)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(data_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Le réseau de prédiction final de la vélocité
        # Input : CNN Local (hidden/2) + Global Sig (sig_channels) + Time (time_dim)
        total_features = (hidden_dim // 2) + self.sig_channels + time_dim
        
        self.velocity_net = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim) # Output: Vélocité par pas de temps
        )

    def forward(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        precomputed_sig: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, L, data_dim) - La séquence courante sur le chemin de diffusion
            t: (B, 1) - Temps de diffusion τ ∈ [0, 1]
        Returns:
            v: (B, L, data_dim) - Le champ de vélocité prédit pour CHAQUE point
        """
        device = x_t.device
        B, L, C = x_t.shape
        
        # 1. Feature Extraction Locale (CNN)
        # Permutation pour Conv1d : (B, L, C) -> (B, C, L)
        x_cnn = x_t.permute(0, 2, 1)
        local_features = self.local_cnn(x_cnn) # Shape: (B, hidden/2, L)
        local_features = local_features.permute(0, 2, 1) # Retour à (B, L, hidden/2)
        
        # 2. Contexte Global (Path Signature de x_t)
        if precomputed_sig is not None:
            sig = precomputed_sig
        else:
            time_tensor = torch.linspace(0, 1, L, device=device).unsqueeze(0).unsqueeze(-1)
            time_tensor = time_tensor.expand(B, -1, -1)
            path_augmented = torch.cat([time_tensor, x_t], dim=-1)
            
            # Calcul via iisignature (Nécessite CPU, c'est le goulot d'étranglement de l'inférence)
            sig = compute_path_signature(path_augmented, self.sig_depth)
            
        # Broadcast de la signature pour l'attacher à chaque point temporel
        sig_broadcast = sig.unsqueeze(1).expand(-1, L, -1) # Shape: (B, L, sig_channels)
        
        # 3. Time Embedding (Flow-time)
        t_flat = t.view(B, 1)
        t_emb = self.time_embed(t_flat) # Shape: (B, time_dim)
        t_emb_broadcast = t_emb.unsqueeze(1).expand(-1, L, -1) # Shape: (B, L, time_dim)
        
        # 4. Fusion et Prédiction
        h = torch.cat([local_features, sig_broadcast, t_emb_broadcast], dim=-1)
        v_pred = self.velocity_net(h) # Shape: (B, L, 2)
        
        return v_pred

    @torch.no_grad()
    def generate_synthetic_paths(
        self,
        interpolant,
        n_paths: int = 1000,
        seq_len: int = 60,
        n_ode_steps: int = 50,
        solver: str = "euler", # "euler" ou "rk4"
    ) -> torch.Tensor:
        """
        Génère n_paths trajectoires parallèles simultanément (Data Factory).
        
        Args:
            interpolant: L'OUPriorInterpolant pour générer le X_0 de base.
            n_paths: Nombre d'univers parallèles à générer.
            seq_len: Longueur de chaque trajectoire.
            n_ode_steps: Résolution de l'intégration numérique.
            
        Returns:
            generated: (n_paths, seq_len, data_dim)
        """
        device = next(self.parameters()).device
        
        print(f"🌌 Initialisation de {n_paths} trajectoires OU de {seq_len} heures...")
        target_shape = (n_paths, seq_len, self.data_dim)
        x_t = interpolant.sample_x0(target_shape).to(device)
        
        dt = 1.0 / n_ode_steps
        
        # ⚡ OPTIMISATION 1 : Pré-allocation des tenseurs de temps
        t_tensor = torch.empty((n_paths, 1), device=device)
        if solver == "rk4":
            t_mid = torch.empty((n_paths, 1), device=device)
            t_end = torch.empty((n_paths, 1), device=device)
            
        print(f"⚡ Intégration Flow Matching ({solver}, {n_ode_steps} pas)...")
        for i in range(n_ode_steps):
            t_val = i * dt
            
            # Mise à jour in-place (0 appel à l'allocateur mémoire du GPU)
            t_tensor.fill_(t_val)
            
            if solver == "euler":
                v = self.forward(x_t, t_tensor)
                v = torch.clamp(v, min=-0.5, max=0.5)
                # ⚡ OPTIMISATION 2 : Opération in-place (modifie x_t directement)
                x_t.add_(v, alpha=dt) 
                
            elif solver == "rk4":
                t_mid.fill_(t_val + 0.5 * dt)
                t_end.fill_(t_val + dt)
                
                k1 = self.forward(x_t, t_tensor)
                
                # On évite de multiplier k par dt plusieurs fois
                half_dt = 0.5 * dt
                k2 = self.forward(x_t + k1 * half_dt, t_mid)
                k3 = self.forward(x_t + k2 * half_dt, t_mid)
                k4 = self.forward(x_t + k3 * dt, t_end)
                
                # Fusion des k et mise à jour in-place
                x_t.add_(k1 + 2.0 * k2 + 2.0 * k3 + k4, alpha=dt / 6.0)
                
        return x_t