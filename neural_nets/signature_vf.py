import torch
import torch.nn as nn
import signatory

class TimeEmbedding(nn.Module):
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

class CausalVelocityField(nn.Module):
    """
    Teacher Causal : v_θ(x_t, t | Signature(X_{t-w:t}))
    Apprend le champ de vitesse conditionné par la géométrie du passé.
    """
    def __init__(self, data_dim=2, time_dim=64, sig_depth=3, hidden_dim=256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        
        # Calcul de la dimension de la signature 
        # (data_dim + 1 car on ajoute le temps à la trajectoire pour le Lead-Lag)
        self.sig_channels = signatory.signature_channels(data_dim + 1, sig_depth)
        self.sig_depth = sig_depth
        
        # Le réseau prend maintenant : position (x) + temps (t) + signature du passé (sig)
        input_dim = data_dim + time_dim + self.sig_channels
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t, past_path):
        """
        x: (B, data_dim) - position courante à t
        t: (B, 1) - temps de diffusion
        past_path: (B, window_size, data_dim) - la trajectoire historique de la paire
        """
        device = x.device
        batch_size, window_size, _ = past_path.shape
        
        # 1. Augmentation temporelle du chemin (pour capturer la vélocité via la signature)
        time_tensor = torch.linspace(0, 1, window_size, device=device).unsqueeze(0).unsqueeze(-1)
        time_tensor = time_tensor.expand(batch_size, -1, -1)
        path_augmented = torch.cat([time_tensor, past_path], dim=-1)
        
        # 2. Calcul de la Signature
        sig = signatory.signature(path_augmented, self.sig_depth)
        
        # 3. Embeddings et Concaténation
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb, sig], dim=-1)
        
        # 4. Prédiction de la vélocité
        return self.net(h)