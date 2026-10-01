import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torchsde

class GeneratorSDE(nn.Module):
    """
    Neural SDE pour la génération de trajectoires de paires de trading.
    Utilise une structure Drift-Diffusion (formulation Ito).
    """
    def __init__(self, data_dim=2, hidden_dim=128, target_vol=0.1, min_vol=1e-4, condition_dim=0):
        super().__init__()
        self.data_dim = data_dim
        self.noise_type = "general" # Permet une matrice de diffusion complète (L)
        self.sde_type = "ito"
        self.min_vol = min_vol

        # Concatenation [y_t, spread_deviation, condition_aux]
        # spread_deviation = (yA - yB) - (yA0 - yB0)
        input_dim = data_dim + 1 + condition_dim 

        # Le réseau Drift (Tendance locale)
        self.drift_net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim) 
        )
        
        # Le réseau Diffusion (Volatilité instantanée locale)
        # Output for Cholesky: [L11, L21, L22] | dim = 3 
        cholesky_dim = (data_dim * (data_dim + 1)) // 2
        self.diffusion_net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.SiLU(),
            nn.Linear(hidden_dim, cholesky_dim), 
        )
        
        # Ajustement des poids pour commencer de manière stable
        for m in self.drift_net:
            if isinstance(m, nn.Linear):
                w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
                w.data.normal_(mean=0.0, std=1e-3)
                m.bias.data.fill_(0.)
        
        # On force la dernière couche à ZERO (zéro drift absolu au départ)
        self.drift_net[-1].weight.data.zero_()
        self.drift_net[-1].bias.data.zero_()

        for m in self.diffusion_net:
            if isinstance(m, nn.Linear):
                w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
                w.data.normal_(mean=0.0, std=1e-3)
                m.bias.data.zero_()

        # Dernières couches : on écrase pour la stabilité pure
        self.diffusion_net[-1].weight.data.zero_()
        self.diffusion_net[-1].bias.data.fill_(0.0)
        # FIX CHIRURGICAL : softplus(-3) \approx 0.05
        # On ancre la diffusion à 5% du target_vol (quand G est multiplié par target_vol plus tard)
        # Sauf qu'ici le target_vol n'est pas multiplié dans G, il servait au init_bias.
        # On utilise donc le -3.0 comme ancre absolue.
        self.diffusion_net[-1].bias.data[0] = -3.0 # L11
        self.diffusion_net[-1].bias.data[2] = -3.0 # L22

        # Variables de session
        self.current_y0 = None
        self.current_conditions = None
        self.cached_initial_spread = None

    def build_spread_and_condition(self, y):
        """Construit spread pas à pas"""
        current_spread = y[:, 0:1] - y[:, 1:2]
        spread_deviation = current_spread - self.cached_initial_spread 
        return torch.cat([y, spread_deviation, self.current_conditions], dim=-1)

    def f(self, t, y):
        """Fonction de Drift"""
        y_ext = self.build_spread_and_condition(y)
        raw_drift = self.drift_net(y_ext)
        path_norm = torch.norm(y, dim=-1, keepdim=True) 
        adaptive_bound = torch.sigmoid(-torch.log(path_norm.detach() / 10.0 + 1e-6))
        return raw_drift * adaptive_bound

    def g(self, t, y):
        """Fonction de Diffusion"""
        y_ext = self.build_spread_and_condition(y)
        out = self.diffusion_net(y_ext)
        L11 = torch.nn.functional.softplus(out[:, 0])
        L22 = torch.nn.functional.softplus(out[:, 2])
        L21 = out[:, 1]
        
        L11 = torch.clamp(L11, min=self.min_vol, max=5.0)
        L22 = torch.clamp(L22, min=self.min_vol, max=5.0)
        L21 = torch.clamp(L21, min=-5.0, max=5.0)

        # Reconstitution de la matrice de Cholesky (Batch, 2, 2)
        batch_size = y.shape[0]
        G = torch.zeros(batch_size, 2, 2, device=y.device, dtype=y.dtype)
        G[:, 0, 0] = L11
        G[:, 1, 0] = L21
        G[:, 1, 1] = L22
        return G

    def forward(self, y0, ts, conditions):
        """Intégration de la SDE"""
        self.current_y0 = y0
        self.current_conditions = conditions
        self.cached_initial_spread = y0[:, 0:1] - y0[:, 1:2]
        
        # Intégration
        dt = 0.25
        paths = torchsde.sdeint(self, y0, ts, dt=dt, method="euler")
        return paths.transpose(0, 1) # (Batch, Seq_Len, Dim)

    def sample(self, y0, seq_len, conditions):
        """Wrapper simplifié pour la génération"""
        ts = torch.linspace(0.0, float(seq_len - 1), seq_len, device=y0.device)
        # Ensure float32 for SDE solver
        y0 = y0.float()
        conditions = conditions.float()
        return self.forward(y0, ts, conditions)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device="cpu"):
        """Charge un modèle et ses métadonnées de scaling avec détection dynamique"""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})

        # Nettoyage des clés (si wrapé dans _orig_mod via torch.compile)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v

        # Inférence dynamique des dimensions du réseau
        # drift_net.0.weight shape: [hidden_dim, input_dim]
        # input_dim = data_dim + 1 + condition_dim
        first_layer_name = "drift_net.0.weight_orig" if "drift_net.0.weight_orig" in new_state_dict else "drift_net.0.weight"
        
        hidden_dim, input_dim = new_state_dict[first_layer_name].shape
        data_dim = model_cfg.get("data_dim", 2)
        condition_dim = input_dim - data_dim - 1

        model = cls(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim
        )
        
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model, ckpt
