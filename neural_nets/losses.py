import torch
import torch.nn.functional as F

# ============================================
# Loss pour entrainer le velocity field (Teacher causal)
# ============================================

class ConditionalFlowMatchingLoss:
    """
    Loss pour entraîner le velocity field conditionné par le passé.
    """
    def __init__(self, velocity_field):
        self.velocity_field = velocity_field
    
    def __call__(self, x_0, x_1, past_path): 
        """
        Args:
            x_0: (B, data_dim) - Bruit source p_0
            x_1: (B, data_dim) - Vraie variation cible p_1
            past_path: (B, window_size, data_dim) - Trajectoire historique
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample temps uniformément
        t = torch.rand(batch_size, 1, device=device)
        
        # Interpolation conditionnelle (conditional flow)
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target velocity (formule analytique de dx_t/dt)
        v_target = x_1 - x_0
        
        # Prédiction du réseau CONDITIONNÉE par la signature du passé
        v_pred = self.velocity_field(x_t, t, past_path) 
        
        # Loss MSE
        loss = F.mse_loss(v_pred, v_target)
        
        return loss

# ============================================
# Bridge Matching Loss — DSBM Causal
# ============================================

class BridgeMatchingLoss:
    """
    Loss de Bridge Matching conditionnée par le passé.
    """
    def __init__(self, drift_network, sigma: float = 0.5):
        self.drift_network = drift_network
        self.sigma = sigma

    def __call__(self, x0: torch.Tensor, x1: torch.Tensor, past_path: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        device = x0.device

        # 1. Tire t ~ U[0, 1-ε]
        eps = 1e-3
        t = torch.rand(batch_size, 1, device=device) * (1.0 - eps)

        # 2. Tire un point x_t du Brownian Bridge conditionnel
        mean_t = (1.0 - t) * x0 + t * x1
        std_t = self.sigma * torch.sqrt(t * (1.0 - t) + 1e-8)
        z = torch.randn_like(x0)
        x_t = mean_t + std_t * z

        # 3. Dérive analytique cible
        denom = torch.clamp(1.0 - t, min=eps)
        u_target = (x1 - x_t) / denom

        # 4. Prédiction du réseau conditionnée
        u_pred = self.drift_network(x_t, t, past_path)

        # 5. Loss MSE
        loss = F.mse_loss(u_pred, u_target)

        return loss

# ============================================
# Fonction d'entraînement (La boucle principale)
# ============================================

def train_teacher(velocity_field, data_loader, num_epochs=1000, lr=1e-3, device='cuda'):
    """
    Entraîne le velocity field causal avec Conditional Flow Matching.
    """
    cfm_loss = ConditionalFlowMatchingLoss(velocity_field)
    optimizer = torch.optim.AdamW(velocity_field.parameters(), lr=lr, weight_decay=1e-4) 
    
    velocity_field.to(device)
    velocity_field.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for past_path, x_1 in data_loader:
            past_path = past_path.to(device)
            x_1 = x_1.to(device)
            
            # x_0 ~ N(0, I) (Bruit source)
            x_0 = torch.randn_like(x_1)
            
            optimizer.zero_grad()
            
            # Calcul de la perte avec le conditionnement causal
            loss = cfm_loss(x_0, x_1, past_path)
            
            loss.backward()
            
            # Gradient clipping (indispensable avec les signatures qui peuvent exploser)
            torch.nn.utils.clip_grad_norm_(velocity_field.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if epoch % 10 == 0: # Affichage plus fréquent
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch:04d}/{num_epochs}, CFM Loss: {avg_loss:.6f}")
    
    return velocity_field
