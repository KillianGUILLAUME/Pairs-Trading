import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CryptoPairsDataset(Dataset):
    """
    Dataset PyTorch pour le Pairs Trading.
    Transforme les prix bruts en log-rendements standardisés et 
    génère des fenêtres glissantes (past_path, target).
    """
    def __init__(self, price_a: np.ndarray, price_b: np.ndarray, window_size: int = 60):
        # 1. Calcul des log-rendements
        log_ret_a = np.diff(np.log(price_a))
        log_ret_b = np.diff(np.log(price_b))
        
        # 2. Concaténation en un tableau [N, 2]
        self.raw_returns = np.stack([log_ret_a, log_ret_b], axis=1)
        
        # 3. Standardisation (Crucial pour le Flow Matching)
        self.mean = np.mean(self.raw_returns, axis=0)
        self.std = np.std(self.raw_returns, axis=0)
        
        # On évite la division par zéro avec 1e-8
        self.scaled_returns = (self.raw_returns - self.mean) / (self.std + 1e-8)
        
        # Conversion en Tenseurs PyTorch
        self.scaled_returns = torch.tensor(self.scaled_returns, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        # Si on a 1000 points et window=60, on peut faire 940 fenêtres
        return len(self.scaled_returns) - self.window_size

    def __getitem__(self, idx):
        """
        Retourne :
        - past_path : Le passé de taille [window_size, 2]
        - target (x_1) : La valeur EXACTEMENT suivante de taille [2]
        """
        past_path = self.scaled_returns[idx : idx + self.window_size]
        target = self.scaled_returns[idx + self.window_size]
        
        return past_path, target
    
    def inverse_transform(self, scaled_tensor: torch.Tensor) -> np.ndarray:
        """
        Fonction utilitaire pour remettre les rendements générés par l'IA 
        à leur vraie échelle crypto (pour le backtest).
        """
        numpy_array = scaled_tensor.cpu().numpy()
        return (numpy_array * self.std) + self.mean