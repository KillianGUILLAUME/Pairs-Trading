import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass 
class MLConfig:
    model_type: str   = "lstm"    # "xgb" | "lstm" | "transformer"
    input_dim:  int   = 32        # signature_dim + kalman_features
    hidden_dim: int   = 64
    n_layers:   int   = 2
    dropout:    float = 0.2
    seq_len:    int   = 48        # 48h lookback


class LSTMPairModel(nn.Module):
    """
    LSTM pour prédire le label triple-barrier.
    Input  : (batch, seq_len, input_dim) — signature + z-score + OBI
    Output : (batch, 3) — proba [stop_loss, time_stop, profit]
    """

    def __init__(self, cfg: MLConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = cfg.input_dim,
            hidden_size  = cfg.hidden_dim,
            num_layers   = cfg.n_layers,
            dropout      = cfg.dropout,
            batch_first  = True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # dernier timestep
