# NEW: data/connectors/base_connector.py
from abc import ABC, abstractmethod
import pandas as pd
import os
from typing import Union
from datetime import datetime

class BaseConnector(ABC):
    """Unified interface — make real/synthetic data interchangeable."""
    
    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, 
              start: Union[str, datetime], end: Union[str, datetime]) -> pd.DataFrame:
        """
        Retourne l'historique entre deux dates.
        DOIT retourner au minimum les colonnes : 
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        pass
    
    @abstractmethod
    def fetch_latest(self, symbol: str, timeframe: str, 
                     n_bars: int) -> pd.DataFrame:
        """
        Récupère les N dernières barres.
        Crucial pour le Live Trading ou pour "amorcer" le modèle SDE génératif.
        """
        pass