import asyncio
import ccxt.async_support as ccxt
from loguru import logger
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class ExchangeBridge:
    """
    Interface Asynchrone avec l'Exchange (Binance via CCXT).
    Gère les flux de données et l'exécution des ordres.
    """
    def __init__(self, api_key: str = "", secret: str = "", exchange_id: str = "binance"):
        self.exchange_class = getattr(ccxt, exchange_id)
        self.exchange = self.exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        self.is_connected = False

    async def connect(self):
        try:
            await self.exchange.load_markets()
            self.is_connected = True
            logger.info(f"🚀 Connecté à {self.exchange.id}")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion : {e}")
            raise

    async def fetch_ohlcv_dataframe(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """
        Récupère les bougies OHLCV et les convertit en DataFrame.
        """
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df

    async def get_spread_prices(self, symbol_a: str, symbol_b: str) -> Dict[str, float]:
        """
        Récupère les prix mid/close actuels pour une paire d'actifs.
        """
        tickers = await self.exchange.fetch_tickers([symbol_a, symbol_b])
        return {
            'a': tickers[symbol_a]['last'],
            'b': tickers[symbol_b]['last']
        }

    async def get_balance(self, currency: str = 'USDT') -> float:
        """
        Récupère le solde disponible pour une devise.
        """
        balance = await self.exchange.fetch_balance()
        return balance['free'].get(currency, 0.0)

    async def execute_order(self, symbol: str, side: str, amount: float, order_type: str = 'market'):
        """
        Place un ordre sur le marché.
        """
        try:
            logger.info(f"🔔 PLACEMENT ORDRE : {side} {amount} {symbol}")
            order = await self.exchange.create_order(symbol, order_type, side, amount)
            return order
        except Exception as e:
            logger.error(f"❌ Échec de l'ordre sur {symbol} : {e}")
            return None

    async def close(self):
        await self.exchange.close()
