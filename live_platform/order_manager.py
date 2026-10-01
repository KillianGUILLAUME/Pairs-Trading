from loguru import logger
from live_platform.exchange_bridge import ExchangeBridge
import asyncio

class OrderManager:
    """
    Gestionnaire d'ordres institutionnel.
    Gère l'exécution atomique des paires (Double-Leg execution).
    """
    def __init__(self, bridge: ExchangeBridge):
        self.bridge = bridge
        self.active_positions = {} # {pair_name: {'side': 'long/short', 'size_a', 'size_b'}}

    async def open_pair_trade(self, symbol_a: str, symbol_b: str, side: str, amount_a: float, amount_b: float):
        """
        Ouvre une position de pairs trading (Long A / Short B ou vice versa).
        """
        pair_name = f"{symbol_a}_{symbol_b}"
        logger.info(f"🚀 OUVERTURE DE POSITION : {pair_name} | Sens: {side}")
        
        side_a = 'buy' if side == 'long' else 'sell'
        side_b = 'sell' if side == 'long' else 'buy'
        
        # Exécution en parallèle pour minimiser le slippage entre les legs
        res_a, res_b = await asyncio.gather(
            self.bridge.execute_order(symbol_a, side_a, amount_a),
            self.bridge.execute_order(symbol_b, side_b, amount_b)
        )
        
        if res_a and res_b:
            logger.info(f"✅ Position {pair_name} ouverte avec succès.")
            self.active_positions[pair_name] = {
                'side': side,
                'sym_a': symbol_a,
                'sym_b': symbol_b,
                'amount_a': amount_a,
                'amount_b': amount_b
            }
            return True
        else:
            # Gestion d'erreur critique : Leg orpheline
            logger.error(f"⚠️ ÉCHEC D'OUVERTURE ATOMIQUE SUR {pair_name}. Nettoyage nécessaire !")
            if res_a: await self.bridge.execute_order(symbol_a, 'sell' if side_a == 'buy' else 'buy', amount_a)
            if res_b: await self.bridge.execute_order(symbol_b, 'sell' if side_b == 'buy' else 'buy', amount_b)
            return False

    async def close_pair_trade(self, pair_name: str):
        """
        Ferme une position active.
        """
        if pair_name not in self.active_positions:
            return False
            
        pos = self.active_positions[pair_name]
        logger.info(f"🏁 FERMETURE DE POSITION : {pair_name}")
        
        side_a = 'sell' if pos['side'] == 'long' else 'buy'
        side_b = 'buy' if pos['side'] == 'long' else 'sell'
        
        res_a, res_b = await asyncio.gather(
            self.bridge.execute_order(pos['sym_a'], side_a, pos['amount_a']),
            self.bridge.execute_order(pos['sym_b'], side_b, pos['amount_b'])
        )
        
        if res_a and res_b:
            del self.active_positions[pair_name]
            logger.info(f"✅ Position {pair_name} fermée.")
            return True
        return False
