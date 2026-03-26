import yaml
from loguru import logger
from tqdm import tqdm

from data.connectors.binance_connector import BinanceConnector
from data.storage.parquet_storage import ParquetStorage


class DataPipeline:
    """
    Orchestre : sélection univers → fetch → stockage.
    Point d'entrée principal pour la couche data.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.connector = BinanceConnector()
        self.storage   = ParquetStorage(
            self.config["data"]["storage"]["base_path"]
        )
        logger.info("DataPipeline initialisé")

    def run(
        self,
        symbols: list[str] | None = None,
        force_refresh: bool = False,
    ) -> None:
        """
        Lance le pipeline complet.
        - symbols : liste custom, sinon on prend l'univers du config
        - force_refresh : re-télécharge même si le fichier existe
        """
        cfg = self.config["data"]

        # 1. Univers
        if symbols is None:
            symbols = self.connector.get_universe(
                quote_currency=cfg["universe"]["quote_currency"],
                min_volume_usdt=cfg["universe"]["min_volume_usdt"],
                top_n=cfg["universe"]["top_n"],
            )

        timeframes = cfg["timeframes"]
        start_date = cfg["history"]["start_date"]
        end_date   = cfg["history"]["end_date"]

        logger.info(f"Pipeline : {len(symbols)} symboles × {len(timeframes)} timeframes")

        # 2. Fetch & Store
        for tf in timeframes:
            logger.info(f"--- Timeframe : {tf} ---")
            for symbol in tqdm(symbols, desc=f"Fetching {tf}"):

                if not force_refresh and self.storage.exists(symbol, tf):
                    logger.debug(f"Skip {symbol} {tf} (déjà en cache)")
                    continue

                df = self.connector.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                )
                self.storage.save(df, symbol, tf)

        logger.info("Pipeline terminé ✓")

    def load_universe(self, timeframe: str) -> dict:
        """
        Charge tous les symboles disponibles pour un timeframe.
        Retourne un dict {symbol: DataFrame}
        """
        symbols = self.storage.list_available(timeframe)
        data = {}
        for symbol in symbols:
            df = self.storage.load(symbol, timeframe)
            if not df.is_empty():
                data[symbol] = df
        logger.info(f"Chargé {len(data)} symboles pour {timeframe}")
        return data


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
