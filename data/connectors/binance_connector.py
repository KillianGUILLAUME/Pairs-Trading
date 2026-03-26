import ccxt
import polars as pl
import time
from loguru import logger
from datetime import datetime
from typing import Optional


class BinanceConnector:
    """
    Connector Binance via CCXT.
    Gère la récupération des OHLCV et des métadonnées de marché.
    """

    TIMEFRAME_MS = {
        "1m":  60_000,
        "5m":  300_000,
        "15m": 900_000,
        "1h":  3_600_000,
        "4h":  14_400_000,
        "1d":  86_400_000,
    }

    def __init__(self, rate_limit: bool = True):
        self.exchange = ccxt.binance({
            "enableRateLimit": rate_limit,
            "options": {"defaultType": "spot"},  # spot ou future
        })
        logger.info("BinanceConnector initialisé")

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(
        self,
        quote_currency: str = "USDT",
        min_volume_usdt: float = 10_000_000,
        top_n: int = 50,
    ) -> list[str]:
        """
        Retourne les top_n symboles les plus liquides
        filtrés par quote_currency et volume minimum.
        """
        logger.info("Récupération de l'univers Binance...")

        tickers = self.exchange.fetch_tickers()

        records = []
        for symbol, data in tickers.items():
            if not symbol.endswith(f"/{quote_currency}"):
                continue
            volume = data.get("quoteVolume") or 0
            if volume < min_volume_usdt:
                continue
            records.append({
                "symbol": symbol,
                "volume_24h_usdt": volume,
                "last_price": data.get("last") or 0,
            })

        df = (
            pl.DataFrame(records)
            .sort("volume_24h_usdt", descending=True)
            .head(top_n)
        )

        symbols = df["symbol"].to_list()
        logger.info(f"{len(symbols)} symboles sélectionnés")
        return symbols

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None,
        pause_sec: float = 0.2,
    ) -> pl.DataFrame:
        """
        Fetch complet des OHLCV pour un symbole donné.
        Gère la pagination automatiquement (Binance limite à 1000 candles/appel).
        """
        if timeframe not in self.TIMEFRAME_MS:
            raise ValueError(f"Timeframe {timeframe} non supporté. Choix : {list(self.TIMEFRAME_MS)}")

        since_ms = self._date_to_ms(start_date)
        end_ms   = self._date_to_ms(end_date) if end_date else int(datetime.now().timestamp() * 1000)
        tf_ms    = self.TIMEFRAME_MS[timeframe]

        all_candles = []

        logger.info(f"Fetching {symbol} | {timeframe} | {start_date} → {end_date}")

        while since_ms < end_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=1000,
                )
            except ccxt.NetworkError as e:
                logger.warning(f"Network error sur {symbol} : {e} — retry dans 5s")
                time.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error sur {symbol} : {e}")
                break

            if not candles:
                break

            all_candles.extend(candles)

            last_ts = candles[-1][0]
            since_ms = last_ts + tf_ms  # avance d'une bougie

            # on a dépassé la fin
            if last_ts >= end_ms:
                break

            time.sleep(pause_sec)

        if not all_candles:
            logger.warning(f"Aucune donnée récupérée pour {symbol}")
            return pl.DataFrame()

        df = self._candles_to_dataframe(all_candles, symbol)

        # Filtre propre sur la plage de dates
        df = df.filter(
            (pl.col("timestamp") >= since_ms - (end_ms - self._date_to_ms(start_date)))
            & (pl.col("timestamp") <= end_ms)
        )

        logger.info(f"{symbol} → {len(df)} candles récupérées")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candles_to_dataframe(self, candles: list, symbol: str) -> pl.DataFrame:
        """Convertit la liste brute CCXT en DataFrame Polars typé."""
        return pl.DataFrame(
            {
                "timestamp": [c[0] for c in candles],
                "open":      [c[1] for c in candles],
                "high":      [c[2] for c in candles],
                "low":       [c[3] for c in candles],
                "close":     [c[4] for c in candles],
                "volume":    [c[5] for c in candles],
                "symbol":    [symbol] * len(candles),
            },
            schema={
                "timestamp": pl.Int64,
                "open":      pl.Float64,
                "high":      pl.Float64,
                "low":       pl.Float64,
                "close":     pl.Float64,
                "volume":    pl.Float64,
                "symbol":    pl.Utf8,
            }
        ).with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
        ).unique("timestamp").sort("timestamp")

    @staticmethod
    def _date_to_ms(date_str: str) -> int:
        """Convertit 'YYYY-MM-DD' en timestamp milliseconds."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)
