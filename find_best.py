import os
import pandas as pd
import sys
sys.path.append('research/pairs')
from formation import PairsFormation

data_dir = "data/storage/parquet/1h"
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
print(f"Trouvé {len(files)} fichiers dans {data_dir}")

series = []
for f in files:
    df = pd.read_parquet(os.path.join(data_dir, f))
    df = df.set_index('timestamp')[['close']].rename(columns={'close': f.replace('.parquet', '')})
    series.append(df)

# On merge tout via inner join ou outer join ? Outer join et fill forward est mieux pr eviter de tout perdre
all_prices = pd.concat(series, axis=1, join='inner').dropna()
print(f"Dataset shape après alignement: {all_prices.shape}")

pf = PairsFormation(n_clusters=3, min_correlation=0.5)
results = pf.fit(all_prices)

for i, r in enumerate(results[:5]):
    print(f"Top {i+1}: {r.symbol_a} - {r.symbol_b} | Score: {r.score:.2f} | EG p-val: {r.eg_pvalue:.4f} | Johansen: {r.jh_cointegrated} | HL: {r.half_life:.2f}h | Spread_std: {r.spread_std:.2f}")

