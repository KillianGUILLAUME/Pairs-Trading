import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from matplotlib.colors import ListedColormap

from features import extract_hmm_features

def align_states_by_volatility(hmm_model, X, n_components):
    """
    Assure que:
    0 = Calme (Faible Variabilité)
    1 = Tendance (Variabilité Moyenne)
    2 = Krach/Extrême (Haute Variabilité)
    """
    hidden_states = hmm_model.predict(X)
    state_vols = []
    
    for i in range(n_components):
        mask = (hidden_states == i)
        if np.sum(mask) > 0:
            # On calcule la variance des signaux features assignés à cet état
            state_vols.append(np.var(X[mask, :]))
        else:
            state_vols.append(0.0)
            
    # Tri indirect pour réassigner
    sorted_order = np.argsort(state_vols)
    idx_map = {old: new for new, old in enumerate(sorted_order)}
    
    # On clone le modèle pour réordonner ses poids sans perturber hmmlearn
    from copy import deepcopy
    new_model = deepcopy(hmm_model)
    
    new_model.means_ = hmm_model.means_[sorted_order]
    new_model.covars_ = hmm_model.covars_[sorted_order]
    if hasattr(hmm_model, "transmat_"):
        new_model.transmat_ = hmm_model.transmat_[sorted_order][:, sorted_order]
    if hasattr(hmm_model, "startprob_"):
        new_model.startprob_ = hmm_model.startprob_[sorted_order]
        
    return new_model

def train_and_plot_hmm(parquet_path: str, model_save_pwd: str, n_components: int = 3):
    print(f"📊 Chargement de l'historique BTC : {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    log_prices = np.log(df['close'].values)
    
    print("🧠 Calcul des Features de Régime...")
    X, valid_indices = extract_hmm_features(log_prices)
    
    print(f"⚙️ Entraînement du Modèle GaussianHMM ({n_components} États Cachés) via Expectation-Maximization...")
    model = GaussianHMM(
        n_components=n_components, 
        covariance_type="full", 
        n_iter=100, 
        random_state=42,
        tol=1e-4
    )
    
    model.fit(X)
    print(f"✅ Modèle convergé : {model.monitor_.converged}")
    
    # Réarrangement sémantique des états selon la volatilité
    model = align_states_by_volatility(model, X, n_components)
    
    hidden_states = model.predict(X)
    
    os.makedirs(model_save_pwd, exist_ok=True)
    save_path = os.path.join(model_save_pwd, "btc_regime_hmm.pkl")
    joblib.dump(model, save_path)
    print(f"💾 Modèle HMM sauvegardé dans {save_path}")
    
    # Visualization
    print("📈 Génération du Trace Plot...")
    valid_prices = df['close'].values[valid_indices]
    valid_dates = df['timestamp'].values[valid_indices]
    
    if isinstance(valid_dates[0], str) or isinstance(valid_dates[0], np.str_):
        valid_dates = pd.to_datetime(valid_dates)
    else:
        valid_dates = pd.to_datetime(valid_dates, unit='ms')

    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Color Maps pour les régimes
    cmap = ListedColormap(['#2ecc71', '#f1c40f', '#e74c3c']) # Vert (0), Jaune (1), Rouge (2)
    
    # Graphique en scatter avec changement de couleur continu
    scatter = ax.scatter(valid_dates, valid_prices, c=hidden_states, cmap=cmap, s=2, alpha=0.8)
    
    # Légendes et décorations
    legend_elements = scatter.legend_elements()[0]
    ax.legend(handles=legend_elements, labels=['Régime 0: Calme', 'Régime 1: Tendance/Normal', 'Régime 2: Krach/Haute Vol'])
    
    ax.set_title("Trading Regimes HMM : Bitcoin Crypto-Market")
    ax.set_yscale('log')
    ax.set_ylabel("BTC Price (Log Scale)")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(model_save_pwd, "hmm_regimes_plot.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"🖼️ Plot sauvegardé dans {plot_path}")
    plt.close()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    btc_path = os.path.join(project_root, "data", "storage", "parquet", "1h", "BTC_USDT.parquet")
    
    save_pwd = os.path.join(project_root, "data", "models", "hmm")
    
    if not os.path.exists(btc_path):
        print(f"❌ Bitcoin Parquet non trouvé {btc_path}.")
    else:
        train_and_plot_hmm(btc_path, save_pwd, n_components=3)
