"""
Modelos de previsão usados no estudo. Todos recebem a série em log até a origem
(pd.Series mensal) e devolvem a previsão do log para h = 1..12 meses à frente.

Estatísticos: sazonal ingênuo, ETS/Holt-Winters (seleção por AICc), SARIMA.
IA (aprendizado de máquina): Random Forest, XGBoost, LightGBM e rede neural MLP,
treinados sobre a variação anual em log  d_t = log y_t - log y_{t-12}
com estratégia recursiva; a previsão do nível é  log y_{t+h} = log y_{t+h-12} + d_{t+h}.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

H = 12
LAGS = (1, 2, 3, 6, 12)
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- estatísticos
def prev_snaive(logy, h=H):
    return logy.iloc[-12:].values[:h].copy()


ETS_CONFIGS = [dict(trend=None, damped_trend=False),
               dict(trend="add", damped_trend=False),
               dict(trend="add", damped_trend=True)]


def prev_ets(logy, h=H, devolver_config=False):
    melhor, melhor_aicc, melhor_cfg = None, np.inf, None
    for cfg in ETS_CONFIGS:
        try:
            m = ETSModel(logy, error="add", seasonal="add", seasonal_periods=12, **cfg)
            r = m.fit(disp=False, maxiter=2000)
            if r.aicc < melhor_aicc:
                melhor, melhor_aicc, melhor_cfg = r, r.aicc, cfg
        except Exception:
            continue
    f = np.asarray(melhor.forecast(h))
    return (f, melhor_cfg) if devolver_config else f


def seleciona_sarima(logy):
    """Grade (p,1,q)(P,1,Q)12 com p,q<=2 e P,Q<=1, escolhida por AICc."""
    melhor, melhor_aicc = None, np.inf
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        r = SARIMAX(logy, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
                        if r.aicc < melhor_aicc:
                            melhor, melhor_aicc = ((p, 1, q), (P, 1, Q, 12)), r.aicc
                    except Exception:
                        continue
    return melhor


def prev_sarima(logy, ordem, h=H):
    r = SARIMAX(logy, order=ordem[0], seasonal_order=ordem[1]).fit(disp=False)
    return np.asarray(r.forecast(h))


# ----------------------------------------------------------------- IA (ML)
def montar_treino(d, cal, one_hot):
    """d: série de variação anual em log (sem NaN); cal: DataFrame de calendário (diferenças)."""
    vals = d.values
    idx = d.index
    X, y = [], []
    for i in range(max(LAGS), len(vals)):
        X.append(_linha(vals, i, idx[i], cal, one_hot))
        y.append(vals[i])
    return np.array(X), np.array(y)


def _linha(vals, i, data_alvo, cal, one_hot):
    lags = [vals[i - L] for L in LAGS]
    c = cal.loc[data_alvo]
    calend = [c["d_sabados"], c["d_domingos"], c["d_feriados"]]
    mes = data_alvo.month
    if one_hot:
        m = [1.0 if mes == k else 0.0 for k in range(1, 13)]
    else:
        m = [mes]
    return lags + calend + m


def prev_ml(logy, cal, fabrica, one_hot=False, h=H, escalonar=False):
    """Previsão recursiva de d e reconstrução do log-nível."""
    d = (logy - logy.shift(12)).dropna()
    X, yv = montar_treino(d, cal, one_hot)
    if escalonar:
        sx = StandardScaler().fit(X)
        my, sy = yv.mean(), yv.std()
        modelo = fabrica().fit(sx.transform(X), (yv - my) / sy)
        prever = lambda linha: modelo.predict(sx.transform(np.array([linha])))[0] * sy + my
    else:
        modelo = fabrica().fit(X, yv)
        prever = lambda linha: modelo.predict(np.array([linha]))[0]
    vals = list(d.values)
    datas = list(d.index)
    prev_d = []
    for k in range(1, h + 1):
        data_alvo = datas[-1] + pd.DateOffset(months=1)
        linha = _linha(np.array(vals + [np.nan]), len(vals), data_alvo, cal, one_hot)
        dk = prever(linha)
        prev_d.append(dk)
        vals.append(dk)
        datas.append(data_alvo)
    base = logy.iloc[-12:].values[:h]          # log y_{o+h-12}
    return base + np.array(prev_d)


class MediaSementes:
    """Média de várias redes MLP com sementes diferentes (reduz a variância da rede)."""
    def __init__(self, sementes=5, **kw):
        self.sementes, self.kw = sementes, kw

    def fit(self, X, y):
        self.ms = [MLPRegressor(random_state=s, **self.kw).fit(X, y) for s in range(self.sementes)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)


def fabrica_rf(max_features, min_samples_leaf):
    return lambda: RandomForestRegressor(n_estimators=500, max_features=max_features,
                                         min_samples_leaf=min_samples_leaf,
                                         random_state=42, n_jobs=1)


def fabrica_lgbm(num_leaves, min_child_samples):
    return lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=num_leaves,
                                     min_child_samples=min_child_samples, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.8, random_state=42,
                                     verbose=-1, n_jobs=1)


def fabrica_xgb(max_depth, min_child_weight):
    return lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=max_depth,
                                    min_child_weight=min_child_weight, subsample=0.8,
                                    colsample_bytree=0.8, random_state=42, n_jobs=1)


def fabrica_mlp(hidden, alpha):
    return lambda: MediaSementes(sementes=5, hidden_layer_sizes=hidden, alpha=alpha,
                                 solver="lbfgs", max_iter=3000, activation="relu")


GRADES = {
    "Random Forest": [dict(max_features=mf, min_samples_leaf=ml) for mf in (0.5, 1.0) for ml in (1, 5)],
    "LightGBM": [dict(num_leaves=nl, min_child_samples=mc) for nl in (4, 8) for mc in (5, 15)],
    "XGBoost": [dict(max_depth=md, min_child_weight=mw) for md in (2, 4) for mw in (1, 5)],
    "Rede Neural MLP": [dict(hidden=hd, alpha=a) for hd in ((8,), (16,)) for a in (0.1, 1.0)],
}
FABRICAS = {"Random Forest": fabrica_rf, "LightGBM": fabrica_lgbm,
            "XGBoost": fabrica_xgb, "Rede Neural MLP": fabrica_mlp}
ONE_HOT = {"Random Forest": False, "LightGBM": False, "XGBoost": False, "Rede Neural MLP": True}
ESCALONAR = {"Random Forest": False, "LightGBM": False, "XGBoost": False, "Rede Neural MLP": True}


def prepara_calendario(df):
    cal = df[["sabados", "domingos", "feriados"]].copy()
    for c in ["sabados", "domingos", "feriados"]:
        cal["d_" + c] = cal[c] - cal[c].shift(12)
    return cal
