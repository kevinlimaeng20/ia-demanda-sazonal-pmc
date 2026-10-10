"""
Verificação (comentário editorial C0): o padronizador da MLP é ajustado apenas à amostra
de treino de cada origem e depois aplicado, sem novo ajuste, às entradas de previsão?
Também registra as sementes das cinco redes. Gera ../saidas/revisao/verificacao_mlp.json.
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import modelos as M
from sklearn.preprocessing import StandardScaler

os.makedirs("../saidas/revisao", exist_ok=True)
df = pd.read_csv("../saidas/serie_preparada.csv", parse_dates=["data"]).set_index("data").asfreq("MS")
cal = M.prepara_calendario(df)
logy = np.log(df["indice_tratado"].dropna())
hip = json.load(open("../saidas/hiperparametros_escolhidos.json"))["Rede Neural MLP"]

registro = []
_fit, _transform = StandardScaler.fit, StandardScaler.transform


def fit_espiao(self, X, y=None, **kw):
    out = _fit(self, X, y, **kw)
    registro.append({"evento": "fit", "obj": id(self), "linhas": int(np.asarray(X).shape[0]),
                     "media_confere": bool(np.allclose(self.mean_, np.asarray(X).mean(axis=0)))})
    return out


def transform_espiao(self, X, *a, **kw):
    registro.append({"evento": "transform", "obj": id(self), "linhas": int(np.asarray(X).shape[0])})
    return _transform(self, X, *a, **kw)


StandardScaler.fit, StandardScaler.transform = fit_espiao, transform_espiao

resultado = []
for o in ["2022-07-01", "2024-07-01", "2026-06-01"]:
    registro.clear()
    tr = logy[:o]
    d = (tr - tr.shift(12)).dropna()
    X, yv = M.montar_treino(d, cal, one_hot=True)
    capturado = {}
    fab = M.fabrica_mlp(tuple(hip["hidden"]), hip["alpha"])

    def fab_espia():
        m = fab()
        capturado["modelo"] = m
        return m

    M.prev_ml(tr, cal, fab_espia, one_hot=True, escalonar=True)
    fits = [r for r in registro if r["evento"] == "fit"]
    trans = [r for r in registro if r["evento"] == "transform"]
    resultado.append({
        "origem": o,
        "ultima_data_de_treino": str(tr.index.max().date()),
        "linhas_de_treino_X": int(X.shape[0]),
        "ajustes_do_padronizador": len(fits),
        "linhas_no_ajuste": fits[0]["linhas"],
        "media_igual_a_media_do_treino": fits[0]["media_confere"],
        "transformacoes": len(trans),
        "transformacoes_de_previsao_1_linha": sum(1 for r in trans if r["linhas"] == 1),
        "mesmo_objeto_em_todas": len({r["obj"] for r in registro}) == 1,
        "sementes_das_redes": [m.random_state for m in capturado["modelo"].ms],
    })

StandardScaler.fit, StandardScaler.transform = _fit, _transform
ok = all(r["ajustes_do_padronizador"] == 1 and r["linhas_no_ajuste"] == r["linhas_de_treino_X"]
         and r["media_igual_a_media_do_treino"] and r["mesmo_objeto_em_todas"]
         and r["transformacoes_de_previsao_1_linha"] == 12 for r in resultado)
saida = {"verificacao_aprovada": ok, "origens_testadas": resultado,
         "sementes_outros_modelos": {"Random Forest": 42, "XGBoost": 42, "LightGBM": 42,
                                     "simulacao_monte_carlo (numpy default_rng)": 2026}}
json.dump(saida, open("../saidas/revisao/verificacao_mlp.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(saida, indent=1, ensure_ascii=False))
