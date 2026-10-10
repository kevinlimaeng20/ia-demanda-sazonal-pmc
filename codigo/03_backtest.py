"""
Avaliação fora da amostra com origem móvel (rolling origin) e janela expansível.

- Validação: origens jan/2014-fev/2019 (alvos até fev/2020, antes da pandemia).
  Usada para escolher os hiperparâmetros dos modelos de IA e para formar o
  conjunto histórico de trajetórias de erro da simulação de Monte Carlo.
- Teste: origens jul/2022-jun/2026 (alvos ago/2022-jul/2026), com os hiperparâmetros
  fixados na validação.
- Final: origem jul/2026 (último dado), previsões ago/2026-jul/2027.
"""
import sys, json, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, ".")
import modelos as M

SAIDA = sys.argv[1]
df = pd.read_csv("../saidas/serie_preparada.csv", parse_dates=["data"]).set_index("data").asfreq("MS")
cal = M.prepara_calendario(df)
real = df["indice"]
logy_trat = np.log(df["indice_tratado"].dropna())

VALID = pd.date_range("2014-01-01", "2019-02-01", freq="MS")
TESTE = pd.date_range("2022-07-01", "2026-06-01", freq="MS")
FINAL = [pd.Timestamp("2026-07-01")]


def roda_origem(o, periodo, ordem_sarima, configs_ml):
    tr = logy_trat[:o]
    alvos = pd.date_range(o + pd.DateOffset(months=1), periods=M.H, freq="MS")
    saida = []

    def guarda(modelo, cfg, prev_log):
        for h, (dt, p) in enumerate(zip(alvos, np.exp(prev_log)), start=1):
            saida.append(dict(periodo=periodo, modelo=modelo, config=cfg, origem=o, h=h,
                              data_alvo=dt, prev=p, real=real.get(dt, np.nan)))

    guarda("Sazonal ingênuo", "-", M.prev_snaive(tr))
    f, cfg = M.prev_ets(tr, devolver_config=True)
    guarda("Holt-Winters (ETS)", json.dumps(cfg), f)
    guarda("SARIMA", str(ordem_sarima), M.prev_sarima(tr, ordem_sarima))
    for nome, lista in configs_ml.items():
        for p in lista:
            f = M.prev_ml(tr, cal, M.FABRICAS[nome](**p), one_hot=M.ONE_HOT[nome],
                          escalonar=M.ESCALONAR[nome])
            guarda(nome, json.dumps(p, default=str), f)
    return saida


def roda_periodo(origens, periodo, configs_ml):
    t = time.time()
    ordem = M.seleciona_sarima(logy_trat[:origens[0]])
    print(f"[{periodo}] SARIMA escolhido por AICc na 1ª origem: {ordem}", flush=True)
    res = Parallel(n_jobs=2)(delayed(roda_origem)(o, periodo, ordem, configs_ml) for o in origens)
    out = pd.DataFrame([r for bloco in res for r in bloco])
    print(f"[{periodo}] {len(origens)} origens em {time.time()-t:.0f}s", flush=True)
    return out, ordem


def mape(g):
    return (100 * (g.real - g.prev).abs() / g.real).mean()


# 1) validação com toda a grade de hiperparâmetros
val, ordem_val = roda_periodo(VALID, "validacao", M.GRADES)
escolha = {}
for nome in M.GRADES:
    m = val[val.modelo == nome].groupby("config").apply(mape).sort_values()
    print(nome, "\n", m.round(2).to_string(), flush=True)
    escolha[nome] = [json.loads(m.index[0])]
    if nome == "Rede Neural MLP":
        escolha[nome][0]["hidden"] = tuple(escolha[nome][0]["hidden"])
json.dump({k: v[0] for k, v in escolha.items()}, open("../saidas/hiperparametros_escolhidos.json", "w"),
          default=str, indent=1)
# mantém na validação apenas a configuração escolhida (para o pool de erros da simulação)
keep = val.modelo.isin(["Sazonal ingênuo", "Holt-Winters (ETS)", "SARIMA"])
for nome in M.GRADES:
    keep |= (val.modelo == nome) & (val.config == json.dumps(escolha[nome][0], default=str))
val_escolhida = val[keep]

# 2) teste e 3) previsão final com os hiperparâmetros fixados
teste, ordem_teste = roda_periodo(TESTE, "teste", escolha)
final, _ = roda_periodo(FINAL, "final", escolha)

todos = pd.concat([val_escolhida, teste, final], ignore_index=True)
todos.to_csv(SAIDA, index=False)
val.to_csv("../saidas/validacao_grade_completa.csv", index=False)
json.dump({"sarima_validacao": str(ordem_val), "sarima_teste": str(ordem_teste)},
          open("../saidas/ordens_sarima.json", "w"), indent=1)
print("ok", todos.shape)
