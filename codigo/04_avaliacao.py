"""
Métricas de acurácia, testes de Diebold-Mariano, desempenho nos meses de pico e
simulação de Monte Carlo (bootstrap de trajetórias de erro) com avaliação de calibração.
Gera tabelas em ../saidas/resultados/.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

OUT = "../saidas/resultados"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)
N_SIM = 10_000

prev = pd.read_csv("../saidas/previsoes.csv", parse_dates=["origem", "data_alvo"])
serie = pd.read_csv("../saidas/serie_preparada.csv", parse_dates=["data"]).set_index("data")

ESTAT = ["Sazonal ingênuo", "Holt-Winters (ETS)", "SARIMA"]
IA = ["Random Forest", "XGBoost", "LightGBM", "Rede Neural MLP"]

# --- combinações pré-especificadas (média dos logs) ---------------------------------
def combina(df, membros, nome):
    sub = df[df.modelo.isin(membros)]
    g = sub.groupby(["periodo", "origem", "h", "data_alvo"], as_index=False).agg(
        prev=("prev", lambda v: float(np.exp(np.mean(np.log(v))))), real=("real", "first"),
        n=("prev", "size"))
    g = g[g.n == len(membros)].drop(columns="n")
    g["modelo"], g["config"] = nome, "média"
    return g

prev = pd.concat([prev,
                  combina(prev, IA, "Combinação IA"),
                  combina(prev, ["Holt-Winters (ETS)", "SARIMA"] + IA, "Combinação geral")],
                 ignore_index=True)
MODELOS = ESTAT + IA + ["Combinação IA", "Combinação geral"]
prev["erro_log"] = np.log(prev.real) - np.log(prev.prev)
prev["ape"] = 100 * (prev.real - prev.prev).abs() / prev.real
prev.to_csv("../saidas/previsoes_com_combinacoes.csv", index=False)

# --- escala do MASE: EAM do sazonal ingênuo dentro da amostra de treino de cada origem --
y_trat = serie["indice_tratado"].dropna()
escala = {}
for o in prev.origem.unique():
    tr = y_trat[:o]
    escala[pd.Timestamp(o)] = (tr - tr.shift(12)).abs().dropna().mean()
prev["ae_esc"] = (prev.real - prev.prev).abs() / prev.origem.map(escala)


def dm_hln(e1, e2, h):
    """Diebold-Mariano com correção de Harvey, Leybourne e Newbold (perda = erro absoluto %).
    Variância de longo prazo: autocovariâncias amostrais (divisor n) até a defasagem h-1, núcleo
    retangular; se a estimativa não for positiva, usa-se a variância de defasagem zero."""
    d = np.asarray(e1) - np.asarray(e2)
    n = len(d)
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        var = gam[0] / n
    dm = dbar / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * corr
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


def dm_por_horizonte(df, m_a, m_b, h):
    """Pares (origem) com valor real observado; devolve estatística e p-valor."""
    a = df[(df.modelo == m_a) & (df.h == h)].set_index("origem").ape
    b = df[(df.modelo == m_b) & (df.h == h)].set_index("origem").ape
    par = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().sort_index()
    st, p = dm_hln(par.a.values, par.b.values, h)
    return st, p, par.a.mean(), par.b.mean(), len(par)


def avalia(periodo):
    df = prev[(prev.periodo == periodo) & prev.real.notna()]
    linhas = []
    for m in MODELOS:
        g = df[df.modelo == m]
        r = dict(modelo=m,
                 MAPE=g.ape.mean(),
                 RMSE=np.sqrt(((g.real - g.prev) ** 2).mean()),
                 MASE=g.ae_esc.mean(),
                 MAPE_h1_3=g[g.h <= 3].ape.mean(),
                 MAPE_h4_6=g[(g.h >= 4) & (g.h <= 6)].ape.mean(),
                 MAPE_h7_12=g[g.h >= 7].ape.mean(),
                 MAPE_nov=g[g.data_alvo.dt.month == 11].ape.mean(),
                 MAPE_dez=g[g.data_alvo.dt.month == 12].ape.mean(),
                 MAPE_mai=g[g.data_alvo.dt.month == 5].ape.mean(),
                 vies_pct=100 * (g.prev / g.real - 1).mean(),
                 n=len(g))
        for ref, rot in [("Sazonal ingênuo", "ingenuo"), ("Holt-Winters (ETS)", "ets")]:
            if m == ref:
                continue
            melhor, pior = 0, 0
            for h in range(1, 13):
                st, p, *_ = dm_por_horizonte(df, m, ref, h)
                if p < 0.05 and st < 0:
                    melhor += 1
                elif p < 0.05 and st > 0:
                    pior += 1
            r[f"DM_melhor_{rot}"], r[f"DM_pior_{rot}"] = melhor, pior
        linhas.append(r)
    return pd.DataFrame(linhas)

tab_val = avalia("validacao")
tab_teste = avalia("teste")
tab_val.to_csv(f"{OUT}/metricas_validacao.csv", index=False)
tab_teste.to_csv(f"{OUT}/metricas_teste.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("=== VALIDAÇÃO"); print(tab_val.round(2).to_string(index=False))
print("=== TESTE"); print(tab_teste.round(2).to_string(index=False))

# MAPE por horizonte (teste)
por_h = prev[prev.periodo == "teste"].pivot_table(index="h", columns="modelo", values="ape", aggfunc="mean")[MODELOS]
por_h.to_csv(f"{OUT}/mape_por_horizonte_teste.csv")
print(por_h.round(1).to_string())

# DM entre o melhor modelo de IA e o melhor estatístico (por horizonte), no teste
t = tab_teste.set_index("modelo")
melhor_est = t.loc[["Holt-Winters (ETS)", "SARIMA"], "MAPE"].idxmin()
melhor_ia = t.loc[IA, "MAPE"].idxmin()
dm_rows = []
te = prev[prev.periodo == "teste"]
te_obs = te[te.real.notna()]
for h in range(1, 13):
    st, p, ma, mb, n = dm_por_horizonte(te_obs, melhor_ia, melhor_est, h)
    dm_rows.append(dict(h=h, mape_ia=ma, mape_est=mb, estat=st, p=p, n=n))
dm_tab = pd.DataFrame(dm_rows)

# comparação descritiva restrita aos alvos de dezembro (a análise por evento está em 08_analises_revisao.py)
dz = te_obs[te_obs.data_alvo.dt.month == 12]
par = dz[dz.modelo.isin([melhor_ia, melhor_est])].pivot_table(index=["origem", "h"], columns="modelo",
                                                              values="ape").dropna()
json.dump({"n_previsoes_dezembro": int(len(par)), "mape_dez_ia": float(par[melhor_ia].mean()),
           "mape_dez_est": float(par[melhor_est].mean()),
           "ia_melhor_em_pct_das_previsoes": float(100 * (par[melhor_ia] < par[melhor_est]).mean())},
          open(f"{OUT}/dezembro_ia_vs_estat.json", "w"), indent=1)
print("Dezembro:", json.load(open(f"{OUT}/dezembro_ia_vs_estat.json")))
dm_tab.to_csv(f"{OUT}/dm_melhor_ia_vs_melhor_estat.csv", index=False)
print(f"DM {melhor_ia} x {melhor_est}:"); print(dm_tab.round(3).to_string(index=False))

# Dezembro previsto em julho (h = 5) em cada ano do teste
dez = te[(te.origem.dt.month == 7) & (te.h == 5)].pivot_table(
    index="data_alvo", columns="modelo", values="prev")[MODELOS]
dez.insert(0, "Real", te[(te.origem.dt.month == 7) & (te.h == 5)].groupby("data_alvo").real.first())
dez.to_csv(f"{OUT}/dezembro_previsto_em_julho.csv")
print(dez.round(1).to_string())

# --- simulação de Monte Carlo ------------------------------------------------------------
def trajetorias(m, ate=None):
    """Trajetórias completas (12 horizontes) de erro em log observadas até a data 'ate'."""
    g = prev[(prev.modelo == m) & (prev.periodo.isin(["validacao", "teste"]))].dropna(subset=["real"])
    if ate is not None:
        g = g[g.data_alvo <= ate]
    piv = g.pivot_table(index="origem", columns="h", values="erro_log")
    return piv.dropna()  # somente origens com os 12 horizontes observados


def crps_amostra(x, y):
    x = np.sort(x)
    n = len(x)
    termo1 = np.mean(np.abs(x - y))
    # E|X - X'| para amostra ordenada: 2/n^2 * sum((2i - n - 1) x_i)
    i = np.arange(1, n + 1)
    termo2 = 2.0 / n ** 2 * np.sum((2 * i - n - 1) * x)
    return termo1 - 0.5 * termo2


calib = []
for m in MODELOS:
    te_m = te[te.modelo == m]
    for o, go in te_m.groupby("origem"):
        pool = trajetorias(m, ate=o)
        idx = rng.integers(0, len(pool), size=N_SIM)
        E = pool.values[idx]                                   # N_SIM x 12
        for _, r in go.iterrows():
            if np.isnan(r.real):
                continue
            sims = r.prev * np.exp(E[:, r.h - 1])
            q = np.percentile(sims, [5, 10, 50, 90, 95])
            calib.append(dict(modelo=m, origem=o, h=r.h, data_alvo=r.data_alvo, real=r.real,
                              p05=q[0], p10=q[1], p50=q[2],
                              p90=q[3], p95=q[4], tam_pool=len(pool),
                              crps=crps_amostra(sims[:2000], r.real)))
calib = pd.DataFrame(calib)
calib["cob80"] = (calib.real >= calib.p10) & (calib.real <= calib.p90)
calib["cob90"] = (calib.real >= calib.p05) & (calib.real <= calib.p95)
calib["larg80_pct"] = 100 * (calib.p90 - calib.p10) / calib.p50
calib.to_csv(f"{OUT}/simulacao_teste_detalhe.csv", index=False)
tab_cal = calib.groupby("modelo").agg(cobertura_80=("cob80", "mean"), cobertura_90=("cob90", "mean"),
                                      largura_80_pct=("larg80_pct", "mean"), CRPS=("crps", "mean"),
                                      n=("crps", "size")).reindex(MODELOS)
tab_cal[["cobertura_80", "cobertura_90"]] *= 100
tab_cal.to_csv(f"{OUT}/simulacao_calibracao_teste.csv")
print("=== CALIBRAÇÃO (teste)"); print(tab_cal.round(2).to_string())

# calibração restrita aos meses de pico (novembro e dezembro)
pk = calib[pd.to_datetime(calib.data_alvo).dt.month.isin([11, 12])]
tab_pk = pk.groupby("modelo").agg(cobertura_80=("cob80", "mean"), cobertura_90=("cob90", "mean"),
                                  CRPS=("crps", "mean"), n=("crps", "size")).reindex(MODELOS)
tab_pk[["cobertura_80", "cobertura_90"]] *= 100
tab_pk.to_csv(f"{OUT}/simulacao_calibracao_picos_teste.csv")
print("=== CALIBRAÇÃO nov/dez (teste)"); print(tab_pk.round(2).to_string())
# --- simulação final: origem jul/2026, cenários para ago/2026-jul/2027 ----------------------
fin = prev[prev.periodo == "final"]
dez25 = serie.loc["2025-12-01", "indice"]
mai26 = serie.loc["2026-05-01", "indice"]
cen_linhas, resumo = [], []
paths = {}
for m in MODELOS:
    g = fin[fin.modelo == m].sort_values("h")
    pool = trajetorias(m, ate=pd.Timestamp("2026-07-01"))
    idx = rng.integers(0, len(pool), size=N_SIM)
    P = g.prev.values[None, :] * np.exp(pool.values[idx])      # N_SIM x 12 cenários
    paths[m] = P
    for k, dt in enumerate(g.data_alvo):
        q = np.percentile(P[:, k], [5, 10, 50, 90, 95])
        cen_linhas.append(dict(modelo=m, data_alvo=dt, previsao_pontual=g.prev.values[k],
                               p05=q[0], p10=q[1], p50=q[2], p90=q[3], p95=q[4]))
    meses = list(g.data_alvo)
    i_nov, i_dez, i_mai = meses.index(pd.Timestamp("2026-11-01")), meses.index(pd.Timestamp("2026-12-01")), \
        meses.index(pd.Timestamp("2027-05-01"))
    novdez = P[:, i_nov] + P[:, i_dez]
    resumo.append(dict(modelo=m, tam_pool=len(pool),
                       dez_p50=np.median(P[:, i_dez]), dez_p90=np.percentile(P[:, i_dez], 90),
                       margem_p90_sobre_p50_pct=100 * (np.percentile(P[:, i_dez], 90) / np.median(P[:, i_dez]) - 1),
                       prob_dez26_maior_dez25=100 * (P[:, i_dez] > dez25).mean(),
                       prob_mai27_maior_mai26=100 * (P[:, i_mai] > mai26).mean(),
                       novdez_p10=np.percentile(novdez, 10), novdez_p50=np.median(novdez),
                       novdez_p90=np.percentile(novdez, 90)))
pd.DataFrame(cen_linhas).to_csv(f"{OUT}/cenarios_finais.csv", index=False)
res_fin = pd.DataFrame(resumo).set_index("modelo")
res_fin.to_csv(f"{OUT}/cenarios_finais_resumo.csv")
print("=== CENÁRIOS FINAIS (dez/2025 real = %.1f; mai/2026 real = %.1f)" % (dez25, mai26))
print(res_fin.round(1).to_string())
np.save(f"{OUT}/paths_final_{'ets'}.npy", paths["Holt-Winters (ETS)"])
np.save(f"{OUT}/paths_final_{'mlp'}.npy", paths["Rede Neural MLP"])

json.dump({"melhor_estatistico": melhor_est, "melhor_ia": melhor_ia},
          open(f"{OUT}/melhores.json", "w"), ensure_ascii=False, indent=1)
print("ok")
