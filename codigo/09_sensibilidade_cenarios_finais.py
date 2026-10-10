"""Sensibilidade dos cenários finais (origem jul/2026) ao conjunto de trajetórias reamostradas."""
import numpy as np, pandas as pd
prev = pd.read_csv("../saidas/previsoes_com_combinacoes.csv", parse_dates=["origem", "data_alvo"])
serie = pd.read_csv("../saidas/serie_preparada.csv", parse_dates=["data"]).set_index("data")["indice"]
fin = prev[prev.periodo == "final"]
ORIGEM = pd.Timestamp("2026-07-01")
linhas = []
for m in ("Holt-Winters (ETS)", "Rede Neural MLP"):
    g = prev[(prev.modelo == m) & prev.periodo.isin(["validacao", "teste"])].dropna(subset=["real"])
    p = g.pivot_table(index="origem", columns="h", values="erro_log").dropna()
    p = p[p.index + pd.DateOffset(months=12) <= ORIGEM]
    ini, fim = p.index + pd.DateOffset(months=1), p.index + pd.DateOffset(months=12)
    pools = {"base": p, "sem_recessao": p[~((fim >= "2015-01-01") & (ini <= "2016-12-01"))], "recentes_36": p.iloc[-36:]}
    f = fin[fin.modelo == m].sort_values("h")
    # o conjunto completo ("base") vem dos resultados oficiais (Tabela 5), gerados por 04_avaliacao.py
    c = pd.read_csv("../saidas/resultados/cenarios_finais.csv", parse_dates=["data_alvo"])
    r = pd.read_csv("../saidas/resultados/cenarios_finais_resumo.csv").set_index("modelo").loc[m]
    d = c[(c.modelo == m) & (c.data_alvo == "2026-12-01")].iloc[0]
    linhas.append(dict(modelo=m, conjunto="base", trajetorias=len(p), dez_p10=d.p10, dez_p50=d.p50, dez_p90=d.p90,
                       margem_p90_p50=100 * (d.p90 / d.p50 - 1), prob_supera_dez25=r.prob_dez26_maior_dez25))
    for nome, pl in pools.items():
        if nome == "base":
            continue
        rng = np.random.default_rng(2026)
        P = f.prev.values[None, :] * np.exp(pl.values[rng.integers(0, len(pl), size=10_000)])
        dz = P[:, 4]  # dezembro de 2026 (h = 5)
        linhas.append(dict(modelo=m, conjunto=nome, trajetorias=len(pl), dez_p10=np.percentile(dz, 10),
                           dez_p50=np.median(dz), dez_p90=np.percentile(dz, 90),
                           margem_p90_p50=100 * (np.percentile(dz, 90) / np.median(dz) - 1),
                           prob_supera_dez25=100 * (dz > serie.loc["2025-12-01"]).mean()))
t = pd.DataFrame(linhas)
t.to_csv("../saidas/revisao/cenarios_finais_sensibilidade_pool.csv", index=False)
print(t.round(1).to_string(index=False))
