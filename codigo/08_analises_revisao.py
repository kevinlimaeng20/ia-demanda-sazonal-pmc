"""
Análises adicionais pedidas na revisão editorial (comentários C1, C2, C4, C5 e C6).
Lê as previsões já geradas (03_backtest.py / 04_avaliacao.py) e grava tabelas em ../saidas/revisao/.

C1  avaliação em pseudo-tempo real com as datas oficiais de divulgação da PMC
C2  Diebold-Mariano documentado: perda, variância de longo prazo, defasagens, Holm, p por horizonte
C4  sensibilidade por horizonte com os mesmos 37 meses-alvo (jul/2023-jul/2026)
C5  dezembros como eventos (validação + teste), teste do sinal por evento
C6  cobertura/CRPS por horizonte e mês-alvo; sensibilidade do conjunto de trajetórias do bootstrap;
    intervalos de confiança das coberturas por bootstrap em blocos de origens
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

OUT = "../saidas/revisao"
os.makedirs(OUT, exist_ok=True)
prev = pd.read_csv("../saidas/previsoes_com_combinacoes.csv", parse_dates=["origem", "data_alvo"])
prev["origem"] = prev.origem.dt.to_period("M").dt.to_timestamp()
ETS, MLP, SAR, ING, CG = "Holt-Winters (ETS)", "Rede Neural MLP", "SARIMA", "Sazonal ingênuo", "Combinação geral"
MODELOS = [ING, ETS, SAR, "Random Forest", "XGBoost", "LightGBM", MLP, "Combinação IA", CG]
te = prev[(prev.periodo == "teste") & prev.real.notna()].copy()
va = prev[(prev.periodo == "validacao")].copy()
resumo = {}


# ----------------------------------------------------------------------------- C2: DM
def dm_hln(d, h):
    """d: diferenças de perda em ordem cronológica de origem. Variância de longo prazo com
    autocovariâncias até a defasagem h-1 (núcleo retangular, Diebold e Mariano, 1995);
    se a estimativa não for positiva, usa-se apenas a variância (defasagem 0)."""
    d = np.asarray(d, float)
    n = len(d)
    db = d.mean()
    gam = [np.sum((d[k:] - db) * (d[:n - k] - db)) / n for k in range(h)]
    lrv = gam[0] + 2 * sum(gam[1:])
    recuo = lrv <= 0
    if recuo:
        lrv = gam[0]
    dm = db / np.sqrt(lrv / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    est = dm * corr
    p = 2 * stats.t.sf(abs(est), df=n - 1)
    return est, p, recuo


def holm(p):
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    acum = 0.0
    for i, k in enumerate(ordem):
        acum = max(acum, (m - i) * p[k])
        aj[k] = min(1.0, acum)
    return aj


def dm_tabela(df, a, b, rotulo):
    linhas = []
    for h in range(1, 13):
        x = df[(df.modelo == a) & (df.h == h)].set_index("origem").ape
        y = df[(df.modelo == b) & (df.h == h)].set_index("origem").ape
        par = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna().sort_index()
        est, p, recuo = dm_hln(par.a - par.b, h)
        linhas.append(dict(h=h, n=len(par), mape_a=par.a.mean(), mape_b=par.b.mean(),
                           dif=par.a.mean() - par.b.mean(), estat=est, p=p, recuo_variancia=recuo))
    t = pd.DataFrame(linhas)
    t["p_holm"] = holm(t.p)
    t["comparacao"] = rotulo
    return t


dm_mlp_ets = dm_tabela(te, MLP, ETS, "MLP x ETS (teste)")
dm_mlp_ets.to_csv(f"{OUT}/dm_mlp_vs_ets_por_horizonte.csv", index=False)
contagens = []
for m in MODELOS:
    for ref in (ING, ETS):
        if m == ref:
            continue
        t = dm_tabela(te, m, ref, f"{m} x {ref}")
        contagens.append(dict(modelo=m, referencia=ref,
                              melhor_5pct=int(((t.p < .05) & (t.estat < 0)).sum()),
                              pior_5pct=int(((t.p < .05) & (t.estat > 0)).sum()),
                              melhor_holm=int(((t.p_holm < .05) & (t.estat < 0)).sum()),
                              pior_holm=int(((t.p_holm < .05) & (t.estat > 0)).sum()),
                              recuos_variancia=int(t.recuo_variancia.sum())))
contagens = pd.DataFrame(contagens)
contagens.to_csv(f"{OUT}/dm_contagens_holm.csv", index=False)
resumo["dm_mlp_ets"] = dm_mlp_ets.round(4).to_dict("records")
resumo["dm_recuos_variancia_total"] = int(contagens.recuos_variancia.sum()) + int(dm_mlp_ets.recuo_variancia.sum())

# ----------------------------------------------------------------------------- C4: 37 alvos comuns
comum = te[(te.data_alvo >= "2023-07-01") & (te.data_alvo <= "2026-07-01")]
n_alvos = comum.groupby(["modelo", "h"]).size().unstack()
assert (n_alvos == 37).all().all(), n_alvos
mape_comum = comum.pivot_table(index="h", columns="modelo", values="ape", aggfunc="mean")[MODELOS]
mape_comum.to_csv(f"{OUT}/mape_por_horizonte_37_alvos_comuns.csv")
dm_comum = dm_tabela(comum, MLP, ETS, "MLP x ETS (37 alvos comuns)")
dm_comum.to_csv(f"{OUT}/dm_mlp_vs_ets_37_alvos_comuns.csv", index=False)
mape_total = te.pivot_table(index="h", columns="modelo", values="ape", aggfunc="mean")[MODELOS]
resumo["comum_mape_ets_h"] = mape_comum[ETS].round(2).tolist()
resumo["comum_mape_mlp_h"] = mape_comum[MLP].round(2).tolist()
resumo["comum_media_h1_6"] = {m: round(mape_comum.loc[1:6, m].mean(), 2) for m in (ETS, MLP, SAR, ING)}
resumo["comum_media_h7_12"] = {m: round(mape_comum.loc[7:12, m].mean(), 2) for m in (ETS, MLP, SAR, ING)}
resumo["total_media_h7_12"] = {m: round(mape_total.loc[7:12, m].mean(), 2) for m in (ETS, MLP, SAR, ING)}

# ----------------------------------------------------------------------------- C1: pseudo-tempo real
cald = pd.read_csv("../dados_ibge/calendario_divulgacao_pmc.csv")
cald["mes_referencia"] = pd.to_datetime(cald.mes_referencia + "-01")
cald["data_divulgacao"] = pd.to_datetime(cald.data_divulgacao)
cald["mes_divulgacao"] = cald.data_divulgacao.dt.to_period("M").dt.to_timestamp()
cald["defasagem_meses"] = ((cald.mes_divulgacao.dt.year - cald.mes_referencia.dt.year) * 12
                           + cald.mes_divulgacao.dt.month - cald.mes_referencia.dt.month)
cald.to_csv(f"{OUT}/calendario_divulgacao_com_defasagem.csv", index=False)
rt = te.merge(cald[["mes_referencia", "data_divulgacao", "mes_divulgacao", "defasagem_meses"]],
              left_on="origem", right_on="mes_referencia", how="left")
assert rt.data_divulgacao.notna().all()
rt["antecedencia_real"] = ((rt.data_alvo.dt.year - rt.mes_divulgacao.dt.year) * 12
                           + rt.data_alvo.dt.month - rt.mes_divulgacao.dt.month)
rt_fut = rt[rt.antecedencia_real >= 0]     # alvos ainda não ocorridos no mês da divulgação
mape_rt = rt_fut.pivot_table(index="antecedencia_real", columns="modelo", values="ape", aggfunc="mean")[MODELOS]
n_rt = rt_fut[rt_fut.modelo == ETS].groupby("antecedencia_real").size()
mape_rt.insert(0, "n", n_rt)
mape_rt.to_csv(f"{OUT}/mape_por_antecedencia_real.csv")
# dezembro em tempo real: previsões disponíveis em cada mês de divulgação
dz_rt = rt_fut[rt_fut.data_alvo.dt.month == 12].pivot_table(
    index=["data_alvo", "mes_divulgacao", "origem", "h", "antecedencia_real"], columns="modelo", values="ape").reset_index()
dz_rt.to_csv(f"{OUT}/dezembro_tempo_real_detalhe.csv", index=False)
resumo["defasagens_observadas"] = cald.defasagem_meses.value_counts().to_dict()
resumo["rt_n_previsoes"] = int(len(rt_fut[rt_fut.modelo == ETS]))
resumo["rt_descartadas_nowcast"] = int(len(rt[(rt.modelo == ETS) & (rt.antecedencia_real < 0)]))
resumo["rt_mape_global"] = {m: round(rt_fut[rt_fut.modelo == m].ape.mean(), 2) for m in MODELOS}

# ----------------------------------------------------------------------------- C5: dezembros como eventos
dz = pd.concat([va, te])
dz = dz[(dz.data_alvo.dt.month == 12) & dz.real.notna() & dz.modelo.isin([ETS, MLP, SAR, ING])]
ev = dz.pivot_table(index=["data_alvo"], columns="modelo", values="ape", aggfunc="mean")
ev["n_previsoes"] = dz[dz.modelo == ETS].groupby("data_alvo").size()
ev["periodo"] = np.where(ev.index.year >= 2022, "teste", "validação")
ev["mlp_menor_que_ets"] = ev[MLP] < ev[ETS]
ev.to_csv(f"{OUT}/dezembros_por_evento.csv")


def sinal(sub):
    k, n = int(sub.mlp_menor_que_ets.sum()), len(sub)
    return dict(n_eventos=n, mlp_vence=k, p_binomial_bilateral=float(stats.binomtest(k, n, 0.5).pvalue))


resumo["dez_teste"] = sinal(ev[ev.periodo == "teste"])
resumo["dez_validacao"] = sinal(ev[ev.periodo == "validação"])
resumo["dez_todos"] = sinal(ev)
resumo["dez_mape_validacao"] = {m: round(dz[(dz.data_alvo.dt.year < 2022) & (dz.modelo == m)].ape.mean(), 2) for m in (ETS, MLP)}
resumo["dez_mape_teste"] = {m: round(dz[(dz.data_alvo.dt.year >= 2022) & (dz.modelo == m)].ape.mean(), 2) for m in (ETS, MLP)}

# ----------------------------------------------------------------------------- C6: simulação
rng_base = 2026
N_SIM = 10_000


def crps_amostra(x, y):
    x = np.sort(x)
    n = len(x)
    i = np.arange(1, n + 1)
    return np.mean(np.abs(x - y)) - 0.5 * (2.0 / n ** 2 * np.sum((2 * i - n - 1) * x))


trajs = {}
for m in MODELOS:
    g = prev[(prev.modelo == m) & prev.periodo.isin(["validacao", "teste"])].dropna(subset=["real"])
    piv = g.pivot_table(index="origem", columns="h", values="erro_log").dropna()
    trajs[m] = piv  # trajetórias completas (12 horizontes), indexadas pela origem


def pool(m, o, variante):
    p = trajs[m]
    fim = p.index + pd.DateOffset(months=12)          # último alvo de cada trajetória
    p = p[fim <= o]                                    # completamente observadas até a origem
    if variante == "sem_recessao":
        ini = p.index + pd.DateOffset(months=1)
        fim2 = p.index + pd.DateOffset(months=12)
        p = p[~((fim2 >= "2015-01-01") & (ini <= "2016-12-01"))]
    elif variante == "recentes_36":
        p = p.iloc[-36:]
    elif variante == "so_teste":
        p = p[p.index >= "2022-07-01"]
    return p


def simula(m, variante, origens=None):
    rng = np.random.default_rng(rng_base)
    g = te[te.modelo == m]
    linhas = []
    for o, go in g.groupby("origem"):
        if origens is not None and o not in origens:
            continue
        pl = pool(m, o, variante)
        if len(pl) == 0:
            continue
        E = pl.values[rng.integers(0, len(pl), size=N_SIM)]
        for _, r in go.iterrows():
            sims = r.prev * np.exp(E[:, r.h - 1])
            q = np.percentile(sims, [5, 10, 50, 90, 95])
            linhas.append(dict(modelo=m, variante=variante, origem=o, h=r.h, data_alvo=r.data_alvo, real=r.real,
                               p05=q[0], p10=q[1], p50=q[2], p90=q[3], p95=q[4], tam_pool=len(pl),
                               crps=crps_amostra(sims[:2000], r.real)))
    s = pd.DataFrame(linhas)
    s["cob80"] = (s.real >= s.p10) & (s.real <= s.p90)
    s["cob90"] = (s.real >= s.p05) & (s.real <= s.p95)
    s["larg80"] = 100 * (s.p90 - s.p10) / s.p50
    return s


def agrega(s):
    return pd.Series(dict(n=len(s), cob80=100 * s.cob80.mean(), cob90=100 * s.cob90.mean(),
                          larg80=s.larg80.mean(), crps=s.crps.mean()))


# o conjunto completo ("base") é exatamente a simulação da Tabela 4, gravada por 04_avaliacao.py,
# para que coberturas e CRPS deste script coincidam com os do artigo
_det = pd.read_csv("../saidas/resultados/simulacao_teste_detalhe.csv", parse_dates=["origem", "data_alvo"])
_det = _det.rename(columns={"larg80_pct": "larg80"})
base = {m: _det[_det.modelo == m].assign(variante="base").reset_index(drop=True) for m in (ETS, MLP)}
por_h = pd.concat({m: base[m].groupby("h").apply(agrega) for m in base}, axis=1)
por_mes = pd.concat({m: base[m].groupby(base[m].data_alvo.dt.month).apply(agrega) for m in base}, axis=1)
por_h.to_csv(f"{OUT}/simulacao_por_horizonte.csv")
por_mes.to_csv(f"{OUT}/simulacao_por_mes_alvo.csv")

# sensibilidade do conjunto de trajetórias
sens = []
origens_so_teste = sorted({o for o in te.origem.unique() if len(pool(ETS, o, "so_teste")) >= 12})
for m in (ETS, MLP):
    for var in ("base", "sem_recessao", "recentes_36"):
        s = base[m] if var == "base" else simula(m, var)
        r = agrega(s); r["variante"], r["modelo"], r["recorte"] = var, m, "teste completo"
        r["pool_min"], r["pool_max"] = s.tam_pool.min(), s.tam_pool.max()
        sens.append(r)
    for var in ("base", "so_teste"):
        s = base[m][base[m].origem.isin(origens_so_teste)] if var == "base" else \
            simula(m, var, origens=set(origens_so_teste))
        r = agrega(s); r["variante"], r["modelo"], r["recorte"] = var, m, "origens a partir de " + str(origens_so_teste[0].date())
        r["pool_min"], r["pool_max"] = s.tam_pool.min(), s.tam_pool.max()
        sens.append(r)
sens = pd.DataFrame(sens)
sens.to_csv(f"{OUT}/simulacao_sensibilidade_pool.csv", index=False)


# intervalos de confiança das coberturas: bootstrap circular em blocos de 12 origens consecutivas
# (o bloco circular dá a mesma probabilidade a todas as origens, evitando o viés de borda)
def ic_blocos(s, col, B=2000, L=12, seed=7):
    rng = np.random.default_rng(seed)
    origens = sorted(s.origem.unique())
    por_o = {o: s[s.origem == o][col].values for o in origens}
    n = len(origens)
    blocos = [[origens[(i + j) % n] for j in range(L)] for i in range(n)]
    k = int(np.ceil(n / L))
    est = []
    for _ in range(B):
        esc = [blocos[j] for j in rng.integers(0, len(blocos), size=k)]
        vals = np.concatenate([por_o[o] for b in esc for o in b])
        est.append(100 * vals.mean())
    return np.percentile(est, [2.5, 97.5])


ics = []
for m in (ETS, MLP):
    for col in ("cob80", "cob90"):
        lo, hi = ic_blocos(base[m], col)
        ics.append(dict(modelo=m, medida=col, estimativa=100 * base[m][col].mean(), ic95_inf=lo, ic95_sup=hi))
ics = pd.DataFrame(ics)
ics.to_csv(f"{OUT}/simulacao_ic_coberturas_blocos.csv", index=False)

json.dump(resumo, open(f"{OUT}/resumo_revisao.json", "w"), indent=1, ensure_ascii=False, default=str)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("== DM MLP x ETS (teste)\n", dm_mlp_ets.round(4).to_string(index=False))
print("== contagens DM (5% e Holm)\n", contagens.to_string(index=False))
print("== 37 alvos comuns: MAPE por h\n", mape_comum.round(2).to_string())
print("== DM MLP x ETS (37 alvos)\n", dm_comum.round(4).to_string(index=False))
print("== pseudo-tempo real\n", mape_rt.round(2).to_string())
print("== dezembros\n", ev.round(2).to_string())
print("== simulação por horizonte\n", por_h.round(2).to_string())
print("== simulação por mês-alvo\n", por_mes.round(2).to_string())
print("== sensibilidade\n", sens.round(2).to_string(index=False))
print("== IC blocos\n", ics.round(1).to_string(index=False))
print(json.dumps({k: v for k, v in resumo.items() if k != "dm_mlp_ets"}, indent=1, ensure_ascii=False, default=str))
