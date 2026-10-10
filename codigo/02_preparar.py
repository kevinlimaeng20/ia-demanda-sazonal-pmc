"""
Preparação da série para modelagem.

1) Tratamento do choque da pandemia de COVID-19 (apenas para estimação dos modelos):
   os meses de fechamento compulsório do comércio (mar-jul/2020 e mar-abr/2021) são
   substituídos por valores imputados via interpolação linear da série dessazonalizada
   (log). Os fatores sazonais vêm de uma decomposição STL robusta ajustada somente ao
   período pré-pandemia (jan/2000-fev/2020), usando os fatores dos 12 últimos meses
   desse ajuste. Nenhum dado posterior à primeira origem de teste é usado.
2) Variáveis de calendário (conhecidas antecipadamente): nº de sábados, domingos e
   feriados nacionais (seg-sáb) por mês.
"""
import sys
from datetime import date, timedelta
import numpy as np
import pandas as pd
from dateutil.easter import easter
from statsmodels.tsa.seasonal import STL

src, out = sys.argv[1], sys.argv[2]
y = pd.read_csv(src, parse_dates=["data"]).set_index("data")["indice"].asfreq("MS")

JANELA_COVID = list(pd.date_range("2020-03-01", "2020-07-01", freq="MS")) + \
               list(pd.date_range("2021-03-01", "2021-04-01", freq="MS"))
CORTE_SEM_VAZAMENTO = "2022-07-01"   # primeira origem de previsão do período de teste

# --- 1) imputação ---------------------------------------------------------------
logy = np.log(y)
pre = logy[:"2020-02-01"]
stl = STL(pre, period=12, robust=True).fit()
fator_mes = stl.seasonal[-12:].groupby(stl.seasonal[-12:].index.month).mean()  # log
hist = logy[:CORTE_SEM_VAZAMENTO]
saz = pd.Series(fator_mes.loc[hist.index.month].values, index=hist.index)
dessaz_imp = (hist - saz).copy()
dessaz_imp[JANELA_COVID] = np.nan
dessaz_imp = dessaz_imp.interpolate(method="linear")
logy_trat = logy.copy()
logy_trat[JANELA_COVID] = (dessaz_imp + saz)[JANELA_COVID]
y_trat = np.exp(logy_trat)
print("Fatores sazonais pré-pandemia (multiplicativos):",
      np.exp(fator_mes).round(3).tolist())

# --- 2) calendário ----------------------------------------------------------------
def feriados_nacionais(ano):
    fixos = [(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25)]
    f = {date(ano, m, d) for m, d in fixos}
    if ano >= 2024:                       # Lei 14.759/2023 - Dia da Consciência Negra
        f.add(date(ano, 11, 20))
    p = easter(ano)
    f |= {p - timedelta(days=48), p - timedelta(days=47),   # carnaval (seg e ter)
          p - timedelta(days=2),                              # sexta-feira santa
          p + timedelta(days=60)}                             # corpus christi
    return f

def calendario(idx):
    linhas = []
    for d in idx:
        dias = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="D")
        fer = feriados_nacionais(d.year)
        linhas.append({"data": d,
                       "sabados": int((dias.dayofweek == 5).sum()),
                       "domingos": int((dias.dayofweek == 6).sum()),
                       "feriados": int(sum(1 for x in dias if x.date() in fer and x.dayofweek < 6))})
    return pd.DataFrame(linhas).set_index("data")

idx_total = pd.date_range(y.index.min(), "2027-12-01", freq="MS")   # inclui horizonte futuro
cal = calendario(idx_total)

res = pd.DataFrame({"indice": y, "indice_tratado": y_trat}).reindex(idx_total)
res["imputado"] = res.index.isin(JANELA_COVID)
res = res.join(cal)
res.index.name = "data"
res.to_csv(out)

comp = res.loc[res.imputado, ["indice", "indice_tratado"]].round(1)
print("Meses imputados (original x tratado):")
print(comp.to_string())
