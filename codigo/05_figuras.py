"""Figuras do artigo (PNG 300 dpi, largura útil da página A4 com margens 3/2 cm = 16 cm)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = "../saidas/figuras"
import os; os.makedirs(OUT, exist_ok=True)

# paleta de referência (validada: scripts/validate_palette.js, modo claro) + tinta de texto
AZUL, LARANJA, AQUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, MUDO, GRADE, EIXO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": EIXO, "axes.labelcolor": TINTA2, "xtick.color": MUDO, "ytick.color": MUDO,
    "text.color": TINTA, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "savefig.dpi": 300, "figure.dpi": 100, "legend.frameon": False,
})


def salva(fig, nome):
    fig.savefig(f"{OUT}/{nome}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


serie = pd.read_csv("../saidas/serie_preparada.csv", parse_dates=["data"]).set_index("data")
y = serie["indice"].dropna()
y_trat = serie["indice_tratado"].dropna()
imp = serie[serie.imputado.astype(bool)]

# ---------------------------------------------------------------- Figura 1: série histórica
fig, ax = plt.subplots(figsize=(16 * CM, 7 * CM))
ax.plot(y.index, y.values, color=AZUL, lw=1.2, label="Volume de vendas (original)")
ax.scatter(imp.index, imp.indice_tratado, s=22, color=LARANJA, edgecolor="white", linewidth=1.0, zorder=3,
           label="Valor imputado para estimação (meses de fechamento do comércio)")
dezs = y[y.index.month == 12]
ax.annotate("Picos de dezembro", xy=(dezs.index[13], dezs.iloc[13]), xytext=(pd.Timestamp("2005-06-01"), 238),
            fontsize=8, color=TINTA2, arrowprops=dict(arrowstyle="-", color=MUDO, lw=0.7))
ax.set_ylabel("Índice de volume (2022 = 100)")
ax.set_ylim(0, 270)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=1, handlelength=1.6)
salva(fig, "fig1_serie")

# ---------------------------------------------------------------- Figura 2: índices sazonais
ma = y.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
razao = (y / ma).to_frame("r").assign(ano=y.index.year, mes=y.index.month)
f1 = razao[(razao.ano >= 2010) & (razao.ano <= 2019)].groupby("mes").r.mean() * 100
f2 = razao[(razao.ano >= 2022) & (razao.ano <= 2025)].groupby("mes").r.mean() * 100
pd.DataFrame({"2010-2019": f1, "2022-2025": f2}).to_csv("../saidas/resultados/indices_sazonais.csv")
meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
fig, ax = plt.subplots(figsize=(16 * CM, 6.8 * CM))
x = np.arange(12)
w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, f1.values, width=w, color=AZUL, label="Média 2010–2019 (pré-pandemia)")
b2 = ax.bar(x + w / 2 + 0.01, f2.values, width=w, color=LARANJA, label="Média 2022–2025 (pós-pandemia)")
for k in (4, 11):
    topo = max(f1.iloc[k], f2.iloc[k])
    ax.text(k, topo + 4, f"{f1.iloc[k]:.0f} → {f2.iloc[k]:.0f}", ha="center", va="bottom",
            fontsize=7.5, color=TINTA2)
ax.axhline(100, color=MUDO, lw=0.8, ls=(0, (3, 2)))
ax.text(-0.6, 103, "média anual = 100", fontsize=7.5, color=MUDO, va="bottom")
ax.set_xticks(x, meses)
ax.set_ylabel("Índice sazonal (%)")
ax.set_ylim(0, 215)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=1)
salva(fig, "fig2_sazonalidade")

# ---------------------------------------------------------------- Figura 3: delineamento (linha do tempo)
fig, ax = plt.subplots(figsize=(16 * CM, 5.6 * CM))
T = pd.Timestamp
linhas = [
    ("Janela de treino\n(expansível)", T("2000-01-01"), T("2026-07-31"), "#d9d8d2", None),
    ("Validação\n62 origens", T("2014-01-01"), T("2019-02-28"), AZUL, "alvos até fev/2020"),
    ("Pandemia: meses\ntratados", None, None, LARANJA, None),
    ("Teste\n48 origens", T("2022-07-01"), T("2026-06-30"), AQUA, "alvos ago/2022–jul/2026"),
    ("Simulação final", T("2026-08-01"), T("2027-07-31"), AMARELO, "ago/2026–jul/2027"),
]
for i, (rot, a, b, cor, nota) in enumerate(linhas):
    yy = len(linhas) - 1 - i
    if a is not None:
        ax.barh(yy, (b - a).days, left=mdates.date2num(a), height=0.42, color=cor)
        if nota:
            ax.text(mdates.date2num(b) + 60, yy, nota, va="center", fontsize=7.5, color=TINTA2)
    else:
        for d in imp.index:
            ax.barh(yy, 29, left=mdates.date2num(d), height=0.42, color=cor)
        ax.text(mdates.date2num(T("2021-06-01")) + 60, yy, "mar–jul/2020 e mar–abr/2021",
                va="center", fontsize=7.5, color=TINTA2)
ax.set_yticks(range(len(linhas)), [l[0] for l in linhas][::-1])
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(mdates.date2num(T("1999-09-01")), mdates.date2num(T("2031-06-01")))
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0, labelcolor=TINTA2)
ax.spines["left"].set_visible(False)
salva(fig, "fig3_delineamento")

# ---------------------------------------------------------------- Figura 4: MAPE por horizonte
por_h = pd.read_csv("../saidas/resultados/mape_por_horizonte_teste.csv", index_col="h")
fig, ax = plt.subplots(figsize=(16 * CM, 7.2 * CM))
estilos = [("Sazonal ingênuo", MUDO, (0, (4, 2)), "x"),
           ("Holt-Winters (ETS)", AZUL, "-", "o"),
           ("SARIMA", AQUA, "-", "s"),
           ("Random Forest", AMARELO, "-", "D"),
           ("Rede Neural MLP", LARANJA, "-", "^")]
for nome, cor, ls, mk in estilos:
    ax.plot(por_h.index, por_h[nome], color=cor, ls=ls, marker=mk, ms=4.2, mec="white", mew=0.8, label=nome)
ax.set_xlabel("Horizonte de previsão (meses à frente)")
ax.set_ylabel("MAPE (%)")
ax.set_xticks(range(1, 13))
ax.set_ylim(0, 7)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, handlelength=2.4)
salva(fig, "fig4_mape_horizonte")

# ---------------------------------------------------------------- Figura 5: leque de cenários
cen = pd.read_csv("../saidas/resultados/cenarios_finais.csv", parse_dates=["data_alvo"])
dez25 = y.loc["2025-12-01"]
fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 7.0 * CM), sharey=True)
hist = y["2024-01-01":]
for ax, (nome, cor, rot) in zip(axes, [("Holt-Winters (ETS)", AZUL, "(a) Holt-Winters (ETS)"),
                                       ("Rede Neural MLP", LARANJA, "(b) Rede Neural MLP")]):
    c = cen[cen.modelo == nome].sort_values("data_alvo")
    # liga o último mês observado (jul/2026) ao início do leque, para continuidade visual
    ult = pd.DataFrame({"data_alvo": [hist.index[-1]], "p05": [hist.iloc[-1]], "p10": [hist.iloc[-1]],
                        "p50": [hist.iloc[-1]], "p90": [hist.iloc[-1]], "p95": [hist.iloc[-1]]})
    c2 = pd.concat([ult, c[["data_alvo", "p05", "p10", "p50", "p90", "p95"]]], ignore_index=True)
    ax.plot(hist.index, hist.values, color=TINTA, lw=1.2, label="Observado")
    ax.fill_between(c2.data_alvo, c2.p05, c2.p95, color=cor, alpha=0.12, lw=0, label="Intervalo de 90%")
    ax.fill_between(c2.data_alvo, c2.p10, c2.p90, color=cor, alpha=0.25, lw=0, label="Intervalo de 80%")
    ax.plot(c2.data_alvo, c2.p50, color=cor, lw=1.6, label="Mediana dos cenários")
    d = c[c.data_alvo == "2026-12-01"].iloc[0]
    ax.text(pd.Timestamp("2024-01-15"), 228, f"dez/2026: P50 = {d.p50:.1f}; P90 = {d.p90:.1f}".replace(".", ","),
            fontsize=7.2, color=TINTA2, va="top")
    ax.axhline(dez25, color=MUDO, lw=0.7, ls=(0, (3, 2)))
    ax.set_title(rot, loc="left", color=TINTA)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("jan/%y"))
    ax.grid(axis="x", visible=False)
    ax.set_ylim(40, 235)
axes[0].set_ylabel("Índice de volume (2022 = 100)")
h1 = [Line2D([], [], color=TINTA, lw=1.2), Line2D([], [], color=MUDO, lw=1.6),
      Patch(color=MUDO, alpha=0.45), Patch(color=MUDO, alpha=0.2),
      Line2D([], [], color=MUDO, lw=0.7, ls=(0, (3, 2)))]
fig.legend(h1, ["Observado", "Mediana dos cenários (cor do modelo)", "Intervalo de 80%", "Intervalo de 90%",
                "Nível observado em dez/2025"],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), handlelength=1.8)
fig.subplots_adjust(bottom=0.2, wspace=0.08)
salva(fig, "fig5_cenarios")
print("figuras ok")
