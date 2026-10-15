"""English figures for the JCAES manuscript (Springer Nature artwork guidelines).

Vector PDF and EPS with embedded fonts (Liberation Sans, metrically identical to Arial), lettering
between 7 and 9 pt, widths of 84 mm (single column) or 174 mm (double column), and line styles,
markers and hatching so that every series remains distinguishable in grayscale.
Output: ../saidas/figures_en/Fig1..Fig5 (.pdf and .eps).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = "../saidas/figures_en"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN, YELLOW = "#1f5fae", "#d4561f", "#14875c", "#b88200"
INK, INK2, MUTED, GRID = "#111111", "#3f3e3b", "#6f6e69", "#dedcd5"
MM = 1 / 25.4
SINGLE, DOUBLE = 84 * MM, 174 * MM
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "legend.frameon": False, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})


def tint(hex_color, a):
    """Opaque mix of a color with white (EPS has no transparency)."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(1 - a * (1 - rgb))


def save(fig, name):
    for ext in ("pdf", "eps"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


serie = pd.read_csv("../saidas/serie_preparada.csv", parse_dates=["data"]).set_index("data")
y = serie["indice"].dropna()
imp = serie[serie.imputado.astype(bool)]
T = pd.Timestamp

# ------------------------------------------------------------------ Fig. 1: evaluation design
fig, ax = plt.subplots(figsize=(DOUBLE, 48 * MM))
rows = [
    ("Training window\n(expanding)", T("2000-01-01"), T("2026-07-31"), "#cfcdc6", None, None),
    ("Validation\n(62 origins)", T("2014-01-01"), T("2019-02-28"), BLUE, None, "targets up to Feb 2020"),
    ("Pandemic months\n(imputed for fitting)", None, None, ORANGE, None, "Mar–Jul 2020 and Mar–Apr 2021"),
    ("Test\n(48 origins)", T("2022-07-01"), T("2026-06-30"), GREEN, "////", "targets Aug 2022–Jul 2026"),
    ("Final simulation", T("2026-08-01"), T("2027-07-31"), YELLOW, "....", "Aug 2026–Jul 2027"),
]
for i, (lab, a, b, col, hatch, note) in enumerate(rows):
    yy = len(rows) - 1 - i
    if a is not None:
        ax.barh(yy, (b - a).days, left=mdates.date2num(a), height=0.45, color=col, hatch=hatch,
                edgecolor="white" if hatch else col, lw=0)
        if note:
            ax.text(mdates.date2num(b) + 60, yy, note, va="center", fontsize=7, color=INK2)
    else:
        for d in imp.index:
            ax.barh(yy, 29, left=mdates.date2num(d), height=0.45, color=col)
        ax.text(mdates.date2num(T("2021-06-01")) + 60, yy, note, va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(mdates.date2num(T("1999-09-01")), mdates.date2num(T("2032-06-01")))
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
save(fig, "Fig1")

# ------------------------------------------------------------------ Fig. 2: series
fig, ax = plt.subplots(figsize=(DOUBLE, 58 * MM))
ax.plot(y.index, y.values, color=BLUE, lw=0.9, label="Sales volume index (original series)")
ax.scatter(imp.index, imp.indice_tratado, s=16, marker="o", facecolor="white", edgecolor=ORANGE,
           linewidth=1.1, zorder=3, label="Imputed value used for model fitting (store-closure months)")
dezs = y[y.index.month == 12]
ax.annotate("December peaks", xy=(dezs.index[13], dezs.iloc[13]), xytext=(T("2005-06-01"), 240),
            fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_ylabel("Volume index (2022 = 100)")
ax.set_ylim(0, 270)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, handlelength=1.6)
save(fig, "Fig2")

# ------------------------------------------------------------------ Fig. 3: seasonal indices
f = pd.read_csv("../saidas/resultados/indices_sazonais.csv", index_col=0)
f1, f2 = f["2010-2019"], f["2022-2025"]
months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
fig, ax = plt.subplots(figsize=(SINGLE, 58 * MM))
x = np.arange(12)
w = 0.38
ax.bar(x - w / 2, f1.values, width=w, color=BLUE, label="2010–2019 (pre-pandemic)")
ax.bar(x + w / 2, f2.values, width=w, color="white", edgecolor=ORANGE, hatch="//////", lw=0.8,
       label="2022–2025 (post-pandemic)")
for k in (4, 11):
    top = max(f1.iloc[k], f2.iloc[k])
    ax.text(k, top + 4, f"{f1.iloc[k]:.0f}→{f2.iloc[k]:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.axhline(100, color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.set_xticks(x, months)
ax.set_xlabel("Month")
ax.set_ylabel("Seasonal index (%)")
ax.set_ylim(0, 225)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=1, handlelength=1.4)
save(fig, "Fig3")

# ------------------------------------------------------------------ Fig. 4: MAPE by horizon
ph = pd.read_csv("../saidas/resultados/mape_por_horizonte_teste.csv", index_col="h")
fig, ax = plt.subplots(figsize=(SINGLE, 66 * MM))
styles = [("Sazonal ingênuo", "Seasonal naive", MUTED, (0, (4, 2)), "x"),
          ("Holt-Winters (ETS)", "Holt-Winters (ETS)", BLUE, "-", "o"),
          ("SARIMA", "SARIMA", GREEN, (0, (1, 1)), "s"),
          ("Random Forest", "Random Forest", YELLOW, (0, (5, 1, 1, 1)), "D"),
          ("Rede Neural MLP", "MLP", ORANGE, "-", "^")]
for col, lab, c, ls, mk in styles:
    ax.plot(ph.index, ph[col], color=c, ls=ls, marker=mk, ms=3.6, mfc="white" if mk not in ("x",) else c,
            mew=0.9, lw=1.1, label=lab)
ax.set_xlabel("Forecast horizon (months ahead)")
ax.set_ylabel("MAPE (%)")
ax.set_xticks(range(1, 13))
ax.set_ylim(0, 7)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=3, handlelength=2.6, columnspacing=1.0)
save(fig, "Fig4")

# ------------------------------------------------------------------ Fig. 5: scenario fans
cen = pd.read_csv("../saidas/resultados/cenarios_finais.csv", parse_dates=["data_alvo"])
dec25 = y.loc["2025-12-01"]
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 64 * MM), sharey=True)
hist = y["2024-01-01":]
for ax, (name, col, title) in zip(axes, [("Holt-Winters (ETS)", BLUE, "(a) Holt-Winters (ETS)"),
                                         ("Rede Neural MLP", ORANGE, "(b) MLP")]):
    c = cen[cen.modelo == name].sort_values("data_alvo")
    last = pd.DataFrame({k: [hist.iloc[-1]] for k in ("p05", "p10", "p50", "p90", "p95")})
    last.insert(0, "data_alvo", [hist.index[-1]])
    c2 = pd.concat([last, c[["data_alvo", "p05", "p10", "p50", "p90", "p95"]]], ignore_index=True)
    ax.plot(hist.index, hist.values, color=INK, lw=1.0)
    ax.fill_between(c2.data_alvo, c2.p05, c2.p95, color=tint(col, 0.18), lw=0)
    ax.fill_between(c2.data_alvo, c2.p10, c2.p90, color=tint(col, 0.42), lw=0)
    ax.plot(c2.data_alvo, c2.p50, color=col, lw=1.3, ls=(0, (4, 1.5)))
    d = c[c.data_alvo == "2026-12-01"].iloc[0]
    ax.text(T("2024-01-15"), 228, f"Dec 2026: P50 = {d.p50:.1f}; P90 = {d.p90:.1f}", fontsize=7, color=INK2, va="top")
    ax.axhline(dec25, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_title(title, loc="left", color=INK)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("Jan %Y"))
    ax.grid(axis="x", visible=False)
    ax.set_ylim(40, 235)
axes[0].set_ylabel("Volume index (2022 = 100)")
handles = [Line2D([], [], color=INK, lw=1.0), Line2D([], [], color=MUTED, lw=1.3, ls=(0, (4, 1.5))),
           Patch(color=tint(MUTED, 0.55)), Patch(color=tint(MUTED, 0.25)),
           Line2D([], [], color=MUTED, lw=0.6, ls=(0, (1, 1.5)))]
fig.legend(handles, ["Observed", "Scenario median", "80% interval", "90% interval", "Level observed in Dec 2025"],
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), handlelength=1.8, columnspacing=1.2)
fig.subplots_adjust(bottom=0.2, wspace=0.08)
save(fig, "Fig5")
print("English figures ok")
