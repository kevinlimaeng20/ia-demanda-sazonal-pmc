# AI for seasonal demand forecasting and scenario simulation – PMC/IBGE (textiles, apparel and footwear)

Reproduction package for the article **"Artificial Intelligence for Seasonal Demand Forecasting and Scenario Simulation: A Rolling-Origin Comparison on Brazilian Textile, Apparel and Footwear Retail"**, by Francisco Kevin Lima Alves (2026).

The experiment compares seasonal naive, Holt-Winters (ETS) and SARIMA with random forest, XGBoost, LightGBM and a multilayer perceptron (MLP) neural network for forecasting the sales volume index of Brazilian textile, apparel and footwear retail. Evaluation uses an expanding-window rolling origin, and probabilistic scenarios come from a Monte Carlo simulation that bootstraps complete historical error trajectories.

> Scripts and file names are in Portuguese; this README describes them in English. *Uma versão em português das instruções está no fim deste arquivo.*

## Data

| File | Content |
|---|---|
| `dados_ibge/pmc_8882_vestuario.xml` | Series extracted from the SIDRA API on **10 October 2026**: table 8882, variable 7169 (index number, 2022 = 100), index type 56734 (sales volume), activity 90673 (textiles, apparel and footwear), Brazil, Jan 2000 to Jul 2026. SHA-256: `ba95629c3b452c978595273097a357677f311fc563ca11376869096a5e5f224a` |
| `dados_ibge/calendario_divulgacao_pmc.csv` | Official PMC release dates (reference months Jan 2022 to Jul 2026), transcribed from the IBGE survey page on 10 October 2026 |
| `saidas/serie_pmc_vestuario.csv` | Clean series (319 months) |

Extraction URL (IBGE may have revised the series after this date):
`https://apisidra.ibge.gov.br/values/t/8882/n1/all/v/7169/p/all/c11046/56734/c85/90673/d/v7169%205`

Release calendar: https://www.ibge.gov.br/estatisticas/economicas/comercio/9227-pesquisa-mensal-de-comercio.html

## Environment

- Python 3.13.16 on Ubuntu 24.04 (x86_64), 2 CPU cores
- Exact versions in `requirements.txt` (`pip install -r requirements.txt`)

## Random seeds

| Component | Seed |
|---|---|
| MLP neural network (average of 5 networks) | 0, 1, 2, 3 and 4 |
| Random forest, XGBoost, LightGBM | 42 |
| Monte Carlo simulation (`numpy.random.default_rng`) | 2026 |
| Block bootstrap of the coverage rates | 7 |

## How to run

Run from inside `codigo/`, in this order:

```bash
python 01_carregar_dados.py ../dados_ibge/pmc_8882_vestuario.xml ../saidas/serie_pmc_vestuario.csv  # load the SIDRA XML
python 02_preparar.py ../saidas/serie_pmc_vestuario.csv ../saidas/serie_preparada.csv            # pandemic-month imputation
python 03_backtest.py ../saidas/previsoes.csv      # rolling-origin backtest, about 6 min with 2 cores
python 04_avaliacao.py                              # metrics, Diebold-Mariano tests and Monte Carlo simulation
python 05_figuras.py                                # figures of the Portuguese version
python 06_figures_en.py                             # figures of the English article (PDF and EPS)
python 07_verificacao_mlp.py                        # checks MLP standardization and seeds
python 08_analises_revisao.py                       # robustness analyses (Online Resource 1)
python 09_sensibilidade_cenarios_finais.py          # sensitivity of the Dec 2026 scenarios to the resampled history
```

`saidas/previsoes.csv` is included, so step 3 can be skipped.

**Reproducibility check:** on 10 October 2026 all steps (1 to 9, including the full backtest) were run from a clean copy of this package. The 9,324 forecasts in `previsoes.csv` and all result tables were reproduced without any difference.

## Where each output appears in the article

| Article | File |
|---|---|
| Table 2 (accuracy) | `saidas/resultados/metricas_validacao.csv`, `metricas_teste.csv`, `saidas/revisao/dm_contagens_holm.csv` |
| Table 3 (peak months) | `saidas/resultados/metricas_teste.csv` |
| Table 4 (December forecast from July) | `saidas/resultados/dezembro_previsto_em_julho.csv` |
| Table 5 (calibration) | `saidas/resultados/simulacao_calibracao_teste.csv`, `simulacao_calibracao_picos_teste.csv` |
| Table 6 (scenarios) | `saidas/resultados/cenarios_finais.csv` |
| Figs. 1 to 5 | `saidas/figures_en/` (English) and `saidas/figuras/` (Portuguese) |
| Online Resource 1, Table S1 (Diebold-Mariano by horizon) | `saidas/revisao/dm_mlp_vs_ets_por_horizonte.csv`, `dm_mlp_vs_ets_37_alvos_comuns.csv` |
| Table S2 (pseudo-real-time) | `saidas/revisao/mape_por_antecedencia_real.csv`, `calendario_divulgacao_com_defasagem.csv` |
| Table S3 (Decembers as events) | `saidas/revisao/dezembros_por_evento.csv` |
| Table S4 (coverage and CRPS by horizon and month) | `saidas/revisao/simulacao_por_horizonte.csv`, `simulacao_por_mes_alvo.csv` |
| Tables S5 and S6 (bootstrap sensitivity) | `saidas/revisao/simulacao_sensibilidade_pool.csv`, `simulacao_ic_coberturas_blocos.csv`, `cenarios_finais_sensibilidade_pool.csv` |
| MLP verification | `saidas/revisao/verificacao_mlp.json` |

Model names in the output files are in Portuguese: *Sazonal ingênuo* = seasonal naive, *Rede Neural MLP* = MLP, *Combinação IA* = ML combination, *Combinação geral* = overall combination.

## Implementation details

- **Pandemic treatment:** Mar–Jul 2020 and Mar–Apr 2021 are imputed for model fitting only, with seasonal factors from a robust STL fitted to pre-pandemic data (`02_preparar.py`).
- **ML models:** trained on `d_t = ln y_t − ln y_{t−12}` with recursive forecasting. For the MLP, the `StandardScaler` is fitted only to the training sample of each origin and reapplied to the forecasting inputs, which `07_verificacao_mlp.py` verifies.
- **Diebold-Mariano:** the loss is the absolute percentage error. The long-run variance uses sample autocovariances (divisor n) up to lag h−1 with a rectangular kernel; if not positive, the lag-zero variance is used. The Harvey–Leybourne–Newbold correction is applied with a t distribution with n−1 degrees of freedom, and Holm adjustment over the 12 horizons.
- **Pseudo-real-time:** forecasts are re-indexed by the official release date of the last observation used. Historical vintages of the series were not reconstructed.

## Use of generative AI

A generative AI assistant (Claude, Anthropic) helped write and debug the code and draft the article; the author reviewed the code and verified the results.

## License

Code under the MIT license (see `LICENSE`). Data are public, from IBGE.

---

### Em português

Pacote de reprodução do artigo citado acima (versão em inglês). Para reproduzir: instale `requirements.txt` e rode os scripts de `codigo/` na ordem indicada em "How to run". `previsoes.csv` já está incluído, então a etapa 3 pode ser pulada. As figuras em inglês ficam em `saidas/figures_en/`, e as em português em `saidas/figuras/`.
