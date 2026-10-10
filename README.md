# IA para simulação de demandas em períodos sazonais – PMC/IBGE (vestuário e calçados)

Pacote de reprodução do artigo **"Utilização de inteligência artificial para simulação de demandas em períodos sazonais: um estudo aplicado ao varejo brasileiro de tecidos, vestuário e calçados"**, de Francisco Kevin Lima Alves (Revista Tópicos, 2026).

O experimento compara sazonal ingênuo, Holt-Winters (ETS) e SARIMA com Random Forest, XGBoost, LightGBM e rede neural MLP na previsão do índice de volume de vendas do varejo de tecidos, vestuário e calçados. A avaliação usa origem móvel, e os cenários probabilísticos vêm de uma simulação de Monte Carlo com bootstrap de trajetórias de erro.

## Dados

| Arquivo | Conteúdo |
|---|---|
| `dados_ibge/pmc_8882_vestuario.xml` | Série extraída da API SIDRA em **10/10/2026**: tabela 8882, variável 7169 (número-índice, 2022 = 100), tipo de índice 56734 (volume de vendas), atividade 90673 (tecidos, vestuário e calçados), Brasil, jan/2000 a jul/2026. SHA-256: `ba95629c3b452c978595273097a357677f311fc563ca11376869096a5e5f224a` |
| `dados_ibge/calendario_divulgacao_pmc.csv` | Datas oficiais de divulgação da PMC (referências jan/2022 a jul/2026), transcritas da página da pesquisa no IBGE e consultadas em 10/10/2026 |
| `saidas/serie_pmc_vestuario.csv` | Série limpa (319 meses) |

URL usada na extração (a série atual pode ter sido revisada pelo IBGE depois desta data):
`https://apisidra.ibge.gov.br/values/t/8882/n1/all/v/7169/p/all/c11046/56734/c85/90673/d/v7169%205`

Calendário de divulgação: https://www.ibge.gov.br/estatisticas/economicas/comercio/9227-pesquisa-mensal-de-comercio.html

## Ambiente

- Python 3.13.16 em Ubuntu 24.04 (x86_64), 2 núcleos de CPU
- Versões exatas em `requirements.txt` (instale com `pip install -r requirements.txt`)

## Sementes aleatórias

| Componente | Semente |
|---|---|
| Rede neural MLP (média de 5 redes) | 0, 1, 2, 3 e 4 |
| Random Forest, XGBoost, LightGBM | 42 |
| Simulação de Monte Carlo (`numpy.random.default_rng`) | 2026 |
| Bootstrap em blocos das coberturas | 7 |

## Execução

Rode de dentro da pasta `codigo/`, nesta ordem:

```bash
python 01_carregar_dados.py ../dados_ibge/pmc_8882_vestuario.xml ../saidas/serie_pmc_vestuario.csv
python 02_preparar.py ../saidas/serie_pmc_vestuario.csv ../saidas/serie_preparada.csv
python 03_backtest.py ../saidas/previsoes.csv      # cerca de 6 min com 2 núcleos
python 04_avaliacao.py                              # métricas, Diebold-Mariano e simulação
python 05_figuras.py                                # figuras do artigo
python 07_verificacao_mlp.py                        # verifica a padronização da MLP e as sementes
python 08_analises_revisao.py                       # análises de robustez (apêndices)
python 09_sensibilidade_cenarios_finais.py          # sensibilidade dos cenários de dez/2026 ao histórico reamostrado
```

`saidas/previsoes.csv` já está incluído, então é possível pular a etapa 3 e rodar diretamente a partir da 4.

**Verificação de reprodutibilidade:** em 10/10/2026, todas as etapas (de 1 a 9, incluindo o backtest completo) foram executadas a partir de uma cópia limpa deste pacote. As 9.324 previsões de `previsoes.csv` e todas as tabelas de resultados foram reproduzidas sem nenhuma diferença.

## O que cada saída alimenta no artigo

| Artigo | Arquivo |
|---|---|
| Tabela 1 (acurácia) | `saidas/resultados/metricas_validacao.csv`, `metricas_teste.csv`, `../revisao/dm_contagens_holm.csv` |
| Tabela 2 (meses de pico) | `saidas/resultados/metricas_teste.csv` |
| Tabela 3 (dezembros com origem em julho) | `saidas/resultados/dezembro_previsto_em_julho.csv` |
| Tabela 4 (calibração) | `saidas/resultados/simulacao_calibracao_teste.csv`, `simulacao_calibracao_picos_teste.csv` |
| Tabela 5 (cenários) | `saidas/resultados/cenarios_finais.csv` |
| Figuras 1 a 5 | `saidas/figuras/` |
| Apêndice A (DM por horizonte) | `saidas/revisao/dm_mlp_vs_ets_por_horizonte.csv`, `dm_mlp_vs_ets_37_alvos_comuns.csv` |
| Apêndice B (pseudo-tempo real) | `saidas/revisao/mape_por_antecedencia_real.csv`, `calendario_divulgacao_com_defasagem.csv` |
| Apêndice C (dezembros por evento) | `saidas/revisao/dezembros_por_evento.csv` |
| Apêndice D (simulação por horizonte e mês) | `saidas/revisao/simulacao_por_horizonte.csv`, `simulacao_por_mes_alvo.csv` |
| Apêndice E (sensibilidade do bootstrap) | `saidas/revisao/simulacao_sensibilidade_pool.csv`, `simulacao_ic_coberturas_blocos.csv`, `cenarios_finais_sensibilidade_pool.csv` |
| Verificação da MLP | `saidas/revisao/verificacao_mlp.json` |

## Detalhes metodológicos implementados

- **Tratamento da pandemia:** mar–jul/2020 e mar–abr/2021 são imputados só para estimação, com fatores sazonais de STL robusta ajustada ao período pré-pandemia (`02_preparar.py`).
- **Modelos de IA:** treinados sobre `d_t = ln y_t − ln y_{t−12}`, com previsão recursiva. Na MLP, o `StandardScaler` é ajustado apenas ao treino de cada origem e reaplicado às entradas de previsão, o que é verificado em `07_verificacao_mlp.py`.
- **Diebold-Mariano:** a perda é o erro percentual absoluto. A variância de longo prazo usa autocovariâncias amostrais (divisor n) até a defasagem h−1, com núcleo retangular; se não for positiva, usa-se a variância de defasagem zero. Aplica-se a correção de Harvey, Leybourne e Newbold, com distribuição t de n−1 graus de liberdade, e o ajuste de Holm sobre os 12 horizontes.
- **Pseudo-tempo real:** as previsões são reindexadas pela data oficial de divulgação do último dado. As versões históricas (vintages) da série não foram reconstruídas.

## Licença

Código sob licença MIT (ver `LICENSE`). Os dados são do IBGE, de acesso público.
