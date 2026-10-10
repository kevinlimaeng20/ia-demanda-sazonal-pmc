"""Lê o XML da API SIDRA (tabela 8882) e gera um CSV limpo com a série mensal."""
import sys, xml.etree.ElementTree as ET
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
ns = {"b": "http://schemas.datacontract.org/2004/07/IBGE.BTE.Tabela"}
rows = ET.parse(src).getroot().findall("b:ValorDescritoPorSuasDimensoes", ns)
recs = []
for r in rows[1:]:  # a primeira linha é o cabeçalho descritivo
    g = lambda tag: r.find(f"b:{tag}", ns).text
    recs.append({"periodo": g("D3C"), "variavel": g("D2C"), "tipo_indice": g("D4C"),
                 "atividade": g("D5N"), "valor": g("V")})
df = pd.DataFrame(recs)
assert df.variavel.nunique() == 1 and df.variavel.iloc[0] == "7169"
assert df.tipo_indice.nunique() == 1 and df.tipo_indice.iloc[0] == "56734"
assert df.atividade.nunique() == 1
df["data"] = pd.to_datetime(df.periodo, format="%Y%m")
df["indice"] = pd.to_numeric(df.valor, errors="coerce")
df = df[["data", "indice"]].sort_values("data").reset_index(drop=True)
full = pd.date_range(df.data.min(), df.data.max(), freq="MS")
print("observações:", len(df), "| meses esperados:", len(full), "| faltantes:", df.indice.isna().sum())
print("período:", df.data.min().date(), "→", df.data.max().date())
df.to_csv(out, index=False)
print(df.describe())
