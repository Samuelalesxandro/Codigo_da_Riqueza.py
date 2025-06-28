import pandas as pd
import wbdata
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import streamlit as st

# --- CONFIGURAÇÃO DO PROJETO ---
print("--- INICIANDO ANÁLISE: O CÓDIGO DA RIQUEZA ---")

INDICADORES = {
    "NY.GDP.PCAP.KD": "PIB_per_capita",
    "NE.GDI.FTOT.CD": "Formacao_Bruta_Capital",
    "SE.ADT.1524.LT.ZS": "Alfabetizacao_Jovens",
    "SL.TLF.CACT.ZS": "Participacao_Forca_Trabalho",
    "NE.RSB.GNFS.CD": "Balanca_Comercial",
    "IT.NET.USER.ZS": "Cobertura_Internet",
    "NE.EXP.GNFS.CD": "Valor_Exportacoes",
    "NY.GNP.PCAP.CD": "Renda_Nacional_Bruta_per_Capita",
    "EG.ELC.ACCS.ZS": "Acesso_Eletricidade",
    "SI.POV.GINI": "Gini",
    "SL.UEM.TOTL.ZS": "Desemprego",
    "SE.PRM.CMPT.ZS": "Conclusao_Ensino_Primario",
    "NE.CON.PRVT.CD": "Consumo_Familias",
    "NE.CON.GOVT.CD": "Consumo_Governo",
    "SH.H2O.BASW.ZS": "Cobertura_Agua_Potavel"
}

PAISES_SUL_AMERICA = ['BRA', 'ARG', 'CHL', 'COL', 'PER', 'ECU', 'VEN', 'BOL', 'PRY', 'URY']
PAISES_SUDESTE_ASIATICO = ['IDN', 'THA', 'VNM', 'PHL', 'MYS', 'SGP', 'MMR', 'KHM', 'LAO', 'BRN']
TODOS_PAISES = PAISES_SUL_AMERICA + PAISES_SUDESTE_ASIATICO
DATA_INICIO = datetime(2000, 1, 1)
DATA_FIM = datetime(2020, 12, 31)

# --- COLETA DOS DADOS ---
try:
    print("🔄 Coletando dados do Banco Mundial...")
    df_raw = wbdata.get_dataframe(indicators=INDICADORES, country=TODOS_PAISES, date=(DATA_INICIO, DATA_FIM))
    print("✅ Dados coletados com sucesso.")
except Exception as e:
    print(f"❌ Erro ao baixar os dados: {e}")
    exit()

# --- LIMPEZA E ORGANIZAÇÃO ---
df = df_raw.reset_index()
if 'country' not in df.columns and 'country' in df_raw.index.names:
    df['País'] = df_raw.index.get_level_values('country').values
elif 'country' in df.columns:
    df.rename(columns={'country': 'País'}, inplace=True)

if 'date' not in df.columns and 'date' in df_raw.index.names:
    df['Ano'] = df_raw.index.get_level_values('date').values
elif 'date' in df.columns:
    df.rename(columns={'date': 'Ano'}, inplace=True)

df['Ano'] = pd.to_datetime(df['Ano']).dt.year
df.rename(columns=INDICADORES, inplace=True)

print("✅ Colunas disponíveis:", df.columns.tolist())

if 'País' in df.columns and 'Ano' in df.columns:
    df = df.sort_values(by=['País', 'Ano'])
    df = df.groupby('País', group_keys=False).apply(lambda group: group.ffill().bfill()[group.columns])
    df = df.reset_index(drop=True)
    df = df.dropna()
else:
    print("❌ As colunas 'País' e/ou 'Ano' não estão disponíveis.")
    exit()

print("\n📊 Amostra dos dados limpos:")
print(df.head())

# --- ENGENHARIA DE VARIÁVEIS ---
df_model = df.copy().set_index(['País', 'Ano'])
for var in df_model.columns:
    if var != 'PIB_per_capita':
        df_model[f'{var}_lag1'] = df_model.groupby('País')[var].shift(1)
df_model = df_model.dropna()
print(f"\n📈 Tamanho do dataset final para modelagem: {df_model.shape[0]} observações.")

# --- MODELAGEM COM XGBoost ---
print("\n🚀 Treinando modelo XGBoost para prever PIB per capita...")
TARGET = 'PIB_per_capita'
PREDICTORS = [col for col in df_model.columns if '_lag1' in col]
X = df_model[PREDICTORS]
y = df_model[TARGET]
model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
model.fit(X, y)
r2 = r2_score(y, model.predict(X))
print(f"\n📌 Poder de explicação do modelo (R² no treino): {r2:.4f}")
importance = pd.Series(model.feature_importances_, index=PREDICTORS).sort_values(ascending=False)
print("\n🏆 Fatores mais importantes para o crescimento econômico:")
print(importance.head(10))

# --- COMPARAÇÃO ENTRE REGIÕES ---
def treinar_modelo_por_regiao(df_model, paises_iso, nome_regiao):
    paises_info = wbdata.get_countries()
    iso_para_nome = {p['id']: p['name'] for p in paises_info}
    nomes_paises = [iso_para_nome[iso] for iso in paises_iso if iso in iso_para_nome]
    df_regiao = df_model.loc[df_model.index.get_level_values('País').isin(nomes_paises)]
    if df_regiao.empty:
        print(f"⚠️ Nenhum dado encontrado para {nome_regiao}.")
        return pd.Series(dtype=float)
    X = df_regiao[[col for col in df_regiao.columns if '_lag1' in col]]
    y = df_regiao['PIB_per_capita']
    modelo = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    importancia = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n🔹 {nome_regiao} — Top 5 fatores:")
    print(importancia.head(5))
    return importancia

print("\n🌎 Comparando América do Sul vs Sudeste Asiático...")
importancia_sul = treinar_modelo_por_regiao(df_model, PAISES_SUL_AMERICA, "América do Sul")
importancia_asia = treinar_modelo_por_regiao(df_model, PAISES_SUDESTE_ASIATICO, "Sudeste Asiático")
comparacao = pd.concat([
    importancia_sul.rename("América do Sul"),
    importancia_asia.rename("Sudeste Asiático")
], axis=1).fillna(0)

# --- EXPORTAÇÃO DOS DADOS ---
print("\n💾 Exportando dados e resultados...")
df_export = df_model.reset_index()
df_export.to_csv("dados_modelo_completos.csv", index=False)
importance.to_csv("importancia_geral.csv")
comparacao.to_csv("importancia_por_regiao.csv")

for pais in df_export['País'].unique():
    df_export[df_export['País'] == pais].to_csv(f"dados_{pais.replace(' ', '_')}.csv", index=False)

with pd.ExcelWriter("dados_por_pais.xlsx") as writer:
    for pais in df_export['País'].unique():
        df_export[df_export['País'] == pais].to_excel(writer, sheet_name=pais[:31], index=False)

# --- COMPARAÇÃO ENTRE MODELOS ---
print("\n🤖 Comparando diferentes modelos preditivos...")
def avaliar_modelo(nome, modelo, X, y):
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    return {
        "Modelo": nome,
        "R²": r2_score(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred)
    }

modelos = [
    ("Regressão Linear", LinearRegression()),
    ("Ridge", Ridge(alpha=1.0)),
    ("Lasso", Lasso(alpha=0.1)),
    ("Árvore de Decisão", DecisionTreeRegressor(max_depth=5, random_state=42)),
    ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ("XGBoost", XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1))
]

resultados = [avaliar_modelo(nome, modelo, X, y) for nome, modelo in modelos]
df_resultados = pd.DataFrame(resultados)
print("\n📈 Comparação de desempenho dos modelos:")
print(df_resultados.sort_values(by="R²", ascending=False))

# --- PAINEL INTERATIVO COM STREAMLIT ---
print("\n🖥️ Iniciando painel interativo com Streamlit...")

st.set_page_config(page_title="Código da Riqueza", layout="wide")
st.title("📊 O Código da Riqueza — Painel Interativo")

@st.cache_data
def carregar_dados():
    return pd.read_csv("dados_modelo_completos.csv")

df = carregar_dados()

if not df.empty:
    st.sidebar.header("🔎 Filtros")
    paises = sorted(df['País'].unique())
    pais_selecionado = st.sidebar.selectbox("Selecione um país", paises)

    anos_disponiveis = sorted(df[df['País'] == pais_selecionado]['Ano'].unique())
    ano_inicio, ano_fim = st.sidebar.select_slider(
        "Intervalo de anos",
        options=anos_disponiveis,
        value=(anos_disponiveis[0], anos_disponiveis[-1])
    )

    df_filtrado = df[(df['País'] == pais_selecionado) & (df['Ano'].between(ano_inicio, ano_fim))]

    st.subheader(f"📈 Evolução dos indicadores — {pais_selecionado} ({ano_inicio}–{ano_fim})")

    indicadores = [
        'PIB_per_capita', 'Formacao_Bruta_Capital', 'Alfabetizacao_Jovens',
        'Participacao_Forca_Trabalho', 'Balanca_Comercial', 'Cobertura_Internet',
        'Valor_Exportacoes', 'Renda_Nacional_Bruta_per_Capita', 'Acesso_Eletricidade',
        'Gini', 'Desemprego', 'Conclusao_Ensino_Primario', 'Consumo_Familias',
        'Consumo_Governo', 'Cobertura_Agua_Potavel'
    ]

    indicador_escolhido = st.selectbox("Escolha um indicador para visualizar", indicadores)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_filtrado, x="Ano", y=indicador_escolhido, marker="o", ax=ax)
    ax.set_title(f"{indicador_escolhido.replace('_', ' ')} ao longo do tempo", fontsize=14)
    ax.set_ylabel(indicador_escolhido.replace('_', ' '))
    ax.set_xlabel("Ano")
    st.pyplot(fig)

    st.subheader("📋 Dados filtrados")
    st.dataframe(df_filtrado)

    st.download_button(
        label="📥 Baixar dados filtrados como CSV",
        data=df_filtrado.to_csv(index=False).encode('utf-8'),
        file_name=f"{pais_selecionado}_dados_filtrados.csv",
        mime='text/csv'
    )
else:
    st.warning("Nenhum dado disponível para exibir.")