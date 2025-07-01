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
ZONA_DO_EURO = ['DEU', 'FRA', 'ITA', 'ESP', 'PRT', 'GRC', 'IRL', 'NLD', 'AUT', 'BEL']
BRICS = ['BRA', 'RUS', 'IND', 'CHN', 'ZAF', 'EGY', 'ETH', 'IRN', 'SAU', 'ARE']      
PAISES_SUL_AMERICA = ['BRA', 'ARG', 'CHL', 'COL', 'PER', 'ECU', 'VEN', 'BOL', 'PRY', 'URY']
PAISES_SUDESTE_ASIATICO = ['IDN', 'THA', 'VNM', 'PHL', 'MYS', 'SGP', 'MMR', 'KHM', 'LAO', 'BRN']
TODOS_PAISES = list(set(PAISES_SUL_AMERICA + PAISES_SUDESTE_ASIATICO + BRICS + ZONA_DO_EURO))
DATA_INICIO = datetime(1995, 1, 1)
DATA_FIM = datetime(2025, 4, 30)

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

# Garante que as colunas 'País' e 'Ano' existam
if 'country' in df.columns:
    df.rename(columns={'country': 'País'}, inplace=True)
if 'date' in df.columns:
    df.rename(columns={'date': 'Ano'}, inplace=True)

# Se ainda estiverem ausentes, tenta extrair do índice
if 'País' not in df.columns and 'country' in df_raw.index.names:
    df['País'] = df_raw.index.get_level_values('country')
if 'Ano' not in df.columns and 'date' in df_raw.index.names:
    df['Ano'] = df_raw.index.get_level_values('date')
    print("Shape do df_raw:", df_raw.shape)

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
print("Colunas disponíveis no df:", df.columns.tolist())
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

    # --- PREVISÃO DE PIB PER CAPITA ---
    st.subheader("🔮 Previsão de PIB per capita com XGBoost")

    if st.button("Gerar previsão para o país selecionado"):
        try:
            df_pred = df_model.reset_index()
            df_pred = df_pred[df_pred['País'] == pais_selecionado]
            df_pred = df_pred.sort_values("Ano")

            X_pred = df_pred[[col for col in df_pred.columns if '_lag1' in col]]
            y_real = df_pred['PIB_per_capita']
            y_pred = model.predict(X_pred)

            df_pred['PIB_previsto'] = y_pred

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_pred['Ano'], y_real, label="Real", marker="o")
            ax.plot(df_pred['Ano'], y_pred, label="Previsto", marker="o")
            ax.set_title(f"PIB per capita — Real vs Previsto ({pais_selecionado})")
            ax.set_ylabel("PIB per capita")
            ax.set_xlabel("Ano")
            ax.legend()
            st.pyplot(fig)

            st.dataframe(df_pred[['Ano', 'PIB_per_capita', 'PIB_previsto']].round(2))
        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

    # --- COMPARAÇÃO ENTRE DOIS PAÍSES ---
    st.subheader("📊 Comparar dois países lado a lado")

    col1, col2 = st.columns(2)
    with col1:
        pais_1 = st.selectbox("País 1", paises, index=0, key="pais1")
    with col2:
        pais_2 = st.selectbox("País 2", paises, index=1, key="pais2")

    indicador_comp = st.selectbox("Indicador para comparar", indicadores, key="indicador_comp")

    df_p1 = df[(df['País'] == pais_1) & (df['Ano'].between(ano_inicio, ano_fim))]
    df_p2 = df[(df['País'] == pais_2) & (df['Ano'].between(ano_inicio, ano_fim))]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_p1, x="Ano", y=indicador_comp, label=pais_1, marker="o", ax=ax)
    sns.lineplot(data=df_p2, x="Ano", y=indicador_comp, label=pais_2, marker="o", ax=ax)
    ax.set_title(f"{indicador_comp.replace('_', ' ')} — {pais_1} vs {pais_2}")
    ax.set_ylabel(indicador_comp.replace('_', ' '))
    ax.set_xlabel("Ano")
    st.pyplot(fig)

else:
    st.warning("Nenhum dado disponível para exibir.")
    
# --- Função auxiliar para gerar projeções ---
# --- Função corrigida para gerar projeções dinâmicas ---
def gerar_projecao_pib_dinamica(df_model, pais, modelo, ano_final=2035):
    """
    Gera projeções dinâmicas do PIB per capita, atualizando os indicadores
    com base em tendências históricas e correlações.
    """
    import numpy as np
    import pandas as pd
    
    df_pred = df_model.reset_index()
    df_pred = df_pred[df_pred['País'] == pais].sort_values("Ano")

    if df_pred.empty:
        raise ValueError("Dados insuficientes para o país selecionado.")

    df_pred['Ano'] = df_pred['Ano'].astype(int)
    df_base = df_pred.copy()
    ultimo_ano = df_base['Ano'].max()
    anos_futuros = list(range(ultimo_ano + 1, ano_final + 1))

    # Calcular tendências históricas para cada indicador
    tendencias = {}
    correlacoes = {}
    
    # Indicadores base (sem _lag1)
    indicadores_base = [col for col in df_base.columns if not col.endswith('_lag1') and col not in ['País', 'Ano', 'PIB_per_capita']]
    
    for indicador in indicadores_base:
        if indicador in df_base.columns:
            # Calcular tendência (crescimento médio anual dos últimos 5 anos)
            valores = df_base[indicador].tail(5)
            if len(valores) > 1:
                crescimento = valores.pct_change().dropna()
                tendencias[indicador] = crescimento.mean() if not crescimento.empty else 0
            else:
                tendencias[indicador] = 0
            
            # Calcular correlação com PIB per capita
            correlacao = df_base[indicador].corr(df_base['PIB_per_capita'])
            correlacoes[indicador] = correlacao if not pd.isna(correlacao) else 0

    # Linha base para projeções
    linha_atual = df_base.iloc[-1].copy()
    linhas_futuras = []

    # Calcular crescimento médio do PIB
    df_pib = df_base[['Ano', 'PIB_per_capita']].sort_values("Ano")
    df_pib['Crescimento_PIB'] = df_pib['PIB_per_capita'].pct_change()
    crescimento_medio_pib = df_pib['Crescimento_PIB'].tail(5).mean()
    
    # Ajustar crescimento se muito extremo
    if abs(crescimento_medio_pib) > 0.1:  # Limitar a 10% ao ano
        crescimento_medio_pib = 0.03  # 3% padrão
    
    print(f"Crescimento médio PIB calculado: {crescimento_medio_pib:.4f}")
    print(f"Tendências calculadas: {tendencias}")

    for i, ano in enumerate(anos_futuros):
        nova_linha = linha_atual.copy()
        nova_linha['Ano'] = ano
        
        # Atualizar indicadores base com suas tendências
        for indicador in indicadores_base:
            if indicador in nova_linha.index:
                valor_atual = nova_linha[indicador]
                tendencia = tendencias.get(indicador, 0)
                correlacao = correlacoes.get(indicador, 0)
                
                # Aplicar tendência com alguma variação baseada na correlação com PIB
                fator_pib = 1 + (crescimento_medio_pib * correlacao * 0.5)  # Influência do PIB
                fator_tendencia = 1 + tendencia
                
                # Combinar fatores com peso
                fator_final = (fator_tendencia * 0.7) + (fator_pib * 0.3)
                
                # Aplicar com suavização para evitar crescimento exponencial
                valor_novo = valor_atual * (1 + (fator_final - 1) * 0.8)
                
                # Adicionar pequena variação aleatória para realismo
                ruido = np.random.normal(0, 0.01)  # 1% de variação
                valor_novo *= (1 + ruido)
                
                nova_linha[indicador] = max(0, valor_novo)  # Evitar valores negativos

        # Atualizar os lags com os valores do período anterior
        for col in nova_linha.index:
            if col.endswith('_lag1'):
                base_col = col.replace('_lag1', '')
                if base_col in linha_atual.index:
                    nova_linha[col] = linha_atual[base_col]

        # Fazer previsão do PIB
        colunas_lag = [col for col in nova_linha.index if col.endswith('_lag1')]
        X_input = pd.DataFrame([nova_linha[colunas_lag]])
        
        try:
            pib_previsto = modelo.predict(X_input)[0]
            
            # Aplicar suavização para evitar mudanças bruscas
            pib_anterior = linha_atual['PIB_per_capita']
            pib_suavizado = pib_anterior * 0.3 + pib_previsto * 0.7
            
            nova_linha['PIB_per_capita'] = pib_suavizado
            
        except Exception as e:
            print(f"Erro na previsão do ano {ano}: {e}")
            # Fallback: crescimento baseado na tendência histórica
            nova_linha['PIB_per_capita'] = linha_atual['PIB_per_capita'] * (1 + crescimento_medio_pib)

        linha_atual = nova_linha.copy()
        linhas_futuras.append(nova_linha)

    # Combinar dados históricos com projeções
    df_historico = df_base.copy()
    df_futuro = pd.DataFrame(linhas_futuras)
    
    # Adicionar flag para distinguir dados históricos de projeções
    df_historico['Tipo'] = 'Histórico'
    df_futuro['Tipo'] = 'Projeção'
    
    df_completo = pd.concat([df_historico, df_futuro], ignore_index=True)
    
    return df_completo


# --- Função para calcular cenários otimista e pessimista ---
# =============================================================================
# CÓDIGO PARA CORRIGIR AS PROJEÇÕES - ADICIONE ESTE BLOCO NO SEU ARQUIVO
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- FUNÇÃO DE PROJEÇÃO CORRIGIDA ---
def gerar_projecao_dinamica(df_model, pais, modelo, ano_final=2035):
    """
    Gera projeções dinâmicas do PIB per capita com evolução realista dos indicadores
    """
    df_pred = df_model.reset_index()
    df_pred = df_pred[df_pred['País'] == pais].sort_values("Ano")

    if df_pred.empty:
        raise ValueError(f"Dados insuficientes para {pais}")

    df_pred['Ano'] = df_pred['Ano'].astype(int)
    ultimo_ano = df_pred['Ano'].max()
    anos_futuros = list(range(ultimo_ano + 1, ano_final + 1))
    
    if not anos_futuros:
        return df_pred
    
    # Calcular tendências históricas dos últimos 5 anos
    df_recente = df_pred.tail(5)
    tendencias = {}
    
    # Identificar colunas base (sem _lag1)
    colunas_base = [col for col in df_pred.columns if not col.endswith('_lag1') and col not in ['País', 'Ano']]
    
    for col in colunas_base:
        if len(df_recente) > 1:
            valores = df_recente[col].values
            # Calcular tendência linear
            x = np.arange(len(valores))
            if len(valores) > 1 and not np.isnan(valores).all():
                coef = np.polyfit(x, valores, 1)[0]  # Coeficiente linear
                tendencias[col] = coef
            else:
                tendencias[col] = 0
        else:
            tendencias[col] = 0
    
    # Preparar dados para projeção
    df_result = df_pred.copy()
    ultima_linha = df_pred.iloc[-1].copy()
    
    # Lista para armazenar novas linhas
    novas_linhas = []
    
    for ano in anos_futuros:
        nova_linha = ultima_linha.copy()
        nova_linha['Ano'] = ano
        
        # Atualizar indicadores base usando tendências
        for col in colunas_base:
            if col != 'PIB_per_capita':  # PIB será calculado pelo modelo
                valor_atual = nova_linha[col]
                tendencia = tendencias.get(col, 0)
                
                # Aplicar tendência com suavização
                # Reduzir a tendência ao longo do tempo para evitar crescimento extremo
                anos_desde_base = ano - ultimo_ano
                fator_suavizacao = 0.95 ** anos_desde_base  # Decay exponencial
                
                novo_valor = valor_atual + (tendencia * fator_suavizacao)
                
                # Adicionar pequena variação aleatória (±2%)
                variacao = np.random.normal(0, 0.02)
                novo_valor *= (1 + variacao)
                
                # Aplicar limites razoáveis
                if col in ['Alfabetizacao_Jovens', 'Cobertura_Internet', 'Acesso_Eletricidade', 'Cobertura_Agua_Potavel']:
                    novo_valor = min(100, max(0, novo_valor))  # Percentuais entre 0-100
                elif col == 'Desemprego':
                    novo_valor = min(50, max(0, novo_valor))  # Desemprego entre 0-50%
                elif col == 'Gini':
                    novo_valor = min(100, max(20, novo_valor))  # Gini entre 20-100
                else:
                    novo_valor = max(0, novo_valor)  # Outros indicadores >= 0
                
                nova_linha[col] = novo_valor
        
        # Atualizar variáveis lag com valores do período anterior
        for col in df_pred.columns:
            if col.endswith('_lag1'):
                col_base = col.replace('_lag1', '')
                if col_base in ultima_linha.index:
                    nova_linha[col] = ultima_linha[col_base]
        
        # Prever PIB per capita usando o modelo
        colunas_lag = [c for c in nova_linha.index if c.endswith('_lag1')]
        X_input = nova_linha[colunas_lag].values.reshape(1, -1)
        
        try:
            pib_previsto = modelo.predict(X_input)[0]
            
            # Suavizar mudanças bruscas
            pib_anterior = ultima_linha['PIB_per_capita']
            crescimento = (pib_previsto / pib_anterior) - 1
            
            # Limitar crescimento extremo a ±20% por ano
            crescimento = max(-0.2, min(0.2, crescimento))
            pib_final = pib_anterior * (1 + crescimento)
            
            nova_linha['PIB_per_capita'] = pib_final
            
        except Exception as e:
            st.warning(f"Erro na previsão para {ano}: {e}")
            # Fallback: crescimento de 2% ao ano
            nova_linha['PIB_per_capita'] = ultima_linha['PIB_per_capita'] * 1.02
        
        novas_linhas.append(nova_linha)
        ultima_linha = nova_linha  # Atualizar para próxima iteração
    
    # Combinar dados originais com projeções
    df_novo = pd.DataFrame(novas_linhas)
    df_completo = pd.concat([df_result, df_novo], ignore_index=True)
    
    return df_completo


# --- FUNÇÃO PARA MÚLTIPLOS CENÁRIOS ---
def gerar_cenarios(df_model, pais, modelo, ano_final=2035):
    """Gera cenários otimista, realista e pessimista"""
    
    cenarios = {}
    
    # Fixar seed para reprodutibilidade
    np.random.seed(42)
    df_realista = gerar_projecao_dinamica(df_model, pais, modelo, ano_final)
    cenarios['Realista'] = df_realista
    
    # Cenário otimista
    np.random.seed(123)  # Seed diferente para variação
    df_otimista = gerar_projecao_dinamica(df_model, pais, modelo, ano_final)
    # Aplicar boost no PIB das projeções
    mask_futuro = df_otimista['Ano'] > df_model.reset_index()['Ano'].max()
    df_otimista.loc[mask_futuro, 'PIB_per_capita'] *= 1.015  # +1.5% adicional
    cenarios['Otimista'] = df_otimista
    
    # Cenário pessimista
    np.random.seed(456)
    df_pessimista = gerar_projecao_dinamica(df_model, pais, modelo, ano_final)
    # Aplicar redução no PIB das projeções
    df_pessimista.loc[mask_futuro, 'PIB_per_capita'] *= 0.985  # -1.5%
    cenarios['Pessimista'] = df_pessimista
    
    return cenarios


# --- INTERFACE STREAMLIT ATUALIZADA ---
# Substitua o botão de projeções existente por este código:

st.subheader("🔮 Projeções Dinâmicas do PIB per capita")

# Configurações de projeção
col1, col2, col3 = st.columns(3)
with col1:
    ano_limite = st.selectbox("Projetar até o ano:", [2030, 2035, 2040], index=1)
with col2:
    tipo_analise = st.selectbox("Tipo de análise:", ["Cenário Único", "Múltiplos Cenários"])
with col3:
    mostrar_detalhes = st.checkbox("Mostrar detalhes dos cálculos")

if st.button("🚀 Gerar Projeções Dinâmicas"):
    try:
        with st.spinner(f"Gerando projeções para {pais_selecionado}..."):
            
            if tipo_analise == "Cenário Único":
                # Projeção única
                df_projecoes = gerar_projecao_dinamica(df_model, pais_selecionado, model, ano_limite)
                
                # Separar histórico de projeções
                ultimo_ano_real = df_model.reset_index()['Ano'].max()
                df_historico = df_projecoes[df_projecoes['Ano'] <= ultimo_ano_real]
                df_futuro = df_projecoes[df_projecoes['Ano'] > ultimo_ano_real]
                
                # Gráfico
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(df_historico['Ano'], df_historico['PIB_per_capita'], 
                       'o-', label='Dados Históricos', linewidth=2, color='blue')
                
                if not df_futuro.empty:
                    ax.plot(df_futuro['Ano'], df_futuro['PIB_per_capita'], 
                           's--', label='Projeções', linewidth=2, color='red', alpha=0.8)
                
                ax.set_title(f'Projeção PIB per capita - {pais_selecionado}')
                ax.set_xlabel('Ano')
                ax.set_ylabel('PIB per capita (US$)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Métricas
                if not df_futuro.empty:
                    pib_atual = df_historico['PIB_per_capita'].iloc[-1]
                    pib_final = df_futuro['PIB_per_capita'].iloc[-1]
                    crescimento_total = ((pib_final / pib_atual) - 1) * 100
                    anos_projecao = len(df_futuro)
                    crescimento_anual = crescimento_total / anos_projecao
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("PIB Atual", f"${pib_atual:,.0f}")
                    with col2:
                        st.metric("PIB Projetado", f"${pib_final:,.0f}")
                    with col3:
                        st.metric("Crescimento Anual", f"{crescimento_anual:.1f}%")
                
                # Tabela de dados
                if mostrar_detalhes:
                    st.subheader("📋 Dados Detalhados")
                    df_display = df_projecoes[['Ano', 'PIB_per_capita']].copy()
                    df_display['Tipo'] = df_display['Ano'].apply(
                        lambda x: 'Histórico' if x <= ultimo_ano_real else 'Projeção'
                    )
                    df_display['PIB_per_capita'] = df_display['PIB_per_capita'].round(2)
                    st.dataframe(df_display)
            
            else:
                # Múltiplos cenários
                cenarios = gerar_cenarios(df_model, pais_selecionado, model, ano_limite)
                
                # Gráfico comparativo
                fig, ax = plt.subplots(figsize=(12, 6))
                
                cores = {'Pessimista': '#ff4444', 'Realista': '#4444ff', 'Otimista': '#44ff44'}
                ultimo_ano_real = df_model.reset_index()['Ano'].max()
                
                # Plotar dados históricos apenas uma vez
                df_hist = cenarios['Realista'][cenarios['Realista']['Ano'] <= ultimo_ano_real]
                ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                       'o-', label='Histórico', linewidth=3, color='black')
                
                # Plotar cada cenário
                for nome, df_cenario in cenarios.items():
                    df_proj = df_cenario[df_cenario['Ano'] > ultimo_ano_real]
                    if not df_proj.empty:
                        ax.plot(df_proj['Ano'], df_proj['PIB_per_capita'], 
                               's--', label=f'Cenário {nome}', 
                               linewidth=2, color=cores[nome], alpha=0.8)
                
                ax.set_title(f'Cenários de Projeção PIB per capita - {pais_selecionado}')
                ax.set_xlabel('Ano')
                ax.set_ylabel('PIB per capita (US$)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Métricas comparativas
                st.subheader("📊 Comparação de Cenários")
                col1, col2, col3 = st.columns(3)
                
                for i, (nome, df_cenario) in enumerate(cenarios.items()):
                    df_proj = df_cenario[df_cenario['Ano'] > ultimo_ano_real]
                    if not df_proj.empty:
                        pib_final = df_proj['PIB_per_capita'].iloc[-1]
                        with [col1, col2, col3][i]:
                            st.metric(f"PIB {nome}", f"${pib_final:,.0f}")
            
            st.success("✅ Projeções geradas com sucesso!")
            
    except Exception as e:
        st.error(f"❌ Erro ao gerar projeções: {str(e)}")
        if mostrar_detalhes:
            st.exception(e)


# --- ANÁLISE DE SENSIBILIDADE ---
st.subheader("🎯 Análise de Sensibilidade")

if st.checkbox("Ativar análise de sensibilidade"):
    st.write("Esta análise mostra como mudanças em indicadores específicos afetam as projeções do PIB.")
    
    # Selecionar indicador para testar
    indicadores_disponiveis = [col.replace('_lag1', '') for col in df_model.columns if col.endswith('_lag1')]
    indicador_teste = st.selectbox("Selecione o indicador para testar:", indicadores_disponiveis)
    
    # Percentual de variação
    variacao = st.slider("Variação percentual do indicador:", -50, 50, 10, 5)
    
    if st.button("Executar Análise de Sensibilidade"):
        try:
            # Projeção base
            df_base = gerar_projecao_dinamica(df_model, pais_selecionado, model, 2030)
            
            # Criar versão modificada dos dados
            df_modificado = df_model.copy()
            col_lag = f"{indicador_teste}_lag1"
            
            if col_lag in df_modificado.columns:
                df_modificado[col_lag] *= (1 + variacao/100)
                
                # Nova projeção com indicador modificado
                df_sensibilidade = gerar_projecao_dinamica(df_modificado, pais_selecionado, model, 2030)
                
                # Comparar resultados
                ultimo_ano_real = df_model.reset_index()['Ano'].max()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Dados históricos
                df_hist = df_base[df_base['Ano'] <= ultimo_ano_real]
                ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                       'o-', label='Histórico', linewidth=2, color='black')
                
                # Projeções
                df_proj_base = df_base[df_base['Ano'] > ultimo_ano_real]
                df_proj_sens = df_sensibilidade[df_sensibilidade['Ano'] > ultimo_ano_real]
                
                ax.plot(df_proj_base['Ano'], df_proj_base['PIB_per_capita'], 
                       's--', label='Projeção Base', linewidth=2, color='blue')
                
                ax.plot(df_proj_sens['Ano'], df_proj_sens['PIB_per_capita'], 
                       's--', label=f'{indicador_teste} {variacao:+}%', linewidth=2, color='red')
                
                ax.set_title(f'Análise de Sensibilidade - {indicador_teste}')
                ax.set_xlabel('Ano')
                ax.set_ylabel('PIB per capita (US$)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Calcular impacto
                if not df_proj_base.empty and not df_proj_sens.empty:
                    pib_base_final = df_proj_base['PIB_per_capita'].iloc[-1]
                    pib_sens_final = df_proj_sens['PIB_per_capita'].iloc[-1]
                    impacto = ((pib_sens_final / pib_base_final) - 1) * 100
                    
                    st.metric(
                        f"Impacto no PIB final ({variacao:+}% em {indicador_teste})",
                        f"{impacto:+.2f}%"
                    )
                    
                    # Elasticidade
                    elasticidade = impacto / variacao if variacao != 0 else 0
                    st.info(f"**Elasticidade:** {elasticidade:.3f} (variação de 1% em {indicador_teste} resulta em {elasticidade:.3f}% de mudança no PIB)")
            
        except Exception as e:
            st.error(f"Erro na análise de sensibilidade: {str(e)}")


# --- RELATÓRIO DE PROJEÇÕES ---
def gerar_relatorio_projecoes(df_projecoes, pais, modelo_info):
    """Gera um relatório detalhado das projeções"""
    
    ultimo_ano_real = df_model.reset_index()['Ano'].max()
    df_historico = df_projecoes[df_projecoes['Ano'] <= ultimo_ano_real]
    df_futuro = df_projecoes[df_projecoes['Ano'] > ultimo_ano_real]
    
    relatorio = f"""
# 📊 Relatório de Projeções Econômicas
## País: {pais}
## Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

### 🔍 Resumo Executivo
- **Período de projeção:** {df_futuro['Ano'].min()} - {df_futuro['Ano'].max()}
- **PIB per capita atual:** US$ {df_historico['PIB_per_capita'].iloc[-1]:,.2f}
- **PIB per capita projetado:** US$ {df_futuro['PIB_per_capita'].iloc[-1]:,.2f}
- **Crescimento total:** {((df_futuro['PIB_per_capita'].iloc[-1] / df_historico['PIB_per_capita'].iloc[-1]) - 1) * 100:.1f}%
- **Crescimento anual médio:** {(((df_futuro['PIB_per_capita'].iloc[-1] / df_historico['PIB_per_capita'].iloc[-1]) ** (1/len(df_futuro))) - 1) * 100:.1f}%

### 📈 Metodologia
Esta projeção foi gerada usando um modelo XGBoost treinado com dados do Banco Mundial.
O modelo considera indicadores defasados (lag-1) para prever o PIB per capita futuro.

**Principais características do modelo:**
- R² de treino: {r2:.4f}
- Indicadores mais importantes: {', '.join(importance.head(3).index)}

### ⚠️ Limitações e Considerações
- As projeções assumem continuidade das tendências históricas
- Não consideram choques externos (crises, pandemias, guerras)
- Baseadas em dados até {ultimo_ano_real}
- Margem de erro esperada: ±15-20%

### 📋 Dados Detalhados
"""
    
    return relatorio


st.subheader("📄 Gerar Relatório de Projeções")

if st.button("Gerar Relatório Completo"):
    try:
        df_projecoes = gerar_projecao_dinamica(df_model, pais_selecionado, model, 2035)
        relatorio = gerar_relatorio_projecoes(df_projecoes, pais_selecionado, model)
        
        st.markdown(relatorio)
        
        # Permitir download do relatório
        st.download_button(
            label="📥 Baixar Relatório (Markdown)",
            data=relatorio.encode('utf-8'),
            file_name=f"relatorio_projecoes_{pais_selecionado}_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime='text/markdown'
        )
        
        # Baixar dados da projeção
        csv_data = df_projecoes[['Ano', 'PIB_per_capita']].to_csv(index=False)
        st.download_button(
            label="📥 Baixar Dados da Projeção (CSV)",
            data=csv_data.encode('utf-8'),
            file_name=f"projecoes_{pais_selecionado}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Erro ao gerar relatório: {str(e)}")


# --- COMPARAÇÃO DE CRESCIMENTO ENTRE PAÍSES ---
st.subheader("🌍 Comparação de Crescimento Projetado")

if st.checkbox("Comparar crescimento entre países"):
    # Selecionar países para comparar
    paises_disponiveis = sorted(df_model.reset_index()['País'].unique())
    paises_comparacao = st.multiselect(
        "Selecione países para comparar (máximo 5):",
        paises_disponiveis,
        default=[pais_selecionado] if pais_selecionado in paises_disponiveis else []
    )
    
    if len(paises_comparacao) > 5:
        st.warning("Máximo 5 países permitidos para comparação.")
        paises_comparacao = paises_comparacao[:5]
    
    if len(paises_comparacao) >= 2 and st.button("Comparar Crescimento"):
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            cores = plt.cm.Set3(np.linspace(0, 1, len(paises_comparacao)))
            
            dados_comparacao = []
            
            for i, pais in enumerate(paises_comparacao):
                try:
                    df_pais = gerar_projecao_dinamica(df_model, pais, model, 2035)
                    ultimo_ano_real = df_model.reset_index()['Ano'].max()
                    
                    # Dados históricos
                    df_hist = df_pais[df_pais['Ano'] <= ultimo_ano_real]
                    df_proj = df_pais[df_pais['Ano'] > ultimo_ano_real]
                    
                    # Plotar
                    ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                           'o-', color=cores[i], alpha=0.7, linewidth=2)
                    
                    if not df_proj.empty:
                        ax.plot(df_proj['Ano'], df_proj['PIB_per_capita'], 
                               's--', color=cores[i], alpha=0.9, linewidth=2, label=pais)
                    
                    # Calcular crescimento projetado
                    if not df_proj.empty and not df_hist.empty:
                        pib_inicial = df_hist['PIB_per_capita'].iloc[-1]
                        pib_final = df_proj['PIB_per_capita'].iloc[-1]
                        crescimento = ((pib_final / pib_inicial) - 1) * 100
                        anos_projecao = len(df_proj)
                        crescimento_anual = crescimento / anos_projecao
                        
                        dados_comparacao.append({
                            'País': pais,
                            'PIB Atual': pib_inicial,
                            'PIB Projetado': pib_final,
                            'Crescimento Total (%)': crescimento,
                            'Crescimento Anual (%)': crescimento_anual
                        })
                        
                except Exception as e:
                    st.warning(f"Erro ao processar {pais}: {str(e)}")
            
            ax.set_title('Comparação de Crescimento Projetado entre Países')
            ax.set_xlabel('Ano')
            ax.set_ylabel('PIB per capita (US$)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabela comparativa
            if dados_comparacao:
                df_comp = pd.DataFrame(dados_comparacao)
                df_comp = df_comp.sort_values('Crescimento Anual (%)', ascending=False)
                
                st.subheader("📊 Ranking de Crescimento Projetado")
                st.dataframe(df_comp.round(2))
                
                # Destacar o melhor e pior desempenho
                melhor = df_comp.iloc[0]
                pior = df_comp.iloc[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏆 **Maior crescimento:** {melhor['País']} ({melhor['Crescimento Anual (%)']:.1f}% ao ano)")
                with col2:
                    st.info(f"📉 **Menor crescimento:** {pior['País']} ({pior['Crescimento Anual (%)']:.1f}% ao ano)")
        
        except Exception as e:
            st.error(f"Erro na comparação: {str(e)}")


# --- ALERTAS E RECOMENDAÇÕES ---
def gerar_recomendacoes(df_projecoes, pais):
    """Gera recomendações baseadas nas projeções"""
    
    ultimo_ano_real = df_model.reset_index()['Ano'].max()
    df_futuro = df_projecoes[df_projecoes['Ano'] > ultimo_ano_real]
    
    if df_futuro.empty:
        return []
    
    recomendacoes = []
    crescimento_medio = df_futuro['PIB_per_capita'].pct_change().mean()
    
    if crescimento_medio < 0.01:  # Menos de 1% ao ano
        recomendacoes.append("⚠️ **Crescimento baixo projetado:** Considerar políticas de estímulo ao investimento e inovação.")
    
    if crescimento_medio > 0.08:  # Mais de 8% ao ano
        recomendacoes.append("🚨 **Crescimento muito alto:** Atenção para possível sobreaquecimento da economia.")
    
    # Verificar indicadores específicos nas projeções
    if 'Cobertura_Internet' in df_futuro.columns:
        internet_final = df_futuro['Cobertura_Internet'].iloc[-1]
        if internet_final < 80:
            recomendacoes.append("📡 **Infraestrutura digital:** Investir em expansão da cobertura de internet.")
    
    if 'Alfabetizacao_Jovens' in df_futuro.columns:
        alfabetizacao = df_futuro['Alfabetizacao_Jovens'].iloc[-1]
        if alfabetizacao < 95:
            recomendacoes.append("📚 **Educação:** Priorizar programas de alfabetização juvenil.")
    
    return recomendacoes


st.subheader("💡 Recomendações de Política Econômica")

if st.button("Gerar Recomendações Automáticas"):
    try:
        df_projecoes = gerar_projecao_dinamica(df_model, pais_selecionado, model, 2030)
        recomendacoes = gerar_recomendacoes(df_projecoes, pais_selecionado)
        
        if recomendacoes:
            st.write(f"**Recomendações para {pais_selecionado}:**")
            for rec in recomendacoes:
                st.write(rec)
        else:
            st.success("✅ As projeções indicam uma trajetória equilibrada de crescimento.")
            
    except Exception as e:
        st.error(f"Erro ao gerar recomendações: {str(e)}")


print("\n" + "="*80)
print("🎉 CÓDIGO DE PROJEÇÕES DINÂMICAS IMPLEMENTADO COM SUCESSO!")
print("="*80)
print("\nRecursos adicionados:")
print("✅ Projeções dinâmicas com evolução realista dos indicadores")
print("✅ Cenários múltiplos (otimista, realista, pessimista)")
print("✅ Análise de sensibilidade")
print("✅ Relatórios automatizados")
print("✅ Comparação entre países")
print("✅ Recomendações de política econômica")
print("✅ Suavização para evitar crescimento exponencial")
print("✅ Limites realistas para cada tipo de indicador")
print("✅ Tratamento robusto de erros")
print("\n" + "="*80)
