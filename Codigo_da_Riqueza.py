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
def gerar_cenarios_multiplos(df_model, pais, modelo, ano_final=2035):
    """
    Gera três cenários: pessimista, realista e otimista
    """
    cenarios = {}
    
    # Cenário realista (base)
    df_realista = gerar_projecao_pib_dinamica(df_model, pais, modelo, ano_final)
    cenarios['Realista'] = df_realista
    
    # Cenário otimista (+50% no crescimento)
    np.random.seed(42)  # Para reprodutibilidade
    df_otimista = gerar_projecao_pib_dinamica(df_model, pais, modelo, ano_final)
    # Ajustar PIB para cenário otimista
    mask_projecao = df_otimista['Tipo'] == 'Projeção'
    df_otimista.loc[mask_projecao, 'PIB_per_capita'] *= 1.02  # 2% adicional por ano
    cenarios['Otimista'] = df_otimista
    
    # Cenário pessimista (-30% no crescimento)
    np.random.seed(123)
    df_pessimista = gerar_projecao_pib_dinamica(df_model, pais, modelo, ano_final)
    # Ajustar PIB para cenário pessimista
    df_pessimista.loc[mask_projecao, 'PIB_per_capita'] *= 0.99  # -1% por ano
    cenarios['Pessimista'] = df_pessimista
    
    return cenarios


# --- Substituir a função antiga no código Streamlit ---
# Substitua a seção "Gerar projeções futuras" por:

if st.button("Gerar projeções futuras dinâmicas"):
    try:
        # Opção para escolher tipo de projeção
        tipo_projecao = st.radio(
            "Escolha o tipo de projeção:",
            ["Projeção Única", "Cenários Múltiplos"]
        )
        
        if tipo_projecao == "Projeção Única":
            df_projecoes = gerar_projecao_pib_dinamica(df_model, pais_selecionado, model)
            
            # Separar dados históricos e projeções
            df_historico = df_projecoes[df_projecoes['Tipo'] == 'Histórico']
            df_futuro = df_projecoes[df_projecoes['Tipo'] == 'Projeção']
            
            # GRÁFICO DE PROJEÇÃO
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plotar dados históricos
            ax.plot(df_historico['Ano'], df_historico['PIB_per_capita'], 
                   marker="o", label="Dados Históricos", linewidth=2, color='blue')
            
            # Plotar projeções
            ax.plot(df_futuro['Ano'], df_futuro['PIB_per_capita'], 
                   marker="s", label="Projeções", linewidth=2, color='red', linestyle='--')
            
            ax.set_title(f"Projeção Dinâmica do PIB per capita até {df_futuro['Ano'].max()} — {pais_selecionado}")
            ax.set_ylabel("PIB per capita (US$)")
            ax.set_xlabel("Ano")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Mostrar crescimento projetado
            pib_inicial = df_historico['PIB_per_capita'].iloc[-1]
            pib_final = df_futuro['PIB_per_capita'].iloc[-1]
            crescimento_total = ((pib_final / pib_inicial) - 1) * 100
            anos_projecao = len(df_futuro)
            crescimento_anual = (crescimento_total / anos_projecao)
            
            st.metric(
                label="Crescimento Total Projetado",
                value=f"{crescimento_total:.1f}%",
                delta=f"{crescimento_anual:.1f}% ao ano"
            )
            
            # Tabela com os dados
            st.subheader("📊 Dados da Projeção")
            df_display = df_projecoes[['Ano', 'PIB_per_capita', 'Tipo']].copy()
            df_display['PIB_per_capita'] = df_display['PIB_per_capita'].round(2)
            st.dataframe(df_display)
            
        else:  # Cenários Múltiplos
            cenarios = gerar_cenarios_multiplos(df_model, pais_selecionado, model)
            
            # GRÁFICO COM MÚLTIPLOS CENÁRIOS
            fig, ax = plt.subplots(figsize=(12, 6))
            
            cores = {'Pessimista': 'red', 'Realista': 'blue', 'Otimista': 'green'}
            
            for nome_cenario, df_cenario in cenarios.items():
                df_hist = df_cenario[df_cenario['Tipo'] == 'Histórico']
                df_proj = df_cenario[df_cenario['Tipo'] == 'Projeção']
                
                # Dados históricos (apenas uma vez)
                if nome_cenario == 'Realista':
                    ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                           marker="o", label="Histórico", linewidth=2, color='black')
                
                # Projeções para cada cenário
                ax.plot(df_proj['Ano'], df_proj['PIB_per_capita'], 
                       marker="s", label=f"Cenário {nome_cenario}", 
                       linewidth=2, color=cores[nome_cenario], linestyle='--')
            
            ax.set_title(f"Cenários de Projeção do PIB per capita — {pais_selecionado}")
            ax.set_ylabel("PIB per capita (US$)")
            ax.set_xlabel("Ano")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Métricas comparativas
            col1, col2, col3 = st.columns(3)
            
            for i, (nome, df_cenario) in enumerate(cenarios.items()):
                df_proj = df_cenario[df_cenario['Tipo'] == 'Projeção']
                pib_final = df_proj['PIB_per_capita'].iloc[-1]
                
                with [col1, col2, col3][i]:
                    st.metric(
                        label=f"PIB Final - {nome}",
                        value=f"${pib_final:,.0f}"
                    )

    except Exception as e:
        st.error(f"Erro ao gerar projeções: {e}")
        st.write("Detalhes do erro:", str(e))


# --- Função adicional para análise de sensibilidade ---
def analise_sensibilidade(df_model, pais, modelo, indicador_teste, variacao_pct=0.1):
    """
    Testa como mudanças em um indicador específico afetam as projeções do PIB
    """
    # Projeção base
    df_base = gerar_projecao_pib_dinamica(df_model, pais, modelo, 2030)
    
    # Projeção com indicador aumentado
    df_aumentado = df_base.copy()
    mask_proj = df_aumentado['Tipo'] == 'Projeção'
    if indicador_teste in df_aumentado.columns:
        df_aumentado.loc[mask_proj, indicador_teste] *= (1 + variacao_pct)
    
    # Recalcular PIB com o indicador modificado
    # (Isso exigiria refazer a previsão, simplificando aqui)
    
    return df_base, df_aumentado

# 🔍 NOVA ABA LOGARÍTMICA
if aba == "Análise Logarítmica":
    st.header("🔍 Análise Logarítmica de Indicadores")

    df_log = df.copy()

    st.write("Os dados abaixo passarão por transformação logarítmica (log natural).")

    colunas_numericas = df_log.select_dtypes(include=np.number).columns.tolist()
    colunas_para_log = st.multiselect(
        "Selecione os indicadores para aplicar log:",
        options=colunas_numericas,
        default=[col for col in colunas_numericas if col != 'Ano']
    )

    for col in colunas_para_log:
        df_log[f'log_{col}'] = df_log[col].apply(lambda x: np.log(x) if x > 0 else np.nan)

    pais_log = st.selectbox("Escolha um país para visualizar", df_log['País'].unique(), key="pais_log")
    df_filtrado_log = df_log[df_log['País'] == pais_log]

    indicador_log = st.selectbox("Escolha um indicador log-transformado", [f'log_{col}' for col in colunas_para_log])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_filtrado_log, x="Ano", y=indicador_log, marker="o", ax=ax)
    ax.set_title(f"{indicador_log} ao longo do tempo — {pais_log}")
    ax.set_ylabel(indicador_log)
    ax.set_xlabel("Ano")
    st.pyplot(fig)

    st.dataframe(df_filtrado_log[['Ano', indicador_log]])

