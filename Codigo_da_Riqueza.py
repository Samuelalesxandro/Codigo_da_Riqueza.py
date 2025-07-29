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
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DO PROJETO ---
INDICADORES = {
       "NY.GDP.PCAP.KD": "PIB_per_capita",
    "NE.GDI.FTOT.CD": "Formacao_Bruta_Capital",
    "SE.ADT.1524.LT.ZS": "Alfabetizacao_Jovens",
    "SL.TLF.CACT.ZS": "Participacao_Forca_Trabalho",
    "IT.NET.USER.ZS": "Cobertura_Internet",
    "NE.EXP.GNFS.CD": "Valor_Exportacoes",
    "NY.GNP.PCAP.CD": "Renda_Nacional_Bruta_per_Capita",
    "EG.ELC.ACCS.ZS": "Acesso_Eletricidade",
    "SI.POV.GINI": "Gini",
    "SL.UEM.TOTL.ZS": "Desemprego",
    "SE.PRM.CMPT.ZS": "Conclusao_Ensino_Primario",
    "NE.CON.PRVT.CD": "Consumo_Familias",
    "SH.H2O.BASW.ZS": "Cobertura_Agua_Potavel",
    "FP.CPI.TOTL.ZG": "Inflacao_Anual_Consumidor",
    "BX.KLT.DINV.CD.WD": "Investimento_Estrangeiro_Direto",
    "SE.XPD.TOTL.GD.ZS": "Gastos_Governamentais_Educacao"
    
}

ZONA_DO_EURO = ['DEU', 'FRA', 'ITA', 'ESP', 'PRT', 'GRC', 'IRL', 'NLD', 'AUT', 'BEL']
BRICS = ['BRA', 'RUS', 'IND', 'CHN', 'ZAF', 'EGY', 'ETH', 'IRN', 'SAU', 'ARE']      
PAISES_SUL_AMERICA = ['BRA', 'ARG', 'CHL', 'COL', 'PER', 'ECU', 'VEN', 'BOL', 'PRY', 'URY']
PAISES_SUDESTE_ASIATICO = ['IDN', 'THA', 'VNM', 'PHL', 'MYS', 'SGP', 'MMR', 'KHM', 'LAO', 'BRN']
TODOS_PAISES = list(set(PAISES_SUL_AMERICA + PAISES_SUDESTE_ASIATICO + BRICS + ZONA_DO_EURO))
DATA_INICIO = datetime(1995, 1, 1)
DATA_FIM = datetime(2025, 4, 30)

# Variáveis globais para cache manual
_cached_data = None
_cached_models = None

def carregar_dados_banco_mundial():
    """Carrega dados do Banco Mundial"""
    global _cached_data
    
    if _cached_data is not None:
        return _cached_data
    
    try:
        print("🔄 Coletando dados do Banco Mundial...")
        df_raw = wbdata.get_dataframe(indicators=INDICADORES, country=TODOS_PAISES, date=(DATA_INICIO, DATA_FIM))
        print("✅ Dados coletados com sucesso.")
        _cached_data = pd.DataFrame(df_raw)
        return _cached_data
    except Exception as e:
        st.error(f"❌ Erro ao baixar os dados: {e}")
        return None

def processar_dados(df_raw):
    """Processa e limpa os dados"""
    if df_raw is None:
        return None, None
    
    df = df_raw.reset_index()
    
    # Garante que as colunas 'País' e 'Ano' existam
    if 'country' in df.columns:
        df.rename(columns={'country': 'País'}, inplace=True)
    if 'date' in df.columns:
        df.rename(columns={'date': 'Ano'}, inplace=True)
    
    # Se ainda estiverem ausentes, tenta extrair do índice
    if 'País' not in df.columns and hasattr(df_raw, 'index') and hasattr(df_raw.index, 'get_level_values'):
        try:
            df['País'] = df_raw.index.get_level_values('country')
        except:
            pass
    if 'Ano' not in df.columns and hasattr(df_raw, 'index') and hasattr(df_raw.index, 'get_level_values'):
        try:
            df['Ano'] = df_raw.index.get_level_values('date')
        except:
            pass
    
    if 'País' not in df.columns or 'Ano' not in df.columns:
        st.error("❌ As colunas 'País' e/ou 'Ano' não estão disponíveis.")
        return None, None
    
    # Limpeza dos dados
    df = df.sort_values(by=['País', 'Ano'])
    df = df.groupby('País', group_keys=False).apply(lambda group: group.ffill().bfill())
    df = df.reset_index(drop=True)
    df = df.dropna()
    
    # Engenharia de variáveis
    df_model = df.copy().set_index(['País', 'Ano'])
    for var in df_model.columns:
        if var != 'PIB_per_capita':
            df_model[f'{var}_lag1'] = df_model.groupby('País')[var].shift(1)
    df_model = df_model.dropna()
    
    return df, df_model

def treinar_todos_modelos(df_model):
    """Treina e compara todos os modelos"""
    global _cached_models
    
    if _cached_models is not None:
        return _cached_models
    
    TARGET = 'PIB_per_capita'
    PREDICTORS = [col for col in df_model.columns if '_lag1' in col]
    X = df_model[PREDICTORS]
    y = df_model[TARGET]
    
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "Árvore de Decisão": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    }

    resultados = []
    modelos_treinados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X, y)
        y_pred = modelo.predict(X)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))  # Calcular RMSE manualmente
        mae = mean_absolute_error(y, y_pred)
        
        resultados.append({
            'Modelo': nome,
            'R²': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2)
        })
        
        modelos_treinados[nome] = modelo
        
        # Salvar importância das features para modelos tree-based
        if hasattr(modelo, 'feature_importances_'):
            importancia = pd.Series(modelo.feature_importances_, index=PREDICTORS).sort_values(ascending=False)
            modelos_treinados[f"{nome}_importance"] = importancia

    # Cache manual
    _cached_models = {
        'resultados': pd.DataFrame(resultados).sort_values('R²', ascending=False),
        'modelos': modelos_treinados,
        'X': X,
        'y': y,
        'predictors': PREDICTORS
    }
    
    return _cached_models

def gerar_projecao_realista(df_model, pais, modelo, ano_final=2035):
    """Gera projeções mais realistas do PIB per capita"""
    df_pred = df_model.reset_index()
    df_pred = df_pred[df_pred['País'] == pais].sort_values("Ano")

    if df_pred.empty:
        raise ValueError(f"Dados insuficientes para {pais}")

    df_pred = df_pred.copy()
    df_pred['Ano'] = pd.to_numeric(df_pred['Ano'], errors='coerce').astype(int)
    
    ultimo_ano = int(df_pred['Ano'].max())
    ano_final = int(ano_final)
    anos_futuros = list(range(ultimo_ano + 1, ano_final + 1))
    
    if not anos_futuros:
        return df_pred
    
    # Calcular crescimento histórico médio dos últimos 10 anos (mais conservador)
    df_recente = df_pred.tail(10)
    crescimento_historico = []
    
    for i in range(1, len(df_recente)):
        pib_anterior = df_recente.iloc[i-1]['PIB_per_capita']
        pib_atual = df_recente.iloc[i]['PIB_per_capita']
        if pib_anterior > 0:
            crescimento = (pib_atual / pib_anterior) - 1
            crescimento_historico.append(crescimento)
    
    # Crescimento médio histórico limitado
    if crescimento_historico:
        crescimento_medio = np.mean(crescimento_historico)
        crescimento_medio = max(-0.05, min(0.05, crescimento_medio))  # Limitar entre -5% e +5%
    else:
        crescimento_medio = 0.02  # 2% padrão
    
    # Tendências mais conservadoras para outros indicadores
    df_recente_5 = df_pred.tail(5)  # Usar apenas 5 anos para tendências
    tendencias = {}
    
    colunas_base = [col for col in df_pred.columns if not col.endswith('_lag1') and col not in ['País', 'Ano']]
    
    for col in colunas_base:
        if col != 'PIB_per_capita' and len(df_recente_5) > 1:
            valores = df_recente_5[col].values
            if len(valores) > 1 and not np.isnan(valores).all():
                # Tendência mais suave
                coef = np.polyfit(range(len(valores)), valores, 1)[0]
                tendencias[col] = coef * 0.5  # Reduzir intensidade da tendência
            else:
                tendencias[col] = 0
        else:
            tendencias[col] = 0
    
    # Preparar dados para projeção
    df_result = df_pred.copy()
    ultima_linha = df_pred.iloc[-1].copy()
    novas_linhas = []
    
    # Seed para reproducibilidade
    np.random.seed(42)
    
    for i, ano in enumerate(anos_futuros):
        nova_linha = ultima_linha.copy()
        nova_linha['Ano'] = int(ano)
        
        # Atualizar indicadores base com mudanças mais conservadoras
        for col in colunas_base:
            if col != 'PIB_per_capita':
                valor_atual = nova_linha[col]
                tendencia = tendencias.get(col, 0)
                
                # Decay mais forte para projeções de longo prazo
                anos_desde_base = i + 1
                fator_decay = 0.90 ** anos_desde_base
                
                novo_valor = valor_atual + (tendencia * fator_decay)
                
                # Variação aleatória menor
                variacao = np.random.normal(0, 0.01)  # ±1% ao invés de ±2%
                novo_valor *= (1 + variacao)
                
                # Aplicar limites mais rigorosos
                if col in ['Alfabetizacao_Jovens', 'Cobertura_Internet', 'Acesso_Eletricidade', 'Cobertura_Agua_Potavel']:
                    novo_valor = min(100, max(0, novo_valor))
                elif col == 'Desemprego':
                    novo_valor = min(30, max(0, novo_valor))  # Desemprego máximo 30%
                elif col == 'Gini':
                    novo_valor = min(80, max(20, novo_valor))
                else:
                    # Para indicadores econômicos, limitar mudanças extremas
                    mudanca_maxima = valor_atual * 0.1  # Máximo 10% de mudança por ano
                    novo_valor = max(valor_atual - mudanca_maxima, 
                                   min(valor_atual + mudanca_maxima, novo_valor))
                    novo_valor = max(0, novo_valor)
                
                nova_linha[col] = novo_valor
        
        # Atualizar variáveis lag
        for col in df_pred.columns:
            if col.endswith('_lag1'):
                col_base = col.replace('_lag1', '')
                if col_base in ultima_linha.index:
                    nova_linha[col] = ultima_linha[col_base]
        
        # Calcular PIB usando uma abordagem híbrida (modelo + tendência histórica)
        try:
            colunas_lag = [c for c in nova_linha.index if c.endswith('_lag1')]
            X_input = nova_linha[colunas_lag].values.reshape(1, -1)
            
            # Verificar se todas as features estão presentes
            if len(X_input[0]) == len(colunas_lag):
                pib_modelo = modelo.predict(X_input)[0]
                
                # Combinar previsão do modelo com tendência histórica
                pib_anterior = ultima_linha['PIB_per_capita']
                pib_tendencia = pib_anterior * (1 + crescimento_medio)
                
                # Média ponderada: 60% modelo, 40% tendência histórica
                peso_modelo = 0.6 * (0.95 ** i)  # Peso do modelo diminui com o tempo
                peso_tendencia = 1 - peso_modelo
                
                pib_combinado = (pib_modelo * peso_modelo) + (pib_tendencia * peso_tendencia)
                
                # Limitar crescimento anual a ±8%
                crescimento_anual = (pib_combinado / pib_anterior) - 1
                crescimento_anual = max(-0.08, min(0.08, crescimento_anual))
                
                pib_final = pib_anterior * (1 + crescimento_anual)
                nova_linha['PIB_per_capita'] = pib_final
            else:
                # Fallback se houver problema com as features
                crescimento_fallback = max(-0.02, min(0.03, crescimento_medio))
                nova_linha['PIB_per_capita'] = ultima_linha['PIB_per_capita'] * (1 + crescimento_fallback)
            
        except Exception as e:
            # Fallback mais conservador
            crescimento_fallback = max(-0.02, min(0.03, crescimento_medio))
            nova_linha['PIB_per_capita'] = ultima_linha['PIB_per_capita'] * (1 + crescimento_fallback)
        
        novas_linhas.append(nova_linha)
        ultima_linha = nova_linha
    
    # Combinar dados originais com projeções
    if novas_linhas:
        df_novo = pd.DataFrame(novas_linhas)
        df_novo['Ano'] = df_novo['Ano'].astype(int)
        df_result['Ano'] = df_result['Ano'].astype(int)
        df_completo = pd.concat([df_result, df_novo], ignore_index=True)
    else:
        df_completo = df_result
    
    df_completo['Ano'] = df_completo['Ano'].astype(int)
    return df_completo

def gerar_cenarios_realistas(df_model, pais, modelo, ano_final=2035):
    """Gera cenários mais realistas"""
    cenarios = {}
    ano_final = int(ano_final)
    
    # Cenário realista (base)
    np.random.seed(42)
    df_realista = gerar_projecao_realista(df_model, pais, modelo, ano_final)
    cenarios['Realista'] = df_realista
    
    ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
    
    # Cenário otimista (+1% adicional por ano, mais conservador)
    np.random.seed(123)
    df_otimista = gerar_projecao_realista(df_model, pais, modelo, ano_final)
    mask_futuro = df_otimista['Ano'] > ultimo_ano_real
    if mask_futuro.any():
        anos_futuros = df_otimista.loc[mask_futuro, 'Ano'] - ultimo_ano_real
        # Aplicar multiplicador gradual
        for idx, ano_futuro in enumerate(anos_futuros):
            multiplicador = 1.01 ** ano_futuro
            df_otimista.loc[df_otimista['Ano'] == df_otimista.loc[mask_futuro, 'Ano'].iloc[idx], 'PIB_per_capita'] *= multiplicador
    cenarios['Otimista'] = df_otimista
    
    # Cenário pessimista (-1% por ano, mais conservador)
    np.random.seed(456)
    df_pessimista = gerar_projecao_realista(df_model, pais, modelo, ano_final)
    mask_futuro = df_pessimista['Ano'] > ultimo_ano_real
    if mask_futuro.any():
        anos_futuros = df_pessimista.loc[mask_futuro, 'Ano'] - ultimo_ano_real
        # Aplicar multiplicador gradual
        for idx, ano_futuro in enumerate(anos_futuros):
            multiplicador = 0.99 ** ano_futuro
            df_pessimista.loc[df_pessimista['Ano'] == df_pessimista.loc[mask_futuro, 'Ano'].iloc[idx], 'PIB_per_capita'] *= multiplicador
    cenarios['Pessimista'] = df_pessimista
    
    return cenarios

# --- APLICAÇÃO STREAMLIT ---
def main():
    st.set_page_config(page_title="Código da Riqueza", layout="wide")
    st.title("📊 O Código da Riqueza — Painel Interativo Melhorado")
    
    # Inicializar dados na sessão se não existirem
    if 'df' not in st.session_state or 'df_model' not in st.session_state:
        with st.spinner("Carregando dados do Banco Mundial..."):
            df_raw = carregar_dados_banco_mundial()
            
            if df_raw is None:
                st.error("❌ Erro ao carregar dados do Banco Mundial")
                return
            
            df, df_model = processar_dados(df_raw)
            
            if df is None or df_model is None:
                st.error("❌ Erro ao processar dados")
                return
            
            st.session_state.df = df
            st.session_state.df_model = df_model
    
    df = st.session_state.df
    df_model = st.session_state.df_model
    
    # Treinar todos os modelos
    if 'models_data' not in st.session_state:
        with st.spinner("Treinando e comparando modelos..."):
            models_data = treinar_todos_modelos(df_model)
            st.session_state.models_data = models_data
    
    models_data = st.session_state.models_data
    
    # --- SEÇÃO DE COMPARAÇÃO DE MODELOS ---
    st.header("🤖 Comparação de Modelos de Machine Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performance dos Modelos")
        
        # Criar gráfico de comparação
        df_resultados = models_data['resultados']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R²
        axes[0].bar(df_resultados['Modelo'], df_resultados['R²'], color='skyblue')
        axes[0].set_title('R² Score')
        axes[0].set_ylabel('R²')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE
        axes[1].bar(df_resultados['Modelo'], df_resultados['RMSE'], color='lightcoral')
        axes[1].set_title('RMSE (menor é melhor)')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE
        axes[2].bar(df_resultados['Modelo'], df_resultados['MAE'], color='lightgreen')
        axes[2].set_title('MAE (menor é melhor)')
        axes[2].set_ylabel('MAE')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🏆 Ranking dos Modelos")
        st.dataframe(df_resultados, hide_index=True)
        
        melhor_modelo = df_resultados.iloc[0]['Modelo']
        melhor_r2 = df_resultados.iloc[0]['R²']
        st.success(f"**Melhor Modelo:** {melhor_modelo}\n\n**R²:** {melhor_r2}")
    
    # --- SELEÇÃO DO MODELO PARA PROJEÇÕES ---
    st.subheader("🎯 Seleção do Modelo para Projeções")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        modelo_escolhido = st.selectbox(
            "Escolha o modelo para as projeções:",
            options=df_resultados['Modelo'].tolist(),
            index=0  # Por padrão, o melhor modelo
        )
        
        modelo_selecionado = models_data['modelos'][modelo_escolhido]
        
        # Mostrar importância das features se disponível
        if f"{modelo_escolhido}_importance" in models_data['modelos']:
            st.subheader("📈 Importância das Variáveis")
            importance = models_data['modelos'][f"{modelo_escolhido}_importance"]
            
            # Gráfico de importância
            fig, ax = plt.subplots(figsize=(8, 6))
            top_features = importance.head(10)
            ax.barh(range(len(top_features)), top_features.values)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([feat.replace('_lag1', '').replace('_', ' ') for feat in top_features.index])
            ax.set_xlabel('Importância')
            ax.set_title(f'Top 10 Variáveis - {modelo_escolhido}')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        # Métricas do modelo selecionado
        modelo_info = df_resultados[df_resultados['Modelo'] == modelo_escolhido].iloc[0]
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("R² Score", f"{modelo_info['R²']:.4f}")
        with col_b:
            st.metric("RMSE", f"{modelo_info['RMSE']:.2f}")
        with col_c:
            st.metric("MAE", f"{modelo_info['MAE']:.2f}")
        
        st.info(f"""
        **Interpretação das Métricas:**
        - **R²**: {modelo_info['R²']:.1%} da variação do PIB é explicada pelo modelo
        - **RMSE**: Erro médio de ±${modelo_info['RMSE']:,.0f} nas previsões
        - **MAE**: Erro absoluto médio de ${modelo_info['MAE']:,.0f}
        """)
    
    # --- INTERFACE PRINCIPAL (resto do código permanece similar) ---
    st.header("📈 Análise por País")
    
    st.sidebar.header("🔎 Filtros")
    paises = sorted(df['País'].unique())
    pais_selecionado = st.sidebar.selectbox("Selecione um país", paises)
    
    anos_disponiveis = sorted(df[df['País'] == pais_selecionado]['Ano'].unique())
    if len(anos_disponiveis) > 1:
        ano_inicio, ano_fim = st.sidebar.select_slider(
            "Intervalo de anos",
            options=anos_disponiveis,
            value=(anos_disponiveis[0], anos_disponiveis[-1])
        )
    else:
        ano_inicio = ano_fim = anos_disponiveis[0]
    
    df_filtrado = df[(df['País'] == pais_selecionado) & (df['Ano'].between(ano_inicio, ano_fim))]
    
    # --- PROJEÇÕES REALISTAS ---
    st.subheader("🔮 Projeções Realistas do PIB per capita")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ano_limite = st.selectbox("Projetar até o ano:", [2030, 2035, 2040], index=1)
    with col2:
        tipo_analise = st.selectbox("Tipo de análise:", ["Cenário Único", "Múltiplos Cenários"])
    with col3:
        mostrar_detalhes = st.checkbox("Mostrar detalhes dos cálculos")
    
    if st.button("🚀 Gerar Projeções Realistas"):
        try:
            with st.spinner(f"Gerando projeções para {pais_selecionado}..."):
                
                if tipo_analise == "Cenário Único":
                    df_projecoes = gerar_projecao_realista(df_model, pais_selecionado, modelo_selecionado, ano_limite)
                    
                    ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
                    df_historico = df_projecoes[df_projecoes['Ano'] <= ultimo_ano_real]
                    df_futuro = df_projecoes[df_projecoes['Ano'] > ultimo_ano_real]
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    ax.plot(df_historico['Ano'], df_historico['PIB_per_capita'], 
                           'o-', label='Dados Históricos', linewidth=2, color='blue')
                    
                    if not df_futuro.empty:
                        ax.plot(df_futuro['Ano'], df_futuro['PIB_per_capita'], 
                               's--', label=f'Projeções ({modelo_escolhido})', linewidth=2, color='red', alpha=0.8)
                    
                    ax.set_title(f'Projeção PIB per capita - {pais_selecionado} (Modelo: {modelo_escolhido})')
                    ax.set_xlabel('Ano')
                    ax.set_ylabel('PIB per capita (US$)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Métricas
                    if not df_futuro.empty:
                        pib_atual = df_historico['PIB_per_capita'].iloc[-1]
                        pib_final = df_futuro['PIB_per_capita'].iloc[-1]
                        crescimento_total = ((pib_final / pib_atual) - 1) * 100
                        anos_projecao = len(df_futuro)
                        crescimento_anual = (((pib_final / pib_atual) ** (1/anos_projecao)) - 1) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PIB Atual", f"${pib_atual:,.0f}")
                        with col2:
                            st.metric("PIB Projetado", f"${pib_final:,.0f}")
                        with col3:
                            st.metric("Crescimento Anual", f"{crescimento_anual:.1f}%")
                        
                        # Aviso sobre realismo
                        if crescimento_anual > 6:
                            st.warning("⚠️ Crescimento alto projetado. Considere fatores externos que podem afetar essas previsões.")
                        elif crescimento_anual < -2:
                            st.warning("⚠️ Declínio econômico projetado. Políticas de estímulo podem ser necessárias.")
                        else:
                            st.success("✅ Projeção dentro de parâmetros econômicos realistas.")
                
                else:
                    # Múltiplos cenários
                    cenarios = gerar_cenarios_realistas(df_model, pais_selecionado, modelo_selecionado, ano_limite)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    cores = {'Pessimista': '#ff6b6b', 'Realista': '#4ecdc4', 'Otimista': '#45b7d1'}
                    ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
                    
                    # Dados históricos
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
                    
                    ax.set_title(f'Cenários de Projeção PIB per capita - {pais_selecionado} (Modelo: {modelo_escolhido})')
                    ax.set_xlabel('Ano')
                    ax.set_ylabel('PIB per capita (US$)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Métricas comparativas
                    st.subheader("📊 Comparação de Cenários")
                    col1, col2, col3 = st.columns(3)
                    
                    dados_cenarios = []
                    for i, (nome, df_cenario) in enumerate(cenarios.items()):
                        df_proj = df_cenario[df_cenario['Ano'] > ultimo_ano_real]
                        if not df_proj.empty:
                            pib_inicial = df_hist['PIB_per_capita'].iloc[-1]
                            pib_final = df_proj['PIB_per_capita'].iloc[-1]
                            crescimento_anual = (((pib_final / pib_inicial) ** (1/len(df_proj))) - 1) * 100
                            
                            dados_cenarios.append({
                                'Cenário': nome,
                                'PIB Final': f"${pib_final:,.0f}",
                                'Crescimento Anual': f"{crescimento_anual:.1f}%"
                            })
                            
                            with [col1, col2, col3][i]:
                                st.metric(f"PIB {nome}", f"${pib_final:,.0f}", f"{crescimento_anual:+.1f}% a.a.")
                    
                    # Tabela de comparação
                    if dados_cenarios:
                        df_comp_cenarios = pd.DataFrame(dados_cenarios)
                        st.dataframe(df_comp_cenarios, hide_index=True)
                
                st.success("✅ Projeções geradas com sucesso!")
                
        except Exception as e:
            st.error(f"❌ Erro ao gerar projeções: {str(e)}")
            if mostrar_detalhes:
                st.exception(e)
    
    # --- VISUALIZAÇÃO BÁSICA ---
    st.subheader(f"📈 Evolução dos indicadores — {pais_selecionado} ({ano_inicio}–{ano_fim})")
    
    indicadores = [col for col in df.columns if col not in ['País', 'Ano']]
    indicador_escolhido = st.selectbox("Escolha um indicador para visualizar", indicadores)
    
    if not df_filtrado.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_filtrado, x="Ano", y=indicador_escolhido, marker="o", ax=ax)
        ax.set_title(f"{indicador_escolhido.replace('_', ' ')} ao longo do tempo", fontsize=14)
        ax.set_ylabel(indicador_escolhido.replace('_', ' '))
        ax.set_xlabel("Ano")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # --- ANÁLISE DE SENSIBILIDADE MELHORADA ---
    st.subheader("🎯 Análise de Sensibilidade")
    
    if st.checkbox("Ativar análise de sensibilidade"):
        col1, col2 = st.columns(2)
        
        with col1:
            indicadores_disponiveis = [col.replace('_lag1', '') for col in df_model.columns if col.endswith('_lag1')]
            indicador_teste = st.selectbox("Selecione o indicador para testar:", indicadores_disponiveis)
            variacao = st.slider("Variação percentual do indicador:", -50, 50, 10, 5)
        
        with col2:
            st.info(f"""
            **Análise de Sensibilidade**
            
            Esta análise mostra como mudanças no indicador **{indicador_teste.replace('_', ' ')}** 
            afetam as projeções do PIB per capita.
            
            **Variação:** {variacao:+}%
            """)
        
        if st.button("Executar Análise de Sensibilidade"):
            try:
                # Projeção base
                df_base = gerar_projecao_realista(df_model, pais_selecionado, modelo_selecionado, 2030)
                
                # Criar versão modificada dos dados
                df_modificado = df_model.copy()
                col_lag = f"{indicador_teste}_lag1"
                
                if col_lag in df_modificado.columns:
                    df_modificado[col_lag] *= (1 + variacao/100)
                    
                    # Nova projeção com indicador modificado
                    df_sensibilidade = gerar_projecao_realista(df_modificado, pais_selecionado, modelo_selecionado, 2030)
                    
                    df_base['Ano'] = pd.to_numeric(df_base['Ano'], errors='coerce').astype(int)
                    df_sensibilidade['Ano'] = pd.to_numeric(df_sensibilidade['Ano'], errors='coerce').astype(int)
                    ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Dados históricos
                    df_hist = df_base[df_base['Ano'] <= ultimo_ano_real].copy()
                    if not df_hist.empty:
                        ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                               'o-', label='Histórico', linewidth=2, color='black')
                    
                    # Projeções
                    df_proj_base = df_base[df_base['Ano'] > ultimo_ano_real].copy()
                    df_proj_sens = df_sensibilidade[df_sensibilidade['Ano'] > ultimo_ano_real].copy()
                    
                    if not df_proj_base.empty:
                        ax.plot(df_proj_base['Ano'], df_proj_base['PIB_per_capita'], 
                               's--', label=f'Projeção Base ({modelo_escolhido})', linewidth=2, color='blue')
                    
                    if not df_proj_sens.empty:
                        ax.plot(df_proj_sens['Ano'], df_proj_sens['PIB_per_capita'], 
                               's--', label=f'{indicador_teste.replace("_", " ")} {variacao:+}%', linewidth=2, color='red')
                    
                    ax.set_title(f'Análise de Sensibilidade - {indicador_teste.replace("_", " ")}')
                    ax.set_xlabel('Ano')
                    ax.set_ylabel('PIB per capita (US$)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Calcular impacto
                    if not df_proj_base.empty and not df_proj_sens.empty:
                        pib_base_final = float(df_proj_base['PIB_per_capita'].iloc[-1])
                        pib_sens_final = float(df_proj_sens['PIB_per_capita'].iloc[-1])
                        impacto = ((pib_sens_final / pib_base_final) - 1) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                f"Impacto no PIB final",
                                f"{impacto:+.2f}%",
                                f"Variação de {variacao:+}% em {indicador_teste.replace('_', ' ')}"
                            )
                        
                        with col2:
                            # Elasticidade
                            elasticidade = impacto / variacao if variacao != 0 else 0
                            st.metric(
                                "Elasticidade",
                                f"{elasticidade:.3f}",
                                "Impacto por 1% de mudança"
                            )
                        
                        # Interpretação
                        if abs(elasticidade) > 0.5:
                            st.warning(f"⚠️ **Alta sensibilidade**: {indicador_teste.replace('_', ' ')} tem grande impacto no PIB")
                        elif abs(elasticidade) > 0.1:
                            st.info(f"📊 **Sensibilidade moderada**: {indicador_teste.replace('_', ' ')} tem impacto moderado no PIB")
                        else:
                            st.success(f"✅ **Baixa sensibilidade**: {indicador_teste.replace('_', ' ')} tem pouco impacto no PIB")
                    else:
                        st.warning("Dados insuficientes para calcular o impacto")
                else:
                    st.error(f"Indicador {col_lag} não encontrado no modelo")
                
            except Exception as e:
                st.error(f"❌ Erro na análise de sensibilidade: {str(e)}")
    
    # --- COMPARAÇÃO ENTRE PAÍSES ---
    st.subheader("🌍 Comparação de Crescimento Projetado")
    
    if st.checkbox("Comparar crescimento entre países"):
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
                df_anos = df_model.reset_index()
                df_anos['Ano'] = pd.to_numeric(df_anos['Ano'], errors='coerce').astype(int)
                ultimo_ano_real = int(df_anos['Ano'].max())
                
                for i, pais in enumerate(paises_comparacao):
                    try:
                        df_pais = gerar_projecao_realista(df_model, pais, modelo_selecionado, 2035)
                        df_pais['Ano'] = pd.to_numeric(df_pais['Ano'], errors='coerce').astype(int)
                        
                        df_hist = df_pais[df_pais['Ano'] <= ultimo_ano_real].copy()
                        df_proj = df_pais[df_pais['Ano'] > ultimo_ano_real].copy()
                        
                        if not df_hist.empty:
                            ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                                   'o-', color=cores[i], alpha=0.7, linewidth=2)
                        
                        if not df_proj.empty:
                            ax.plot(df_proj['Ano'], df_proj['PIB_per_capita'], 
                                   's--', color=cores[i], alpha=0.9, linewidth=2, label=pais)
                        
                        if not df_proj.empty and not df_hist.empty:
                            pib_inicial = float(df_hist['PIB_per_capita'].iloc[-1])
                            pib_final = float(df_proj['PIB_per_capita'].iloc[-1])
                            anos_projecao = len(df_proj)
                            crescimento_anual = (((pib_final / pib_inicial) ** (1/anos_projecao)) - 1) * 100
                            
                            dados_comparacao.append({
                                'País': pais,
                                'PIB Atual': pib_inicial,
                                'PIB Projetado': pib_final,
                                'Crescimento Anual': crescimento_anual,
                                'Modelo': modelo_escolhido
                            })
                        
                        st.success(f"✅ Projeção gerada para {pais}")
                            
                    except Exception as e:
                        st.error(f"❌ Erro ao processar {pais}: {str(e)}")
                        continue
                
                if dados_comparacao:
                    ax.set_title(f'Comparação de Crescimento Projetado entre Países (Modelo: {modelo_escolhido})')
                    ax.set_xlabel('Ano')
                    ax.set_ylabel('PIB per capita (US$)')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Tabela comparativa
                    df_comp = pd.DataFrame(dados_comparacao)
                    df_comp = df_comp.sort_values('Crescimento Anual', ascending=False)
                    
                    st.subheader("📊 Ranking de Crescimento Projetado")
                    
                    # Formatação da tabela
                    df_display = df_comp.copy()
                    df_display['PIB Atual'] = df_display['PIB Atual'].apply(lambda x: f"${x:,.0f}")
                    df_display['PIB Projetado'] = df_display['PIB Projetado'].apply(lambda x: f"${x:,.0f}")
                    df_display['Crescimento Anual'] = df_display['Crescimento Anual'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(df_display, hide_index=True)
                    
                    # Destacar o melhor e pior desempenho
                    if len(df_comp) > 1:
                        melhor = df_comp.iloc[0]
                        pior = df_comp.iloc[-1]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"🏆 **Maior crescimento:** {melhor['País']} ({melhor['Crescimento Anual']:.1f}% ao ano)")
                        with col2:
                            st.info(f"📉 **Menor crescimento:** {pior['País']} ({pior['Crescimento Anual']:.1f}% ao ano)")
                else:
                    plt.close()
                    st.warning("❌ Nenhuma projeção foi gerada com sucesso para os países selecionados.")
            
            except Exception as e:
                st.error(f"❌ Erro geral na comparação: {str(e)}")
    
    # --- DADOS FILTRADOS ---
    st.subheader("📋 Dados filtrados")
    if not df_filtrado.empty:
        st.dataframe(df_filtrado)
        
        st.download_button(
            label="📥 Baixar dados filtrados como CSV",
            data=df_filtrado.to_csv(index=False).encode('utf-8'),
            file_name=f"{pais_selecionado}_dados_filtrados.csv",
            mime='text/csv'
        )
    else:
        st.warning("Nenhum dado disponível para os filtros selecionados")
    
    # --- INFORMAÇÕES DETALHADAS DO MODELO ---
    with st.expander("🔍 Informações Detalhadas dos Modelos"):
        
        # Métricas de todos os modelos
        st.subheader("📊 Métricas Completas")
        st.dataframe(models_data['resultados'], hide_index=True)
        
        # Informações do modelo selecionado
        st.subheader(f"🎯 Modelo Selecionado: {modelo_escolhido}")
        modelo_info = models_data['resultados'][models_data['resultados']['Modelo'] == modelo_escolhido].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{modelo_info['R²']:.4f}")
        with col2:
            st.metric("RMSE", f"{modelo_info['RMSE']:.2f}")
        with col3:
            st.metric("MAE", f"{modelo_info['MAE']:.2f}")
        with col4:
            st.metric("Total Observações", f"{len(models_data['y']):,}")
        
        st.write(f"**Período dos dados:** {df['Ano'].min()} - {df['Ano'].max()}")
        st.write(f"**Total de países:** {len(df['País'].unique())}")
        st.write(f"**Número de preditores:** {len(models_data['predictors'])}")
        
        # Explicação das métricas
        st.info("""
        **📚 Explicação das Métricas:**
        - **R²**: Proporção da variância explicada pelo modelo (0-1, maior é melhor)
        - **RMSE**: Raiz do erro quadrático médio em US$ (menor é melhor)
        - **MAE**: Erro absoluto médio em US$ (menor é melhor)
        
        **🎯 Interpretação:**
        - R² > 0.8: Excelente poder explicativo
        - R² 0.6-0.8: Bom poder explicativo  
        - R² < 0.6: Poder explicativo limitado
        """)

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    main()
else:
    # Executar código inicial quando importado
    try:
        print("--- INICIANDO ANÁLISE: O CÓDIGO DA RIQUEZA (VERSÃO MELHORADA) ---")
        print("🔄 Coletando dados do Banco Mundial...")
        df_raw = wbdata.get_dataframe(indicators=INDICADORES, country=TODOS_PAISES, date=(DATA_INICIO, DATA_FIM))
        print("✅ Dados coletados com sucesso.")
        
        df, df_model = processar_dados(pd.DataFrame(df_raw))
        
        if df is not None and df_model is not None:
            print(f"\n📊 Amostra dos dados limpos:")
            print(df.head())
            print(f"\n📈 Tamanho do dataset final para modelagem: {df_model.shape[0]} observações.")
            
            # Treinar e comparar todos os modelos
            print("\n🚀 Treinando e comparando modelos...")
            models_data = treinar_todos_modelos(df_model)
            
            print("\n🏆 Comparação de Modelos:")
            print(models_data['resultados'].to_string(index=False))
            
            melhor_modelo = models_data['resultados'].iloc[0]['Modelo']
            melhor_r2 = models_data['resultados'].iloc[0]['R²']
            print(f"\n🥇 Melhor modelo: {melhor_modelo} (R² = {melhor_r2:.4f})")
            
            # Exportar dados
            print("\n💾 Exportando dados e resultados...")
            df_export = df_model.reset_index()
            df_export.to_csv("dados_modelo_completos.csv", index=False)
            models_data['resultados'].to_csv("comparacao_modelos.csv", index=False)
            
            print("\n🖥️ Execute: streamlit run codigo_riqueza_melhorado.py")
            
        else:
            print("❌ Erro no processamento dos dados")
            
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        print("Execute com: streamlit run codigo_riqueza_melhorado.py")
