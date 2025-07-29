import streamlit as st
import pandas as pd
import numpy as np
import wbdata
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import shap
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Dicionário expandido de indicadores
INDICADORES = {
    # Indicadores Originais
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
    # Novos Indicadores
    "FP.CPI.TOTL.ZG": "Inflacao_Anual_Consumidor",
    "BX.KLT.DINV.CD.WD": "Investimento_Estrangeiro_Direto",
    "SP.DYN.LE00.IN": "Expectativa_de_Vida",
    "SE.XPD.TOTL.GD.ZS": "Gastos_Governamentais_Educacao",
    "GC.DOD.TOTL.GD.ZS": "Divida_Governo_Central_perc_PIB",
    "IQ.CPA.BREG.XQ": "Qualidade_Regulatoria",
    "MS.MIL.XPND.GD.ZS": "Gastos_Militares_perc_PIB",
    "TX.VAL.TECH.MF.ZS": "Exportacoes_Alta_Tecnologia_perc"
}

# Variáveis monetárias para transformação logarítmica
VARIAVEIS_MONETARIAS = [
    'PIB_per_capita', 'Formacao_Bruta_Capital', 'Valor_Exportacoes',
    'Renda_Nacional_Bruta_per_Capita', 'Consumo_Familias',
    'Investimento_Estrangeiro_Direto'
]

@st.cache_data
def carregar_dados_banco_mundial(paises, ano_inicio, ano_fim):
    """
    Carrega dados do Banco Mundial com cache local para evitar erro 429.
    """
    cache_file = "dados_banco_mundial.csv"
    
    try:
        # Tenta carregar dados do cache local
        df_cached = pd.read_csv(cache_file)
        st.success("Dados carregados do cache local!")
        return df_cached
        
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Se não há cache, baixa da API
        st.info("Baixando dados do Banco Mundial...")
        
        try:
            # Cria range de anos no formato correto
            anos_range = (datetime(ano_inicio, 1, 1), datetime(ano_fim, 12, 31))
            
            # Baixa dados da API
            df_raw = wbdata.get_dataframe(
                INDICADORES,
                country=paises,
                data_date=anos_range
            )
            
            if df_raw is not None and not df_raw.empty:
                # Salva no cache para uso futuro
                df_raw.reset_index().to_csv(cache_file, index=False)
                st.success("Dados baixados e salvos no cache!")
                return df_raw
            else:
                st.error("Nenhum dado foi retornado da API.")
                return None
                
        except Exception as e:
            st.error(f"Erro ao acessar API do Banco Mundial: {str(e)}")
            return None

def processar_dados(df_raw):
    """
    Processa e limpa os dados com uma estratégia de imputação robusta
    para maximizar a retenção de países.
    """
    if df_raw is None:
        return None, None
    
    df = df_raw.copy().reset_index()
    
    # Renomear colunas
    if 'country' in df.columns:
        df.rename(columns={'country': 'País'}, inplace=True)
    if 'date' in df.columns:
        df.rename(columns={'date': 'Ano'}, inplace=True)
        df['Ano'] = pd.to_numeric(df['Ano'])
    
    if 'País' not in df.columns or 'Ano' not in df.columns:
        st.error("As colunas 'País' e/ou 'Ano' não estão disponíveis.")
        return None, None
    
    df = df.sort_values(by=['País', 'Ano'])
    
    # --- ESTRATÉGIA DE IMPUTAÇÃO MULTINÍVEL ---
    indicadores_a_processar = [col for col in df.columns if col not in ['País', 'Ano']]
    
    # Aplica imputação por país
    def imputar_grupo(group):
        result = group.set_index('Ano')[indicadores_a_processar]
        result = result.interpolate(method='linear', limit_direction='both')  # 1. Interpolação linear
        result = result.ffill()  # 2. Forward-fill para bordas iniciais
        result = result.bfill()  # 3. Backward-fill para bordas finais
        return result
    
    df_processado = df.groupby('País', group_keys=False).apply(
        lambda group: imputar_grupo(group)
    ).reset_index()
    
    # Como último recurso, preenche com 0
    df_processado.fillna(0, inplace=True)
    
    # --- Preparação para o Modelo ---
    df_model = df_processado.copy().set_index(['País', 'Ano'])
    
    # Remove a variável alvo dos preditores
    if 'PIB_per_capita' in df_model.columns:
        target = df_model['PIB_per_capita']
        predictors_df = df_model.drop(columns=['PIB_per_capita'])
    else:
        st.error("A coluna 'PIB_per_capita' não foi encontrada após o processamento.")
        return None, None
    
    # Engenharia de variáveis (lags)
    for var in predictors_df.columns:
        df_model[f'{var}_lag1'] = predictors_df.groupby('País')[var].shift(1)
    
    # Remove NaNs dos lags (primeira linha de cada país)
    df_model = df_model.dropna()
    
    return df_processado, df_model

def aplicar_transformacao_log(df, colunas_monetarias, aplicar=True):
    """
    Aplica ou reverte transformação logarítmica nas variáveis monetárias.
    """
    if df is None:
        return None
    
    df_transformed = df.copy()
    
    for col in colunas_monetarias:
        if col in df_transformed.columns:
            if aplicar:
                # Aplica log1p (log(1+x) para lidar com zeros)
                df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
            else:
                # Reverte com expm1
                df_transformed[col] = np.expm1(df_transformed[col])
    
    return df_transformed

def treinar_modelos(df_model, usar_log=False):
    """
    Treina múltiplos modelos de machine learning.
    """
    if df_model is None or df_model.empty:
        return None
    
    # Preparar dados
    df_work = df_model.copy()
    
    if usar_log:
        # Aplicar transformação logarítmica
        colunas_para_log = [col for col in VARIAVEIS_MONETARIAS if col in df_work.columns]
        colunas_para_log += [f"{col}_lag1" for col in VARIAVEIS_MONETARIAS if f"{col}_lag1" in df_work.columns]
        df_work = aplicar_transformacao_log(df_work, colunas_para_log, aplicar=True)
    
    # Separar features e target
    feature_cols = [col for col in df_work.columns if col != 'PIB_per_capita']
    X = df_work[feature_cols]
    y = df_work['PIB_per_capita']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Modelos
    modelos = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        ),
        'Linear Regression': LinearRegression()
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        # Treinar
        modelo.fit(X_train, y_train)
        
        # Predições
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Se usou log, reverter as predições
        if usar_log:
            y_pred_train = np.expm1(y_pred_train)
            y_pred_test = np.expm1(y_pred_test)
            y_train_orig = np.expm1(y_train)
            y_test_orig = np.expm1(y_test)
        else:
            y_train_orig = y_train
            y_test_orig = y_test
        
        # Métricas
        mae_test = mean_absolute_error(y_test_orig, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))
        r2_test = r2_score(y_test_orig, y_pred_test)
        
        resultados[nome] = {
            'modelo': modelo,
            'mae': mae_test,
            'rmse': rmse_test,
            'r2': r2_test,
            'y_test': y_test_orig,
            'y_pred': y_pred_test,
            'features': feature_cols
        }
    
    return resultados, X_train, y_train

def gerar_projecao_com_cenario(pais, modelo, df_processado, ano_final, 
                               indicador_para_chocar=None, variacao_percentual=0, 
                               usar_log=False):
    """
    Gera projeção com cenário específico para análise de sensibilidade.
    """
    # Preparar dados do país
    df_pais = df_processado[df_processado['País'] == pais].copy()
    if df_pais.empty:
        return None
    
    df_pais = df_pais.sort_values('Ano')
    ultimo_ano = df_pais['Ano'].max()
    
    # Se o ano final já existe nos dados, usar dados históricos
    if ano_final <= ultimo_ano:
        return df_pais[df_pais['Ano'] <= ano_final]
    
    projecoes = []
    
    # Dados base para projeção
    dados_base = df_pais.copy()
    
    for ano in range(ultimo_ano + 1, ano_final + 1):
        # Pegar dados do ano anterior
        dados_anterior = dados_base[dados_base['Ano'] == (ano - 1)].iloc[0]
        
        # Preparar features para predição
        features_dict = {}
        feature_cols = [col for col in dados_anterior.index if col not in ['País', 'Ano', 'PIB_per_capita']]
        
        for col in feature_cols:
            if col.endswith('_lag1'):
                # Para variáveis lag, usar o valor do ano anterior
                base_var = col.replace('_lag1', '')
                if base_var in dados_anterior.index:
                    features_dict[col] = dados_anterior[base_var]
                else:
                    features_dict[col] = dados_anterior[col]
            else:
                # Para outras variáveis, aplicar cenário se necessário
                valor_base = dados_anterior[col]
                
                if indicador_para_chocar and col == indicador_para_chocar:
                    # Aplicar variação percentual
                    valor_base = valor_base * (1 + variacao_percentual / 100)
                
                features_dict[col] = valor_base
        
        # Criar array de features na ordem correta
        feature_names = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else list(features_dict.keys())
        X_pred = np.array([features_dict.get(name, 0) for name in feature_names]).reshape(1, -1)
        
        # Aplicar transformação log se necessário
        if usar_log:
            for i, name in enumerate(feature_names):
                base_name = name.replace('_lag1', '')
                if base_name in VARIAVEIS_MONETARIAS:
                    X_pred[0, i] = np.log1p(max(0, X_pred[0, i]))
        
        # Fazer predição
        pib_pred = modelo.predict(X_pred)[0]
        
        # Reverter log se necessário
        if usar_log:
            pib_pred = np.expm1(pib_pred)
        
        # Criar nova linha
        nova_linha = dados_anterior.copy()
        nova_linha['Ano'] = ano
        nova_linha['PIB_per_capita'] = pib_pred
        
        # Atualizar indicadores com cenário
        if indicador_para_chocar and indicador_para_chocar in nova_linha.index:
            nova_linha[indicador_para_chocar] = nova_linha[indicador_para_chocar] * (1 + variacao_percentual / 100)
        
        # Adicionar à base de dados para próxima iteração
        dados_base = pd.concat([dados_base, nova_linha.to_frame().T], ignore_index=True)
        projecoes.append(nova_linha)
    
    # Combinar histórico com projeções
    if projecoes:
        df_projecoes = pd.DataFrame(projecoes)
        resultado = pd.concat([df_pais, df_projecoes], ignore_index=True)
    else:
        resultado = df_pais
    
    return resultado.sort_values('Ano')

def calcular_relatorio_qualidade(df_raw):
    """
    Calcula relatório de qualidade dos dados por país.
    """
    if df_raw is None:
        return pd.DataFrame()
    
    df_check = df_raw.copy()
    if hasattr(df_raw, 'reset_index'):
        df_check = df_raw.reset_index()
    
    # Garantir que temos as colunas corretas
    country_col = 'country' if 'country' in df_check.columns else 'País'
    
    if country_col not in df_check.columns:
        return pd.DataFrame()
    
    # Calcular dados faltantes por país
    indicadores_cols = [col for col in df_check.columns if col not in [country_col, 'date', 'Ano']]
    
    missing_data_report = df_check.groupby(country_col)[indicadores_cols].apply(
        lambda x: x.isnull().sum().sum() / (len(x.columns) * len(x))
    ).sort_values(ascending=False)
    
    missing_data_report = missing_data_report[missing_data_report > 0] * 100
    
    if not missing_data_report.empty:
        df_report = pd.DataFrame({
            'País': missing_data_report.index, 
            '% de Dados Faltantes (Original)': missing_data_report.values.round(2)
        })
        return df_report
    
    return pd.DataFrame()

def main():
    st.title("Análise Econômica e Projeção - PIB per Capita")
    
    # Sidebar
    st.sidebar.header("Configurações")
    
    # Seleção de países
    paises_disponiveis = [
        'BR', 'US', 'CN', 'DE', 'JP', 'GB', 'FR', 'IN', 'IT', 'CA',
        'KR', 'RU', 'AU', 'ES', 'MX', 'ID', 'NL', 'SA', 'TR', 'CH'
    ]
    
    paises_selecionados = st.sidebar.multiselect(
        "Selecione os países:",
        paises_disponiveis,
        default=['BR', 'US', 'CN', 'DE']
    )
    
    # Seleção de anos
    ano_inicio = st.sidebar.slider("Ano inicial:", 2000, 2020, 2010)
    ano_fim = st.sidebar.slider("Ano final:", 2021, 2023, 2022)
    
    # Opção de transformação logarítmica
    usar_log = st.sidebar.checkbox("Usar transformação logarítmica para melhorar o modelo")
    
    if not paises_selecionados:
        st.warning("Por favor, selecione pelo menos um país.")
        return
    
    # Carregar dados
    df_raw = carregar_dados_banco_mundial(paises_selecionados, ano_inicio, ano_fim)
    
    if df_raw is None:
        st.error("Não foi possível carregar os dados.")
        return
    
    # Relatório de qualidade dos dados
    st.subheader("Qualidade e Cobertura dos Dados")
    df_qualidade = calcular_relatorio_qualidade(df_raw)
    
    if not df_qualidade.empty:
        st.warning("Atenção: Os países a seguir exigiram preenchimento significativo de dados faltantes, o que pode afetar a precisão das projeções.")
        st.dataframe(df_qualidade.head(10))
    else:
        st.success("Todos os países selecionados possuem boa cobertura de dados.")
    
    # Processar dados
    df_processado, df_model = processar_dados(df_raw)
    
    if df_processado is None:
        st.error("Erro no processamento dos dados.")
        return
    
    # Mostrar estatísticas básicas
    st.subheader("Estatísticas dos Dados")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Países incluídos", df_processado['País'].nunique())
    with col2:
        st.metric("Período de análise", f"{df_processado['Ano'].min()}-{df_processado['Ano'].max()}")
    with col3:
        st.metric("Total de observações", len(df_processado))
    
    # Treinar modelos
    st.subheader("Comparação de Modelos")
    
    resultados, X_train, y_train = treinar_modelos(df_model, usar_log)
    
    if resultados is None:
        st.error("Erro no treinamento dos modelos.")
        return
    
    # Mostrar resultados dos modelos
    df_comparacao = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'MAE': [res['mae'] for res in resultados.values()],
        'RMSE': [res['rmse'] for res in resultados.values()],
        'R²': [res['r2'] for res in resultados.values()]
    })
    
    st.dataframe(df_comparacao)
    
    # Selecionar melhor modelo
    melhor_modelo_nome = df_comparacao.loc[df_comparacao['R²'].idxmax(), 'Modelo']
    melhor_modelo = resultados[melhor_modelo_nome]['modelo']
    
    st.success(f"Melhor modelo: {melhor_modelo_nome} (R² = {df_comparacao['R²'].max():.3f})")
    
    # Projeções
    st.subheader("Projeções Econômicas")
    
    col1, col2 = st.columns(2)
    with col1:
        pais_projecao = st.selectbox("Selecione o país:", df_processado['País'].unique())
    with col2:
        ano_projecao = st.slider("Projetar até o ano:", 
                                df_processado['Ano'].max() + 1, 
                                df_processado['Ano'].max() + 10, 
                                df_processado['Ano'].max() + 5)
    
    if st.button("Gerar Projeção"):
        df_projecao = gerar_projecao_com_cenario(
            pais_projecao, melhor_modelo, df_processado, ano_projecao, usar_log=usar_log
        )
        
        if df_projecao is not None:
            # Gráfico de projeção
            fig = px.line(df_projecao, x='Ano', y='PIB_per_capita', 
                         title=f'Projeção PIB per Capita - {pais_projecao}')
            
            # Marcar divisão entre histórico e projeção
            ultimo_ano_historico = df_processado[df_processado['País'] == pais_projecao]['Ano'].max()
            fig.add_vline(x=ultimo_ano_historico, line_dash="dash", line_color="red",
                         annotation_text="Início das Projeções")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Análise de Sensibilidade
    st.subheader("Análise de Sensibilidade")
    st.info("O PIB per capita é o alvo da previsão e não pode ser selecionado para manipulação.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pais_sensibilidade = st.selectbox("País para análise:", 
                                        df_processado['País'].unique(), 
                                        key="pais_sens")
    
    with col2:
        # Indicadores disponíveis (excluindo PIB per capita)
        indicadores_disponiveis = [col for col in df_processado.columns 
                                 if col not in ['País', 'Ano', 'PIB_per_capita']]
        indicador_choque = st.selectbox("Indicador para variar:", indicadores_disponiveis)
    
    with col3:
        variacao_pct = st.slider("Variação percentual:", -50, 50, 10)
    
    if st.button("Executar Análise de Sensibilidade"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cenário Base:**")
            projecao_base = gerar_projecao_com_cenario(
                pais_sensibilidade, melhor_modelo, df_processado, 
                ano_projecao, usar_log=usar_log
            )
        
        with col2:
            st.write(f"**Cenário com {indicador_choque} variando {variacao_pct:+}%:**")
            projecao_cenario = gerar_projecao_com_cenario(
                pais_sensibilidade, melhor_modelo, df_processado, 
                ano_projecao, indicador_choque, variacao_pct, usar_log=usar_log
            )
        
        if projecao_base is not None and projecao_cenario is not None:
            # Gráfico comparativo
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=projecao_base['Ano'], 
                y=projecao_base['PIB_per_capita'],
                mode='lines+markers',
                name='Cenário Base',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=projecao_cenario['Ano'], 
                y=projecao_cenario['PIB_per_capita'],
                mode='lines+markers',
                name=f'{indicador_choque} {variacao_pct:+}%',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'Análise de Sensibilidade - {pais_sensibilidade}',
                xaxis_title='Ano',
                yaxis_title='PIB per Capita (US$)'
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
