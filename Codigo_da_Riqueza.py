# =============================================================================
# CÓDIGO DA RIQUEZA - VERSÃO MELHORADA
# Previsão de PIB per capita usando Machine Learning
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import wbdata
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES E INDICADORES EXPANDIDOS
# =============================================================================

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

# =============================================================================
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO DE DADOS
# =============================================================================

@st.cache_data
def carregar_dados_banco_mundial(paises, anos):
    """
    Carrega dados do Banco Mundial com cache local para evitar erro 429.
    """
    cache_file = "dados_banco_mundial.csv"
    
    try:
        # Tenta carregar dados do cache local
        st.info("🔄 Tentando carregar dados do cache local...")
        df_cached = pd.read_csv(cache_file)
        
        # Verifica se os dados em cache cobrem os países e anos solicitados
        df_cached['date'] = pd.to_numeric(df_cached['date'])
        paises_cache = df_cached['country'].unique()
        anos_cache = df_cached['date'].unique()
        
        # Filtra pelos países e anos solicitados
        df_filtered = df_cached[
            (df_cached['country'].isin(paises)) & 
            (df_cached['date'].isin(anos))
        ]
        
        if len(df_filtered) > 0:
            st.success("✅ Dados carregados do cache!")
            return df_cached
        else:
            st.warning("⚠️ Cache insuficiente, baixando da API...")
            raise FileNotFoundError("Cache insuficiente")
            
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Se não há cache, baixa da API
        st.info("🌐 Baixando dados do Banco Mundial...")
        
        try:
            # Baixa dados da API
            df_raw = wbdata.get_dataframe(
                INDICADORES,
                country=paises,
                data_date=anos
            )
            
            if df_raw is not None and not df_raw.empty:
                # Salva no cache para uso futuro
                df_raw.reset_index().to_csv(cache_file, index=False)
                st.success("✅ Dados baixados e salvos no cache!")
                return df_raw
            else:
                st.error("❌ Erro ao baixar dados da API.")
                return None
                
        except Exception as e:
            st.error(f"❌ Erro ao acessar API: {str(e)}")
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
    # Para cada país, aplicamos uma série de preenchimentos.
    # Esta é a etapa mais importante para não perder países.
    indicadores_a_processar = [col for col in df.columns if col not in ['País', 'Ano']]
    
    df_processado = df.groupby('País', group_keys=False).apply(
        lambda group: group.set_index('Ano')[indicadores_a_processar]
                           .interpolate(method='linear', limit_direction='both') # 1. Interpolação linear
                           .ffill()                                               # 2. Forward-fill
                           .bfill()                                               # 3. Backward-fill
    ).reset_index()
    
    # Como último recurso, se um indicador inteiro para um país for nulo, preenchemos com 0.
    df_processado.fillna(0, inplace=True)
    
    # Relatório de qualidade dos dados
    dados_faltantes_antes = df_raw.isnull().sum().sum()
    dados_imputados = dados_faltantes_antes - df_processado.isnull().sum().sum()
    print(f"Relatório de Qualidade: {dados_imputados} de {dados_faltantes_antes} pontos de dados faltantes foram imputados.")
    
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
    
    # Após criar os lags, a primeira linha de cada país terá NaNs. Vamos removê-los.
    # Este é o único dropna() necessário e seguro.
    df_model = df_model.dropna()
    
    return df_processado, df_model

def mostrar_relatorio_qualidade(df_raw):
    """
    Mostra relatório de qualidade dos dados.
    """
    st.subheader("✅ Qualidade e Cobertura dos Dados")
    
    if df_raw is not None:
        # Calcular a porcentagem de dados originais que eram nulos para cada país
        df_raw_check = df_raw.reset_index()
        missing_data_report = df_raw_check.groupby('country').apply(
            lambda x: x.isnull().sum().sum() / (len(x.columns) * len(x))
        ).sort_values(ascending=False)
        
        missing_data_report = missing_data_report[missing_data_report > 0] * 100
        
        if not missing_data_report.empty:
            st.warning("⚠️ Atenção: Os países a seguir exigiram preenchimento significativo de dados faltantes, o que pode afetar a precisão das projeções. Os dados foram preenchidos usando interpolação e outras técnicas.")
            
            df_report = pd.DataFrame({
                'País': missing_data_report.index, 
                '% de Dados Faltantes (Original)': missing_data_report.values.round(2)
            })
            
            st.dataframe(df_report.head(10)) # Mostra os 10 piores
        else:
            st.success("✅ Todos os países selecionados possuem boa cobertura de dados.")

# =============================================================================
# FUNÇÕES DE MODELAGEM E PREVISÃO
# =============================================================================

def treinar_modelos(df_model, usar_log=False):
    """
    Treina diferentes modelos de machine learning.
    """
    if df_model is None or df_model.empty:
        return None, None, None
    
    # Preparar dados
    y = df_model['PIB_per_capita'].values
    X = df_model.drop(columns=['PIB_per_capita']).values
    
    # Aplicar transformação logarítmica se solicitado
    if usar_log:
        y = np.log1p(y)  # log(1 + y) para lidar com valores zero
    
    # Modelos
    modelos = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    # Validação cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    resultados = {}
    
    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=tscv, scoring='r2')
        resultados[nome] = {
            'modelo': modelo,
            'r2_medio': scores.mean(),
            'r2_std': scores.std()
        }
    
    # Encontrar melhor modelo
    melhor_nome = max(resultados.keys(), key=lambda k: resultados[k]['r2_medio'])
    melhor_modelo = resultados[melhor_nome]['modelo']
    
    # Treinar modelo final com todos os dados
    melhor_modelo.fit(X, y)
    
    return melhor_modelo, resultados, melhor_nome

def gerar_projecao_com_cenario(pais, modelo, df_processado, ano_final, 
                               indicador_para_chocar=None, variacao_percentual=0):
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
                    # Aplicar variação percentual PERSISTENTE
                    valor_base = valor_base * (1 + variacao_percentual / 100)
                
                features_dict[col] = valor_base
        
        # Criar array de features na ordem correta
        feature_names = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else list(features_dict.keys())
        X_pred = np.array([features_dict.get(name, 0) for name in feature_names]).reshape(1, -1)
        
        # Fazer predição
        pib_pred = modelo.predict(X_pred)[0]
        
        # Criar nova linha
        nova_linha = dados_anterior.copy()
        nova_linha['Ano'] = ano
        nova_linha['PIB_per_capita'] = pib_pred
        
        # Atualizar indicadores com cenário para próxima iteração
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

# =============================================================================
# SEÇÕES DA INTERFACE
# =============================================================================

def secao_analise_sensibilidade(df_processado, melhor_modelo, ano_projecao):
    """
    Seção corrigida da análise de sensibilidade.
    """
    st.subheader("🎯 Análise de Sensibilidade")
    st.info("💡 O PIB per capita é o alvo da previsão e não pode ser selecionado para manipulação.")
    
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
    
    if st.button("🔬 Executar Análise de Sensibilidade"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cenário Base:**")
            projecao_base = gerar_projecao_com_cenario(
                pais_sensibilidade, melhor_modelo, df_processado, ano_projecao
            )
        
        with col2:
            st.write(f"**Cenário com {indicador_choque} variando {variacao_pct:+}%:**")
            projecao_cenario = gerar_projecao_com_cenario(
                pais_sensibilidade, melhor_modelo, df_processado, 
                ano_projecao, indicador_choque, variacao_pct
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
            
            # Marcar divisão histórica
            ultimo_ano_hist = df_processado[df_processado['País'] == pais_sensibilidade]['Ano'].max()
            fig.add_vline(x=ultimo_ano_hist, line_dash="dot", line_color="gray")
            
            fig.update_layout(
                title=f'Análise de Sensibilidade - {pais_sensibilidade}',
                xaxis_title='Ano',
                yaxis_title='PIB per Capita (US$)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular impacto
            anos_projecao = projecao_base[projecao_base['Ano'] > ultimo_ano_hist]['Ano'].values
            if len(anos_projecao) > 0:
                pib_base_final = projecao_base[projecao_base['Ano'] == anos_projecao[-1]]['PIB_per_capita'].iloc[0]
                pib_cenario_final = projecao_cenario[projecao_cenario['Ano'] == anos_projecao[-1]]['PIB_per_capita'].iloc[0]
                impacto_pct = ((pib_cenario_final - pib_base_final) / pib_base_final) * 100
                
                st.metric(
                    f"Impacto no PIB per capita em {anos_projecao[-1]}:",
                    f"{impacto_pct:+.2f}%",
                    f"US$ {pib_cenario_final - pib_base_final:+,.0f}"
                )

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    st.set_page_config(
        page_title="💰 Código da Riqueza",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Código da Riqueza")
    st.subtitle("Previsão de PIB per capita usando Machine Learning")
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção de países
    paises_sugeridos = ['BRA', 'USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'ITA', 'CAN']
    paises_selecionados = st.sidebar.multiselect(
        "Selecione os países:",
        options=paises_sugeridos,
        default=['BRA', 'USA', 'CHN']
    )
    
    # Período de análise
    ano_inicio = st.sidebar.slider("Ano inicial:", 2000, 2020, 2010)
    ano_fim = st.sidebar.slider("Ano final:", 2015, 2023, 2022)
    
    # Ano de projeção
    ano_projecao = st.sidebar.slider("Ano para projeção:", 2024, 2035, 2030)
    
    # Opção de transformação logarítmica
    usar_log = st.sidebar.checkbox("Usar transformação logarítmica para melhorar o modelo")
    
    if not paises_selecionados:
        st.warning("⚠️ Selecione pelo menos um país para continuar.")
        return
    
    # Carregar dados
    anos = list(range(ano_inicio, ano_fim + 1))
    
    with st.spinner("Carregando dados..."):
        df_raw = carregar_dados_banco_mundial(paises_selecionados, anos)
    
    if df_raw is None:
        st.error("❌ Não foi possível carregar os dados. Tente novamente mais tarde.")
        return
    
    # Mostrar relatório de qualidade
    mostrar_relatorio_qualidade(df_raw)
    
    # Processar dados
    with st.spinner("Processando dados..."):
        df_processado, df_model = processar_dados(df_raw)
    
    if df_processado is None or df_model is None:
        st.error("❌ Erro no processamento dos dados.")
        return
    
    # Treinar modelos
    with st.spinner("Treinando modelos..."):
        melhor_modelo, resultados, melhor_nome = treinar_modelos(df_model, usar_log)
    
    if melhor_modelo is None:
        st.error("❌ Erro no treinamento dos modelos.")
        return
    
    # Mostrar resultados dos modelos
    st.subheader("🎯 Desempenho dos Modelos")
    col1, col2, col3 = st.columns(3)
    
    for i, (nome, resultado) in enumerate(resultados.items()):
        with [col1, col2, col3][i]:
            destaque = nome == melhor_nome
            st.metric(
                label=f"{'🏆 ' if destaque else ''}{nome}",
                value=f"{resultado['r2_medio']:.3f}",
                delta=f"±{resultado['r2_std']:.3f}"
            )
    
    st.success(f"✅ Melhor modelo: **{melhor_nome}** (R² = {resultados[melhor_nome]['r2_medio']:.3f})")
    
    # Gráficos principais
    st.subheader("📊 Análise Exploratória")
    
    # Gráfico de PIB per capita ao longo do tempo
    fig = px.line(
        df_processado, 
        x='Ano', 
        y='PIB_per_capita', 
        color='País',
        title='Evolução do PIB per capita'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Projeções
    st.subheader("🔮 Projeções para o Futuro")
    
    for pais in paises_selecionados:
        projecao = gerar_projecao_com_cenario(pais, melhor_modelo, df_processado, ano_projecao)
        
        if projecao is not None:
            fig = go.Figure()
            
            # Dados históricos
            dados_hist = projecao[projecao['Ano'] <= ano_fim]
            fig.add_trace(go.Scatter(
                x=dados_hist['Ano'],
                y=dados_hist['PIB_per_capita'],
                mode='lines+markers',
                name='Histórico',
                line=dict(color='blue')
            ))
            
            # Projeções
            dados_proj = projecao[projecao['Ano'] > ano_fim]
            if not dados_proj.empty:
                fig.add_trace(go.Scatter(
                    x=dados_proj['Ano'],
                    y=dados_proj['PIB_per_capita'],
                    mode='lines+markers',
                    name='Projeção',
                    line=dict(color='red', dash='dash')
                ))
            
            fig.add_vline(x=ano_fim, line_dash="dot", line_color="gray")
            fig.update_layout(
                title=f'Projeção PIB per capita - {pais}',
                xaxis_title='Ano',
                yaxis_title='PIB per Capita (US$)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Análise de sensibilidade
    secao_analise_sensibilidade(df_processado, melhor_modelo, ano_projecao)
    
    # Mostrar dados processados
    with st.expander("📋 Ver dados processados"):
        st.dataframe(df_processado)

if __name__ == "__main__":
    main()
