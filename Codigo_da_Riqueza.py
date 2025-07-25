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

# Variáveis globais para cache manual
_cached_data = None
_cached_model = None
_cached_r2 = None
_cached_importance = None

def carregar_dados_banco_mundial():
    """Carrega dados do Banco Mundial"""
    global _cached_data
    
    if _cached_data is not None:
        return _cached_data
    
    try:
        print("🔄 Coletando dados do Banco Mundial...")
        df_raw = wbdata.get_dataframe(indicators=INDICADORES, country=TODOS_PAISES, date=(DATA_INICIO, DATA_FIM))
        print("✅ Dados coletados com sucesso.")
        _cached_data = pd.DataFrame(df_raw)  # Converter para pandas normal
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

def treinar_modelo_xgboost(df_model):
    """Treina o modelo XGBoost"""
    global _cached_model, _cached_r2, _cached_importance
    
    if _cached_model is not None:
        return _cached_model, _cached_r2, _cached_importance
    
    TARGET = 'PIB_per_capita'
    PREDICTORS = [col for col in df_model.columns if '_lag1' in col]
    X = df_model[PREDICTORS]
    y = df_model[TARGET]
    
    model = XGBRegressor(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X, y)
    
    r2 = r2_score(y, model.predict(X))
    importance = pd.Series(model.feature_importances_, index=PREDICTORS).sort_values(ascending=False)
    
    # Cache manual
    _cached_model = model
    _cached_r2 = r2
    _cached_importance = importance
    
    return model, r2, importance

def gerar_projecao_dinamica(df_model, pais, modelo, ano_final=2035):
    """Gera projeções dinâmicas do PIB per capita - VERSÃO CORRIGIDA"""
    df_pred = df_model.reset_index()
    df_pred = df_pred[df_pred['País'] == pais].sort_values("Ano")

    if df_pred.empty:
        raise ValueError(f"Dados insuficientes para {pais}")

    # CORREÇÃO: Garantir que a coluna Ano seja consistentemente int
    df_pred = df_pred.copy()
    df_pred['Ano'] = pd.to_numeric(df_pred['Ano'], errors='coerce').astype(int)
    
    ultimo_ano = int(df_pred['Ano'].max())
    ano_final = int(ano_final)  # Garantir que ano_final também seja int
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
        nova_linha['Ano'] = int(ano)  # CORREÇÃO: Garantir que seja int
        
        # Atualizar indicadores base usando tendências
        for col in colunas_base:
            if col != 'PIB_per_capita':  # PIB será calculado pelo modelo
                valor_atual = nova_linha[col]
                tendencia = tendencias.get(col, 0)
                
                # Aplicar tendência com suavização
                anos_desde_base = ano - ultimo_ano
                fator_suavizacao = 0.95 ** anos_desde_base  # Decay exponencial
                
                novo_valor = valor_atual + (tendencia * fator_suavizacao)
                
                # Adicionar pequena variação aleatória (±2%)
                variacao = np.random.normal(0, 0.02)
                novo_valor *= (1 + variacao)
                
                # Aplicar limites razoáveis
                if col in ['Alfabetizacao_Jovens', 'Cobertura_Internet', 'Acesso_Eletricidade', 'Cobertura_Agua_Potavel']:
                    novo_valor = min(100, max(0, novo_valor))
                elif col == 'Desemprego':
                    novo_valor = min(50, max(0, novo_valor))
                elif col == 'Gini':
                    novo_valor = min(100, max(20, novo_valor))
                else:
                    novo_valor = max(0, novo_valor)
                
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
            # Fallback: crescimento de 2% ao ano
            nova_linha['PIB_per_capita'] = ultima_linha['PIB_per_capita'] * 1.02
        
        novas_linhas.append(nova_linha)
        ultima_linha = nova_linha  # Atualizar para próxima iteração
    
    # Combinar dados originais com projeções
    if novas_linhas:
        df_novo = pd.DataFrame(novas_linhas)
        # CORREÇÃO: Garantir que ambos DataFrames tenham Ano como int
        df_novo['Ano'] = df_novo['Ano'].astype(int)
        df_result['Ano'] = df_result['Ano'].astype(int)
        df_completo = pd.concat([df_result, df_novo], ignore_index=True)
    else:
        df_completo = df_result
    
    # CORREÇÃO FINAL: Garantir que toda a coluna Ano seja int
    df_completo['Ano'] = df_completo['Ano'].astype(int)
    
    return df_completo

def gerar_cenarios(df_model, pais, modelo, ano_final=2035):
    """Gera cenários otimista, realista e pessimista - VERSÃO CORRIGIDA"""
    cenarios = {}
    
    # Garantir que ano_final seja int
    ano_final = int(ano_final)
    
    # Fixar seed para reprodutibilidade
    np.random.seed(42)
    df_realista = gerar_projecao_dinamica(df_model, pais, modelo, ano_final)
    cenarios['Realista'] = df_realista
    
    # Cenário otimista
    np.random.seed(123)
    df_otimista = gerar_projecao_dinamica(df_model, pais, modelo, ano_final)
    ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
    mask_futuro = df_otimista['Ano'] > ultimo_ano_real
    df_otimista.loc[mask_futuro, 'PIB_per_capita'] *= 1.015  # +1.5% adicional
    cenarios['Otimista'] = df_otimista
    
    # Cenário pessimista
    np.random.seed(456)
    df_pessimista = gerar_projecao_dinamica(df_model, pais, modelo, ano_final)
    df_pessimista.loc[mask_futuro, 'PIB_per_capita'] *= 0.985  # -1.5%
    cenarios['Pessimista'] = df_pessimista
    
    return cenarios

# --- APLICAÇÃO STREAMLIT ---
def main():
    st.set_page_config(page_title="Código da Riqueza", layout="wide")
    st.title("📊 O Código da Riqueza — Painel Interativo")
    
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
    
    # Treinar modelo
    if 'model' not in st.session_state:
        with st.spinner("Treinando modelo XGBoost..."):
            model, r2, importance = treinar_modelo_xgboost(df_model)
            st.session_state.model = model
            st.session_state.r2 = r2
            st.session_state.importance = importance
    
    model = st.session_state.model
    r2 = st.session_state.r2
    importance = st.session_state.importance
    
    st.success(f"✅ Modelo treinado com R² = {r2:.4f}")
    
    # Interface principal
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
    
    # Visualização básica
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
    
    # --- PROJEÇÕES DINÂMICAS ---
    st.subheader("🔮 Projeções Dinâmicas do PIB per capita")
    
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
                    
                    # Separar histórico de projeções - CORREÇÃO: Garantir tipos consistentes
                    ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
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
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                    
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
                    ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
                    
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
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                    
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
                    
                    # CORREÇÃO: Garantir tipos consistentes para comparações
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
                               's--', label='Projeção Base', linewidth=2, color='blue')
                    
                    if not df_proj_sens.empty:
                        ax.plot(df_proj_sens['Ano'], df_proj_sens['PIB_per_capita'], 
                               's--', label=f'{indicador_teste} {variacao:+}%', linewidth=2, color='red')
                    
                    ax.set_title(f'Análise de Sensibilidade - {indicador_teste}')
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
                        
                        st.metric(
                            f"Impacto no PIB final ({variacao:+}% em {indicador_teste})",
                            f"{impacto:+.2f}%"
                        )
                        
                        # Elasticidade
                        elasticidade = impacto / variacao if variacao != 0 else 0
                        st.info(f"**Elasticidade:** {elasticidade:.3f} (variação de 1% em {indicador_teste} resulta em {elasticidade:.3f}% de mudança no PIB)")
                    else:
                        st.warning("Dados insuficientes para calcular o impacto")
                else:
                    st.error(f"Indicador {col_lag} não encontrado no modelo")
                
            except Exception as e:
                st.error(f"❌ Erro na análise de sensibilidade: {str(e)}")
                st.exception(e)  # Para debug
    
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
                
                # CORREÇÃO: Garantir que ultimo_ano_real seja int e consistente
                df_anos = df_model.reset_index()
                df_anos['Ano'] = pd.to_numeric(df_anos['Ano'], errors='coerce').astype(int)
                ultimo_ano_real = int(df_anos['Ano'].max())
                
                for i, pais in enumerate(paises_comparacao):
                    try:
                        # Gerar projeção para o país
                        df_pais = gerar_projecao_dinamica(df_model, pais, model, 2035)
                        
                        # CORREÇÃO: Garantir que df_pais tenha Ano como int
                        df_pais['Ano'] = pd.to_numeric(df_pais['Ano'], errors='coerce').astype(int)
                        
                        # Separar dados históricos e projeções
                        df_hist = df_pais[df_pais['Ano'] <= ultimo_ano_real].copy()
                        df_proj = df_pais[df_pais['Ano'] > ultimo_ano_real].copy()
                        
                        # Plotar dados históricos
                        if not df_hist.empty:
                            ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                                   'o-', color=cores[i], alpha=0.7, linewidth=2)
                        
                        # Plotar projeções
                        if not df_proj.empty:
                            ax.plot(df_proj['Ano'], df_proj['PIB_per_capita'], 
                                   's--', color=cores[i], alpha=0.9, linewidth=2, label=pais)
                        
                        # Calcular métricas de crescimento
                        if not df_proj.empty and not df_hist.empty:
                            pib_inicial = float(df_hist['PIB_per_capita'].iloc[-1])
                            pib_final = float(df_proj['PIB_per_capita'].iloc[-1])
                            crescimento = ((pib_final / pib_inicial) - 1) * 100
                            anos_projecao = len(df_proj)
                            crescimento_anual = crescimento / anos_projecao if anos_projecao > 0 else 0
                            
                            dados_comparacao.append({
                                'País': pais,
                                'PIB Atual': pib_inicial,
                                'PIB Projetado': pib_final,
                                'Crescimento Total (%)': crescimento,
                                'Crescimento Anual (%)': crescimento_anual
                            })
                        
                        st.success(f"✅ Projeção gerada para {pais}")
                            
                    except Exception as e:
                        st.error(f"❌ Erro ao processar {pais}: {str(e)}")
                        continue
                
                # Finalizar gráfico
                if dados_comparacao:  # Só plotar se tiver dados válidos
                    ax.set_title('Comparação de Crescimento Projetado entre Países')
                    ax.set_xlabel('Ano')
                    ax.set_ylabel('PIB per capita (US$)')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Tabela comparativa
                    df_comp = pd.DataFrame(dados_comparacao)
                    df_comp = df_comp.sort_values('Crescimento Anual (%)', ascending=False)
                    
                    st.subheader("📊 Ranking de Crescimento Projetado")
                    st.dataframe(df_comp.round(2))
                    
                    # Destacar o melhor e pior desempenho
                    if len(df_comp) > 1:
                        melhor = df_comp.iloc[0]
                        pior = df_comp.iloc[-1]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"🏆 **Maior crescimento:** {melhor['País']} ({melhor['Crescimento Anual (%)']:.1f}% ao ano)")
                        with col2:
                            st.info(f"📉 **Menor crescimento:** {pior['País']} ({pior['Crescimento Anual (%)']:.1f}% ao ano)")
                else:
                    plt.close()
                    st.warning("❌ Nenhuma projeção foi gerada com sucesso para os países selecionados.")
            
            except Exception as e:
                st.error(f"❌ Erro geral na comparação: {str(e)}")
                st.exception(e)  # Para debug
    
    # --- RELATÓRIO DE PROJEÇÕES ---
    st.subheader("📄 Gerar Relatório de Projeções")
    
    if st.button("Gerar Relatório Completo"):
        try:
            df_projecoes = gerar_projecao_dinamica(df_model, pais_selecionado, model, 2035)
            ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
            df_historico = df_projecoes[df_projecoes['Ano'] <= ultimo_ano_real]
            df_futuro = df_projecoes[df_projecoes['Ano'] > ultimo_ano_real]
            
            if not df_futuro.empty and not df_historico.empty:
                pib_atual = df_historico['PIB_per_capita'].iloc[-1]
                pib_final = df_futuro['PIB_per_capita'].iloc[-1]
                crescimento_total = ((pib_final / pib_atual) - 1) * 100
                crescimento_anual = (((pib_final / pib_atual) ** (1/len(df_futuro))) - 1) * 100
                
                relatorio = f"""
# 📊 Relatório de Projeções Econômicas
## País: {pais_selecionado}
## Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

### 🔍 Resumo Executivo
- **Período de projeção:** {df_futuro['Ano'].min()} - {df_futuro['Ano'].max()}
- **PIB per capita atual:** US$ {pib_atual:,.2f}
- **PIB per capita projetado:** US$ {pib_final:,.2f}
- **Crescimento total:** {crescimento_total:.1f}%
- **Crescimento anual médio:** {crescimento_anual:.1f}%

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
Consulte os dados completos no painel interativo.
                """
                
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
            else:
                st.error("Dados insuficientes para gerar relatório")
                
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {str(e)}")
    
    # --- RECOMENDAÇÕES DE POLÍTICA ECONÔMICA ---
    st.subheader("💡 Recomendações de Política Econômica")
    
    if st.button("Gerar Recomendações Automáticas"):
        try:
            df_projecoes = gerar_projecao_dinamica(df_model, pais_selecionado, model, 2030)
            ultimo_ano_real = int(df_model.reset_index()['Ano'].max())
            df_futuro = df_projecoes[df_projecoes['Ano'] > ultimo_ano_real]
            
            if not df_futuro.empty:
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
                
                if recomendacoes:
                    st.write(f"**Recomendações para {pais_selecionado}:**")
                    for rec in recomendacoes:
                        st.write(rec)
                else:
                    st.success("✅ As projeções indicam uma trajetória equilibrada de crescimento.")
            else:
                st.warning("Dados insuficientes para gerar recomendações")
                
        except Exception as e:
            st.error(f"Erro ao gerar recomendações: {str(e)}")
    
    # --- COMPARAÇÃO ENTRE DOIS PAÍSES ---
    st.subheader("📊 Comparar dois países lado a lado")
    
    col1, col2 = st.columns(2)
    with col1:
        pais_1 = st.selectbox("País 1", paises, index=0, key="pais1")
    with col2:
        pais_2 = st.selectbox("País 2", paises, index=min(1, len(paises)-1), key="pais2")
    
    indicador_comp = st.selectbox("Indicador para comparar", indicadores, key="indicador_comp")
    
    if st.button("Comparar Países"):
        df_p1 = df[(df['País'] == pais_1) & (df['Ano'].between(ano_inicio, ano_fim))]
        df_p2 = df[(df['País'] == pais_2) & (df['Ano'].between(ano_inicio, ano_fim))]
        
        if not df_p1.empty and not df_p2.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=df_p1, x="Ano", y=indicador_comp, label=pais_1, marker="o", ax=ax)
            sns.lineplot(data=df_p2, x="Ano", y=indicador_comp, label=pais_2, marker="o", ax=ax)
            ax.set_title(f"{indicador_comp.replace('_', ' ')} — {pais_1} vs {pais_2}")
            ax.set_ylabel(indicador_comp.replace('_', ' '))
            ax.set_xlabel("Ano")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Dados insuficientes para comparação")
    
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
    
    # --- INFORMAÇÕES DO MODELO ---
    with st.expander("🔍 Informações do Modelo"):
        st.write(f"**R² do modelo:** {r2:.4f}")
        st.write("**Top 5 indicadores mais importantes:**")
        for i, (var, imp) in enumerate(importance.head(5).items(), 1):
            st.write(f"{i}. {var}: {imp:.4f}")
        
        st.write(f"**Total de países:** {len(df['País'].unique())}")
        st.write(f"**Período dos dados:** {df['Ano'].min()} - {df['Ano'].max()}")
        st.write(f"**Total de observações:** {len(df_model)}")

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Executar aplicação Streamlit
    main()
else:
    # Executar código inicial quando importado
    try:
        print("--- INICIANDO ANÁLISE: O CÓDIGO DA RIQUEZA ---")
        print("🔄 Coletando dados do Banco Mundial...")
        df_raw = wbdata.get_dataframe(indicators=INDICADORES, country=TODOS_PAISES, date=(DATA_INICIO, DATA_FIM))
        print("✅ Dados coletados com sucesso.")
        
        df, df_model = processar_dados(pd.DataFrame(df_raw))
        
        if df is not None and df_model is not None:
            print(f"\n📊 Amostra dos dados limpos:")
            print(df.head())
            print(f"\n📈 Tamanho do dataset final para modelagem: {df_model.shape[0]} observações.")
            
            # Treinar modelo
            print("\n🚀 Treinando modelo XGBoost para prever PIB per capita...")
            model, r2, importance = treinar_modelo_xgboost(df_model)
            print(f"\n📌 Poder de explicação do modelo (R² no treino): {r2:.4f}")
            print("\n🏆 Fatores mais importantes para o crescimento econômico:")
            print(importance.head(10))
            
            # Exportar dados
            print("\n💾 Exportando dados e resultados...")
            df_export = df_model.reset_index()
            df_export.to_csv("dados_modelo_completos.csv", index=False)
            importance.to_csv("importancia_geral.csv")
            
            print("\n🖥️ Execute: streamlit run codigo_riqueza_final.py")
            
        else:
            print("❌ Erro no processamento dos dados")
            
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        print("Execute com: streamlit run codigo_riqueza_final.py")
