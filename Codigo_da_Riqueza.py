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
import os
import time
import requests
from typing import Optional, Dict, List, Tuple
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO AVANÇADA DO PROJETO ---
INDICADORES = {
    # Indicadores Econômicos Fundamentais
    "NY.GDP.PCAP.KD": "PIB_per_capita",
    "NE.GDI.FTOT.CD": "Formacao_Bruta_Capital", 
    "NY.GNP.PCAP.CD": "Renda_Nacional_Bruta_per_Capita",
    "NE.EXP.GNFS.CD": "Valor_Exportacoes",
    "NE.CON.PRVT.CD": "Consumo_Familias",
    "BX.KLT.DINV.CD.WD": "Investimento_Estrangeiro_Direto",
    "FP.CPI.TOTL.ZG": "Inflacao_Anual_Consumidor",
    "GC.DOD.TOTL.GD.ZS": "Divida_Governo_Central_perc_PIB",
    
    # Indicadores de Capital Humano
    "SE.ADT.1524.LT.ZS": "Alfabetizacao_Jovens",
    "SE.PRM.CMPT.ZS": "Conclusao_Ensino_Primario",
    "SE.XPD.TOTL.GD.ZS": "Gastos_Governamentais_Educacao",
    "SP.DYN.LE00.IN": "Expectativa_de_Vida",
    
    # Indicadores de Trabalho e Produtividade
    "SL.TLF.CACT.ZS": "Participacao_Forca_Trabalho",
    "SL.UEM.TOTL.ZS": "Desemprego",
    
    # Indicadores de Infraestrutura e Tecnologia
    "IT.NET.USER.ZS": "Cobertura_Internet",
    "EG.ELC.ACCS.ZS": "Acesso_Eletricidade",
    "SH.H2O.BASW.ZS": "Cobertura_Agua_Potavel",
    "TX.VAL.TECH.MF.ZS": "Exportacoes_Alta_Tecnologia_perc",
    
    # Indicadores de Governança e Distribuição
    "SI.POV.GINI": "Gini",
    "IQ.CPA.BREG.XQ": "Qualidade_Regulatoria",
    "MS.MIL.XPND.GD.ZS": "Gastos_Militares_perc_PIB"
}

# Grupos de países para análise comparativa
ZONA_DO_EURO = ['DEU', 'FRA', 'ITA', 'ESP', 'PRT', 'GRC', 'IRL', 'NLD', 'AUT', 'BEL']
BRICS = ['BRA', 'RUS', 'IND', 'CHN', 'ZAF', 'EGY', 'ETH', 'IRN', 'SAU', 'ARE']      
PAISES_SUL_AMERICA = ['BRA', 'ARG', 'CHL', 'COL', 'PER', 'ECU', 'VEN', 'BOL', 'PRY', 'URY']
PAISES_SUDESTE_ASIATICO = ['IDN', 'THA', 'VNM', 'PHL', 'MYS', 'SGP', 'MMR', 'KHM', 'LAO', 'BRN']
TIGRES_ASIATICOS = ['KOR', 'TWN', 'HKG', 'SGP']
ECONOMIAS_AVANCADAS = ['USA', 'JPN', 'GBR', 'CAN', 'AUS', 'CHE', 'NOR', 'SWE', 'DNK']

TODOS_PAISES = list(set(
    PAISES_SUL_AMERICA + PAISES_SUDESTE_ASIATICO + BRICS + 
    ZONA_DO_EURO + TIGRES_ASIATICOS + ECONOMIAS_AVANCADAS
))

DATA_INICIO = datetime(1990, 1, 1)
DATA_FIM = datetime(2025, 6, 30)

# Cache global para evitar múltiplas requisições
_cached_data = None
_cached_models = None
_cache_timestamp = None
CACHE_DURATION = 3600  # 1 hora

class WorldBankDataLoader:
    """Classe para carregar dados do Banco Mundial com sistema robusto de cache e rate limiting"""
    
    def __init__(self):
        self.cache_file = "dados_banco_mundial_cache.csv"
        self.metadata_file = "cache_metadata.json"
        self.max_retries = 3
        self.base_delay = 2
        
    def load_with_cache_and_retry(self) -> Optional[pd.DataFrame]:
        """Carrega dados com sistema de cache local e retry automático"""
        global _cached_data, _cache_timestamp
        
        # Verificar cache em memória
        if _cached_data is not None and _cache_timestamp is not None:
            if time.time() - _cache_timestamp < CACHE_DURATION:
                st.info("📋 Dados carregados do cache em memória")
                return _cached_data
        
        # Tentar carregar do cache local
        try:
            if self._is_cache_valid():
                st.info("💾 Carregando dados do cache local...")
                df_cached = pd.read_csv(self.cache_file)
                # Verificar se tem as colunas corretas
                if 'country' in df_cached.columns and 'date' in df_cached.columns:
                    df_cached = df_cached.set_index(['country', 'date'])
                    _cached_data = df_cached
                    _cache_timestamp = time.time()
                    st.success("✅ Dados carregados do cache local com sucesso!")
                    return _cached_data
        except Exception as e:
            st.warning(f"⚠️ Erro ao ler cache local: {e}")
        
        # Baixar dados da API com retry
        return self._download_with_retry()
    
    def _is_cache_valid(self) -> bool:
        """Verifica se o cache local é válido (não mais que 24 horas)"""
        if not os.path.exists(self.cache_file):
            return False
        
        file_age = time.time() - os.path.getmtime(self.cache_file)
        return file_age < 86400  # 24 horas
    
    def _download_with_retry(self) -> Optional[pd.DataFrame]:
        """Baixa dados da API com sistema de retry exponencial"""
        global _cached_data, _cache_timestamp
        
        for attempt in range(self.max_retries):
            try:
                delay = self.base_delay * (2 ** attempt)
                if attempt > 0:
                    st.warning(f"🔄 Tentativa {attempt + 1}/{self.max_retries} após {delay}s...")
                    time.sleep(delay)
                
                st.info("🌐 Baixando dados do Banco Mundial...")
                df_raw = wbdata.get_dataframe(
                    indicators=INDICADORES, 
                    country=TODOS_PAISES, 
                    date=(DATA_INICIO, DATA_FIM)
                )
                
                # Salvar no cache
                df_to_save = df_raw.reset_index()
                df_to_save.to_csv(self.cache_file, index=False)
                
                _cached_data = df_raw
                _cache_timestamp = time.time()
                
                st.success("✅ Dados baixados e salvos no cache com sucesso!")
                return _cached_data
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    st.error(f"🚫 Rate limit atingido (Tentativa {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        continue
                else:
                    st.error(f"❌ Erro na API (Tentativa {attempt + 1}): {e}")
                    
                if attempt == self.max_retries - 1:
                    st.error("❌ Falha após todas as tentativas. Verifique sua conexão e tente novamente mais tarde.")
                    return None
        
        return None

class DataProcessor:
    """Classe para processar e limpar os dados com estratégias avançadas de imputação"""
    
    def __init__(self):
        self.quality_report = {}
    
    def process_data(self, df_raw: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Processa os dados com estratégia multinível de imputação"""
        if df_raw is None:
            return None, None
            
        df = df_raw.copy().reset_index()
        
        # Padronizar nomes das colunas PRIMEIRO
        df = self._standardize_columns(df)
        
        # Verificar se as colunas foram padronizadas corretamente
        if 'País' not in df.columns or 'Ano' not in df.columns:
            st.error(f"❌ Erro na padronização das colunas. Colunas disponíveis: {list(df.columns)}")
            return None, None
        
        df = df.sort_values(by=['País', 'Ano'])
        
        # Relatório de qualidade inicial
        self._generate_quality_report(df, "inicial")
        
        # Aplicar estratégia de imputação multinível
        df_processed = self._apply_multilevel_imputation(df)
        
        # Relatório de qualidade final
        self._generate_quality_report(df_processed, "final")
        
        # Preparar dados para modelagem
        df_model = self._prepare_for_modeling(df_processed)
        
        return df_processed, df_model
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes das colunas"""
        df = df.copy()
        
        # Debug: mostrar colunas originais
        st.info(f"🔍 Colunas originais: {list(df.columns)}")
        
        # Renomear colunas do índice
        if 'country' in df.columns:
            df.rename(columns={'country': 'País'}, inplace=True)
        if 'date' in df.columns:
            df.rename(columns={'date': 'Ano'}, inplace=True)
            df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce')
        
        # Debug: mostrar colunas após padronização
        st.info(f"🔍 Colunas padronizadas: {list(df.columns)}")
        
        return df
    
    def _generate_quality_report(self, df: pd.DataFrame, stage: str):
        """Gera relatório de qualidade dos dados"""
        numeric_cols = [col for col in df.columns if col not in ['País', 'Ano']]
        
        total_points = len(df) * len(numeric_cols)
        missing_points = df[numeric_cols].isnull().sum().sum()
        missing_percentage = (missing_points / total_points) * 100 if total_points > 0 else 0
        
        self.quality_report[stage] = {
            'total_countries': df['País'].nunique(),
            'total_observations': len(df),
            'missing_points': missing_points,
            'missing_percentage': missing_percentage,
            'period': f"{df['Ano'].min():.0f}-{df['Ano'].max():.0f}" if 'Ano' in df.columns else "N/A"
        }
    
    def _apply_multilevel_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica estratégia multinível de imputação para maximizar retenção de países"""
        
        indicadores = [col for col in df.columns if col not in ['País', 'Ano']]
        
        def process_country_group(group):
            """Processa um país específico com múltiplas estratégias"""
            country_data = group.set_index('Ano')[indicadores]
            
            # 1. Interpolação linear (melhor para séries temporais)
            country_data = country_data.interpolate(method='linear', limit_direction='both')
            
            # 2. Forward fill para dados iniciais
            country_data = country_data.ffill()
            
            # 3. Backward fill para dados finais
            country_data = country_data.bfill()
            
            # 4. Para gaps muito grandes, usar interpolação polinomial suave
            for col in country_data.columns:
                if country_data[col].isnull().sum() > 0:
                    try:
                        country_data[col] = country_data[col].interpolate(method='polynomial', order=2)
                    except:
                        pass
            
            return country_data.reset_index()
        
        # Aplicar processamento por país
        df_processed = df.groupby('País', group_keys=False).apply(process_country_group)
        
        # Adicionar coluna País de volta
        df_processed['País'] = df.groupby('País')['País'].first().repeat(df.groupby('País').size()).values
        
        # Como último recurso, preenchimento com mediana regional
        df_processed = self._fill_with_regional_median(df_processed)
        
        # Remover outliers extremos
        df_processed = self._remove_extreme_outliers(df_processed)
        
        return df_processed
    
    def _fill_with_regional_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche dados faltantes com mediana regional"""
        df = df.copy()
        
        # Mapeamento de países para regiões
        region_mapping = {}
        for country in ZONA_DO_EURO:
            region_mapping[country] = 'Europa'
        for country in BRICS:
            region_mapping[country] = 'BRICS'
        for country in PAISES_SUL_AMERICA:
            region_mapping[country] = 'América do Sul'
        for country in PAISES_SUDESTE_ASIATICO:
            region_mapping[country] = 'Sudeste Asiático'
        
        df['Região'] = df['País'].map(region_mapping).fillna('Outros')
        
        numeric_cols = [col for col in df.columns if col not in ['País', 'Ano', 'Região']]
        
        for col in numeric_cols:
            # Calcular mediana por região e ano
            median_by_region_year = df.groupby(['Região', 'Ano'])[col].median()
            
            # Preencher valores faltantes
            mask = df[col].isnull()
            for idx in df[mask].index:
                region = df.loc[idx, 'Região']
                year = df.loc[idx, 'Ano']
                
                if (region, year) in median_by_region_year:
                    df.loc[idx, col] = median_by_region_year[(region, year)]
                else:
                    # Fallback: mediana global da região
                    regional_median = df[df['Região'] == region][col].median()
                    if not pd.isna(regional_median):
                        df.loc[idx, col] = regional_median
                    else:
                        # Último recurso: mediana global
                        df.loc[idx, col] = df[col].median()
        
        df.drop(columns=['Região'], inplace=True)
        
        # Preencher zeros finais se ainda houver NaNs
        df.fillna(0, inplace=True)
        
        return df
    
    def _remove_extreme_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers extremos usando método IQR por indicador"""
        df = df.copy()
        
        numeric_cols = [col for col in df.columns if col not in ['País', 'Ano']]
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Winsorização ao invés de remoção
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _prepare_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados para modelagem com engenharia de features"""
        df_model = df.copy().set_index(['País', 'Ano'])
        
        # Remover PIB per capita dos preditores
        if 'PIB_per_capita' in df_model.columns:
            target = df_model['PIB_per_capita']
            predictors_df = df_model.drop(columns=['PIB_per_capita'])
        else:
            st.error("❌ PIB_per_capita não encontrado!")
            return None
        
        # Criar variáveis lag
        for var in predictors_df.columns:
            df_model[f'{var}_lag1'] = predictors_df.groupby('País')[var].shift(1)
            # Lag de 2 anos para algumas variáveis importantes
            if var in ['Formacao_Bruta_Capital', 'Alfabetizacao_Jovens', 'Cobertura_Internet']:
                df_model[f'{var}_lag2'] = predictors_df.groupby('País')[var].shift(2)
        
        # Criar variáveis de crescimento
        for var in ['Formacao_Bruta_Capital', 'Valor_Exportacoes', 'Consumo_Familias']:
            if var in predictors_df.columns:
                growth_var = f'{var}_growth'
                df_model[growth_var] = predictors_df.groupby('País')[var].pct_change()
                df_model[f'{growth_var}_lag1'] = df_model.groupby('País')[growth_var].shift(1)
        
        # Remover linhas com NaN após criar lags
        df_model_clean = df_model.dropna()
        
        return df_model_clean
    
    def get_quality_report(self) -> Dict:
        """Retorna relatório de qualidade dos dados"""
        return self.quality_report

class ModelTrainer:
    """Classe para treinar e avaliar múltiplos modelos de machine learning"""
    
    def __init__(self):
        self.models_config = {
            "Regressão Linear": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Lasso Regression": Lasso(alpha=0.1, random_state=42),
            "Árvore de Decisão": DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_split=10, 
                random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6, 
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
        }
    
    def train_and_evaluate(self, df_model: pd.DataFrame) -> Dict:
        """Treina todos os modelos e retorna resultados comparativos"""
        global _cached_models
        
        if _cached_models is not None:
            return _cached_models
        
        TARGET = 'PIB_per_capita'
        PREDICTORS = [col for col in df_model.columns if col != TARGET and 
                     ('_lag1' in col or '_lag2' in col or '_growth_lag1' in col)]
        
        X = df_model[PREDICTORS]
        y = df_model[TARGET]
        
        # Normalizar features para alguns modelos
        X_normalized = (X - X.mean()) / X.std()
        
        results = []
        trained_models = {}
        
        progress_bar = st.progress(0)
        
        for i, (name, model) in enumerate(self.models_config.items()):
            st.info(f"🤖 Treinando modelo: {name}")
            
            try:
                # Usar dados normalizados para modelos lineares
                if name in ["Regressão Linear", "Ridge Regression", "Lasso Regression"]:
                    model.fit(X_normalized, y)
                    y_pred = model.predict(X_normalized)
                else:
                    model.fit(X, y)
                    y_pred = model.predict(X)
                
                # Calcular métricas
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                
                # MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y - y_pred) / y)) * 100
                
                results.append({
                    'Modelo': name,
                    'R²': round(r2, 4),
                    'RMSE': round(rmse, 2),
                    'MAE': round(mae, 2),
                    'MAPE': round(mape, 2)
                })
                
                trained_models[name] = model
                
                # Salvar importância das features
                if hasattr(model, 'feature_importances_'):
                    importance = pd.Series(model.feature_importances_, index=PREDICTORS)
                    trained_models[f"{name}_importance"] = importance.sort_values(ascending=False)
                elif hasattr(model, 'coef_'):
                    # Para modelos lineares, usar valor absoluto dos coeficientes
                    importance = pd.Series(np.abs(model.coef_), index=PREDICTORS)
                    trained_models[f"{name}_importance"] = importance.sort_values(ascending=False)
                
            except Exception as e:
                st.error(f"❌ Erro ao treinar {name}: {e}")
                continue
            
            progress_bar.progress((i + 1) / len(self.models_config))
        
        progress_bar.empty()
        
        results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
        
        _cached_models = {
            'resultados': results_df,
            'modelos': trained_models,
            'X': X,
            'y': y,
            'predictors': PREDICTORS,
            'X_normalized': X_normalized
        }
        
        return _cached_models

# --- APLICAÇÃO STREAMLIT PRINCIPAL ---
def main():
    st.set_page_config(
        page_title="Código da Riqueza - Análise Econométrica Avançada", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ O Código da Riqueza — Análise Econométrica Avançada")
    st.markdown("""
    ### 📈 Sistema Avançado de Análise e Projeção Econômica
    
    Utilizando dados do Banco Mundial e técnicas avançadas de machine learning para 
    análise preditiva do PIB per capita com base em 21 indicadores econômicos fundamentais.
    
    **🔧 Recursos desta versão:**
    - ✅ Sistema robusto de cache e retry para API do Banco Mundial
    - ✅ Estratégia multinível de imputação de dados
    - ✅ 6 modelos de ML com avaliação comparativa
    - ✅ Análise de importância das variáveis
    - ✅ Interface interativa completa
    """)
    
    # Inicialização dos dados
    if not all(key in st.session_state for key in ['df', 'df_model', 'quality_report']):
        with st.spinner("🔄 Inicializando sistema e carregando dados..."):
            
            # Carregar dados
            loader = WorldBankDataLoader()
            df_raw = loader.load_with_cache_and_retry()
            
            if df_raw is None:
                st.error("❌ Falha crítica ao carregar dados do Banco Mundial")
                st.stop()
            
            # Processar dados
            processor = DataProcessor()
            df, df_model = processor.process_data(df_raw)
            
            if df is None or df_model is None:
                st.error("❌ Falha no processamento dos dados")
                st.stop()
            
            # Armazenar na sessão
            st.session_state.df = df
            st.session_state.df_model = df_model
            st.session_state.quality_report = processor.get_quality_report()
    
    df = st.session_state.df
    df_model = st.session_state.df_model
    quality_report = st.session_state.quality_report
    
    # --- SEÇÃO: RELATÓRIO DE QUALIDADE DOS DADOS ---
    st.header("📊 Relatório de Qualidade dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Países Analisados", 
            quality_report['final']['total_countries'],
            delta=None
        )
    
    with col2:
        st.metric(
            "Total de Observações", 
            f"{quality_report['final']['total_observations']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Período de Análise", 
            quality_report['final']['period'],
            delta=None
        )
    
    with col4:
        initial_missing = quality_report['inicial']['missing_percentage']
        final_missing = quality_report['final']['missing_percentage']
        st.metric(
            "Dados Imputados", 
            f"{final_missing:.1f}%",
            delta=f"{final_missing - initial_missing:.1f}%"
        )
    
    if final_missing > 0:
        st.info(f"""
        📋 **Relatório de Imputação**: {initial_missing:.1f}% dos dados originais estavam faltantes. 
        Após aplicar estratégias multinível de imputação (interpolação, preenchimento regional, 
        e fallbacks), conseguimos reduzir para {final_missing:.1f}% e manter 
        {quality_report['final']['total_countries']} países na análise.
        """)
    else:
        st.success("✅ Todos os dados foram processados com sucesso, sem necessidade de imputação significativa!")
    
    # Treinar modelos
    if 'models_data' not in st.session_state:
        with st.spinner("🤖 Treinando e avaliando modelos de machine learning..."):
            trainer = ModelTrainer()
            models_data = trainer.train_and_evaluate(df_model)
            st.session_state.models_data = models_data
    
    models_data = st.session_state.models_data
    
    # --- SEÇÃO: COMPARAÇÃO DE MODELOS ---
    st.header("🎯 Comparação de Modelos de Machine Learning")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📈 Performance Comparativa")
        
        results_df = models_data['resultados']
        
        # Gráfico de barras comparativo
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R²
        axes[0,0].bar(results_df['Modelo'], results_df['R²'], color='skyblue', alpha=0.8)
        axes[0,0].set_title('R² Score (maior é melhor)', fontweight='bold')
        axes[0,0].set_ylabel('R²')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # RMSE
        axes[0,1].bar(results_df['Modelo'], results_df['RMSE'], color='lightcoral', alpha=0.8)
        axes[0,1].set_title('RMSE (menor é melhor)', fontweight='bold')
        axes[0,1].set_ylabel('RMSE (US$)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # MAE
        axes[1,0].bar(results_df['Modelo'], results_df['MAE'], color='lightgreen', alpha=0.8)
        axes[1,0].set_title('MAE (menor é melhor)', fontweight='bold')
        axes[1,0].set_ylabel('MAE (US$)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # MAPE
        axes[1,1].bar(results_df['Modelo'], results_df['MAPE'], color='gold', alpha=0.8)
        axes[1,1].set_title('MAPE % (menor é melhor)', fontweight='bold')
        axes[1,1].set_ylabel('MAPE (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🏆 Ranking dos Modelos")
        
        # Adicionar interpretação das métricas
        for i, row in results_df.iterrows():
            if i == 0:  # Melhor modelo
                st.success(f"""
                **🥇 {row['Modelo']}**  
                R²: {row['R²']:.4f} | RMSE: ${row['RMSE']:,.0f} | MAE: ${row['MAE']:,.0f} | MAPE: {row['MAPE']:.1f}%
                """)
            elif i == 1:  # Segundo melhor
                st.info(f"""
                **🥈 {row['Modelo']}**  
                R²: {row['R²']:.4f} | RMSE: ${row['RMSE']:,.0f} | MAE: ${row['MAE']:,.0f} | MAPE: {row['MAPE']:.1f}%
                """)
            elif i == 2:  # Terceiro melhor
                st.warning(f"""
                **🥉 {row['Modelo']}**  
                R²: {row['R²']:.4f} | RMSE: ${row['RMSE']:,.0f} | MAE: ${row['MAE']:,.0f} | MAPE: {row['MAPE']:.1f}%
                """)
            else:
                st.text(f"""
                {row['Modelo']}  
                R²: {row['R²']:.4f} | RMSE: ${row['RMSE']:,.0f} | MAE: ${row['MAE']:,.0f} | MAPE: {row['MAPE']:.1f}%
                """)
        
        best_model = results_df.iloc[0]
        st.markdown(f"""
        ---
        ### 🎯 Interpretação do Melhor Modelo
        
        **{best_model['Modelo']}** explica **{best_model['R²']:.1%}** da variação do PIB per capita.
        
        - **Erro médio**: ±${best_model['RMSE']:,.0f}
        - **Erro absoluto**: ${best_model['MAE']:,.0f}  
        - **Erro percentual**: {best_model['MAPE']:.1f}%
        """)
    
    # --- SELEÇÃO DO MODELO ---
    st.subheader("⚙️ Configuração do Modelo para Análises")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_model_name = st.selectbox(
            "Selecione o modelo para análises:",
            options=results_df['Modelo'].tolist(),
            index=0,
            help="O melhor modelo é selecionado por padrão"
        )
        
        selected_model = models_data['modelos'][selected_model_name]
        
        # Mostrar importância das variáveis
        importance_key = f"{selected_model_name}_importance"
        if importance_key in models_data['modelos']:
            st.subheader("📊 Top 10 Variáveis Mais Importantes")
            
            importance = models_data['modelos'][importance_key]
            top_10 = importance.head(10)
            
            # Gráfico de importância
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(top_10)), top_10.values, color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels([var.replace('_lag1', '').replace('_lag2', '').replace('_growth_lag1', ' (crescimento)').replace('_', ' ').title() 
                               for var in top_10.index])
            ax.set_xlabel('Importância Relativa')
            ax.set_title(f'Importância das Variáveis - {selected_model_name}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Adicionar valores nas barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        # Métricas detalhadas do modelo selecionado
        model_info = results_df[results_df['Modelo'] == selected_model_name].iloc[0]
        
        st.subheader(f"📋 Detalhes do Modelo: {selected_model_name}")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("R² Score", f"{model_info['R²']:.4f}")
        with col_b:
            st.metric("RMSE", f"${model_info['RMSE']:,.0f}")
        with col_c:
            st.metric("MAE", f"${model_info['MAE']:,.0f}")
        with col_d:
            st.metric("MAPE", f"{model_info['MAPE']:.1f}%")
        
        # Interpretação contextualizada
        if model_info['R²'] >= 0.8:
            performance_level = "Excelente"
            performance_color = "success"
        elif model_info['R²'] >= 0.6:
            performance_level = "Bom"
            performance_color = "info"
        else:
            performance_level = "Limitado"
            performance_color = "warning"
        
        getattr(st, performance_level.lower() if performance_level != "Limitado" else "warning")(f"""
        **Performance: {performance_level}**
        
        Este modelo tem poder explicativo **{performance_level.lower()}** para prever o PIB per capita.
        Com erro médio de ${model_info['MAE']:,.0f}, as projeções têm precisão de 
        ±{model_info['MAPE']:.1f}% em média.
        """)
        
        # Informações do dataset
        st.info(f"""
        **📈 Dataset utilizado:**
        - **Observações**: {len(models_data['y']):,}
        - **Variáveis preditoras**: {len(models_data['predictors'])}
        - **Países**: {df['País'].nunique()}
        - **Período**: {df['Ano'].min():.0f}-{df['Ano'].max():.0f}
        """)
    
    # --- ANÁLISE POR PAÍS ---
    st.header("🌍 Análise Econômica por País")
    
    # Sidebar com filtros
    st.sidebar.header("🔍 Configurações de Análise")
    
    available_countries = sorted(df['País'].unique())
    selected_country = st.sidebar.selectbox(
        "Selecionar país para análise:",
        available_countries,
        help="Escolha o país para análise detalhada"
    )
    
    # Filtros de ano
    available_years = sorted(df[df['País'] == selected_country]['Ano'].unique())
    if len(available_years) > 1:
        year_start, year_end = st.sidebar.select_slider(
            "Período histórico:",
            options=available_years,
            value=(available_years[0], available_years[-1]),
            help="Defini o intervalo de anos para análise histórica"
        )
    else:
        year_start = year_end = available_years[0]
    
    # Dados filtrados
    df_filtered = df[
        (df['País'] == selected_country) & 
        (df['Ano'].between(year_start, year_end))
    ]
    
    # --- VISUALIZAÇÃO HISTÓRICA ---
    st.subheader(f"📈 Evolução Histórica — {selected_country} ({year_start:.0f}–{year_end:.0f})")
    
    # Seletor de indicador
    available_indicators = [col for col in df.columns if col not in ['País', 'Ano']]
    selected_indicator = st.selectbox(
        "Indicador para visualização:",
        available_indicators,
        index=0 if 'PIB_per_capita' in available_indicators else 0,
        help="Escolha o indicador econômico para visualizar sua evolução"
    )
    
    if not df_filtered.empty:
        # Gráfico de linha com área
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_data = df_filtered['Ano']
        y_data = df_filtered[selected_indicator]
        
        # Linha principal
        ax.plot(x_data, y_data, marker='o', linewidth=3, markersize=6, 
                color='steelblue', alpha=0.9, label=selected_indicator.replace('_', ' ').title())
        
        # Área sob a curva
        ax.fill_between(x_data, y_data, alpha=0.3, color='steelblue')
        
        # Linha de tendência
        if len(x_data) > 2:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "--", color='red', alpha=0.8, linewidth=2, label='Tendência')
        
        ax.set_title(f'{selected_indicator.replace("_", " ").title()} — {selected_country}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Ano', fontsize=12)
        ax.set_ylabel(selected_indicator.replace('_', ' ').title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar anotações para valores máximo e mínimo
        max_idx = y_data.idxmax()
        min_idx = y_data.idxmin()
        
        ax.annotate(f'Máx: {y_data.iloc[max_idx]:,.0f}', 
                   xy=(x_data.iloc[max_idx], y_data.iloc[max_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Mín: {y_data.iloc[min_idx]:,.0f}', 
                   xy=(x_data.iloc[min_idx], y_data.iloc[min_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Estatísticas resumidas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Valor Atual", f"{y_data.iloc[-1]:,.0f}")
        with col2:
            growth_rate = ((y_data.iloc[-1] / y_data.iloc[0]) ** (1/(len(y_data)-1)) - 1) * 100 if len(y_data) > 1 else 0
            st.metric("Crescimento Anual Médio", f"{growth_rate:+.1f}%")
        with col3:
            st.metric("Máximo Histórico", f"{y_data.max():,.0f}")
        with col4:
            st.metric("Mínimo Histórico", f"{y_data.min():,.0f}")
    
    else:
        st.warning("⚠️ Nenhum dado disponível para os filtros selecionados")
    
    # --- PROJEÇÕES ECONÔMICAS ---
    st.header("🔮 Projeções Econômicas Avançadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        projection_year = st.selectbox(
            "Projetar até o ano:",
            [2030, 2035, 2040, 2045],
            index=1,
            help="Ano final para as projeções econômicas"
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Tipo de análise:",
            ["Cenário Único", "Múltiplos Cenários", "Análise de Sensibilidade"],
            help="Escolha o tipo de análise preditiva"
        )
    
    with col3:
        show_confidence = st.checkbox(
            "Mostrar intervalos de confiança",
            value=True,
            help="Exibir bandas de incerteza nas projeções"
        )
    
    if st.button("🚀 Gerar Projeções Avançadas", type="primary"):
        
        if selected_country not in df_model.reset_index()['País'].values:
            st.error(f"❌ Dados insuficientes para modelagem de {selected_country}")
        else:
            
            with st.spinner(f"🔄 Gerando projeções para {selected_country}..."):
                
                try:
                    
                    if analysis_type == "Cenário Único":
                        # Projeção única
                        projection_engine = ProjectionEngine(df_model, selected_model, selected_model_name)
                        df_projection = projection_engine.generate_scenario_projection(
                            selected_country, projection_year
                        )
                        
                        last_real_year = int(df_model.reset_index()['Ano'].max())
                        df_historical = df_projection[df_projection['Ano'] <= last_real_year]
                        df_future = df_projection[df_projection['Ano'] > last_real_year]
                        
                        # Gráfico da projeção
                        fig, ax = plt.subplots(figsize=(14, 8))
                        
                        # Dados históricos
                        ax.plot(df_historical['Ano'], df_historical['PIB_per_capita'], 
                               'o-', label='Dados Históricos', linewidth=3, color='steelblue', markersize=6)
                        
                        # Projeções futuras
                        if not df_future.empty:
                            ax.plot(df_future['Ano'], df_future['PIB_per_capita'], 
                                   's--', label=f'Projeções ({selected_model_name})', 
                                   linewidth=3, color='crimson', alpha=0.9, markersize=6)
                            
                            # Intervalo de confiança se solicitado
                            if show_confidence and len(df_future) > 1:
                                # Calcular banda de incerteza (simulação simples)
                                uncertainty = 0.1  # ±10% de incerteza
                                upper_bound = df_future['PIB_per_capita'] * (1 + uncertainty)
                                lower_bound = df_future['PIB_per_capita'] * (1 - uncertainty)
                                
                                ax.fill_between(df_future['Ano'], lower_bound, upper_bound, 
                                              alpha=0.3, color='crimson', label='Intervalo de Confiança (±10%)')
                        
                        ax.set_title(f'Projeção PIB per capita — {selected_country}', 
                                    fontsize=16, fontweight='bold')
                        ax.set_xlabel('Ano', fontsize=12)
                        ax.set_ylabel('PIB per capita (US$)', fontsize=12)
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        
                        # Formatação do eixo Y
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Métricas da projeção
                        if not df_future.empty:
                            current_gdp = df_historical['PIB_per_capita'].iloc[-1]
                            final_gdp = df_future['PIB_per_capita'].iloc[-1]
                            total_growth = ((final_gdp / current_gdp) - 1) * 100
                            projection_years = len(df_future)
                            annual_growth = (((final_gdp / current_gdp) ** (1/projection_years)) - 1) * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("PIB Atual", f"${current_gdp:,.0f}")
                            with col2:
                                st.metric("PIB Projetado", f"${final_gdp:,.0f}", 
                                         f"{total_growth:+.1f}%")
                            with col3:
                                st.metric("Crescimento Anual", f"{annual_growth:.1f}%")
                            with col4:
                                years_to_double = 70 / annual_growth if annual_growth > 0 else float('inf')
                                if years_to_double < 100:
                                    st.metric("Anos para Dobrar", f"{years_to_double:.0f} anos")
                                else:
                                    st.metric("Anos para Dobrar", "100+ anos")
                            
                            # Classificação do crescimento
                            if annual_growth > 5:
                                st.success("🚀 **Crescimento Alto**: Projeção indica expansão econômica robusta!")
                            elif annual_growth > 2:
                                st.info("📈 **Crescimento Moderado**: Expansão econômica sustentável projetada.")
                            elif annual_growth > 0:
                                st.warning("📊 **Crescimento Baixo**: Expansão econômica limitada projetada.")
                            else:
                                st.error("📉 **Contração Econômica**: Declínio econômico projetado. Políticas de estímulo necessárias.")
                    
                    elif analysis_type == "Múltiplos Cenários":
                        # Múltiplos cenários
                        scenarios = generate_multiple_scenarios(
                            df_model, selected_country, selected_model, selected_model_name, projection_year
                        )
                        
                        fig, ax = plt.subplots(figsize=(14, 8))
                        
                        colors = {'Pessimista': '#e74c3c', 'Realista': '#3498db', 'Otimista': '#27ae60'}
                        last_real_year = int(df_model.reset_index()['Ano'].max())
                        
                        # Dados históricos (usar cenário realista)
                        df_hist = scenarios['Realista'][scenarios['Realista']['Ano'] <= last_real_year]
                        ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                               'o-', label='Histórico', linewidth=4, color='black', markersize=6)
                        
                        # Plotar cada cenário
                        scenario_data = {}
                        for scenario_name, df_scenario in scenarios.items():
                            df_proj = df_scenario[df_scenario['Ano'] > last_real_year]
                            if not df_proj.empty:
                                ax.plot(df_proj['Ano'], df_proj['PIB_per_capita'], 
                                       's--', label=f'Cenário {scenario_name}', 
                                       linewidth=3, color=colors[scenario_name], alpha=0.9, markersize=5)
                                
                                # Armazenar dados para métricas
                                initial_gdp = df_hist['PIB_per_capita'].iloc[-1]
                                final_gdp = df_proj['PIB_per_capita'].iloc[-1]
                                annual_growth = (((final_gdp / initial_gdp) ** (1/len(df_proj))) - 1) * 100
                                
                                scenario_data[scenario_name] = {
                                    'final_gdp': final_gdp,
                                    'annual_growth': annual_growth
                                }
                        
                        ax.set_title(f'Cenários de Projeção — {selected_country}', 
                                    fontsize=16, fontweight='bold')
                        ax.set_xlabel('Ano', fontsize=12)
                        ax.set_ylabel('PIB per capita (US$)', fontsize=12)
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Tabela comparativa de cenários
                        st.subheader("📊 Comparação de Cenários")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        for i, (scenario, data) in enumerate(scenario_data.items()):
                            with [col1, col2, col3][i]:
                                color = 'success' if scenario == 'Otimista' else ('warning' if scenario == 'Pessimista' else 'info')
                                getattr(st, color)(f"""
                                **Cenário {scenario}**
                                
                                PIB Final: ${data['final_gdp']:,.0f}
                                
                                Crescimento: {data['annual_growth']:+.1f}% a.a.
                                """)
                        
                        # Análise de risco
                        if scenario_data:
                            best_case = max(scenario_data.values(), key=lambda x: x['final_gdp'])['final_gdp']
                            worst_case = min(scenario_data.values(), key=lambda x: x['final_gdp'])['final_gdp']
                            risk_spread = ((best_case - worst_case) / worst_case) * 100
                            
                            st.info(f"""
                            📈 **Análise de Risco**: A diferença entre o melhor e pior cenário é de 
                            {risk_spread:.1f}%, indicando {'alto' if risk_spread > 50 else 'moderado' if risk_spread > 25 else 'baixo'} 
                            nível de incerteza nas projeções.
                            """)
                    
                    elif analysis_type == "Análise de Sensibilidade":
                        # Análise de sensibilidade
                        st.subheader("🎯 Configuração da Análise de Sensibilidade")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Indicadores disponíveis para choque
                            base_indicators = [col.replace('_lag1', '').replace('_lag2', '').replace('_growth_lag1', '') 
                                             for col in models_data['predictors']]
                            unique_indicators = list(set(base_indicators))
                            unique_indicators = [ind for ind in unique_indicators if not ind.endswith('_growth')]
                            
                            shock_indicator = st.selectbox(
                                "Indicador para análise:",
                                unique_indicators,
                                help="Selecione o indicador para aplicar choque de sensibilidade"
                            )
                            
                            shock_percentage = st.slider(
                                "Variação percentual:",
                                min_value=-50, max_value=50, value=10, step=5,
                                help="Percentual de choque a ser aplicado no indicador"
                            )
                        
                        with col2:
                            st.info(f"""
                            **Análise de Sensibilidade Avançada**
                            
                            Esta análise aplica um **choque persistente** de {shock_percentage:+}% 
                            no indicador **{shock_indicator.replace('_', ' ')}** ao longo de 
                            toda a projeção.
                            
                            O choque é mantido consistentemente durante todo o período, 
                            mostrando o impacto cumulativo no PIB per capita até {projection_year}.
                            """)
                        
                        if st.button("🔬 Executar Análise de Sensibilidade"):
                            with st.spinner("Calculando impactos..."):
                                
                                # Projeção base (sem choque)
                                projection_engine = ProjectionEngine(df_model, selected_model, selected_model_name)
                                df_base = projection_engine.generate_scenario_projection(
                                    selected_country, projection_year
                                )
                                
                                # Projeção com choque
                                df_shocked = projection_engine.generate_scenario_projection(
                                    selected_country, projection_year, shock_indicator, shock_percentage
                                )
                                
                                last_real_year = int(df_model.reset_index()['Ano'].max())
                                
                                # Gráfico comparativo
                                fig, ax = plt.subplots(figsize=(14, 8))
                                
                                # Dados históricos
                                df_hist = df_base[df_base['Ano'] <= last_real_year]
                                ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                                       'o-', label='Histórico', linewidth=3, color='black', markersize=6)
                                
                                # Projeções
                                df_base_future = df_base[df_base['Ano'] > last_real_year]
                                df_shocked_future = df_shocked[df_shocked['Ano'] > last_real_year]
                                
                                if not df_base_future.empty:
                                    ax.plot(df_base_future['Ano'], df_base_future['PIB_per_capita'], 
                                           's--', label='Projeção Base', linewidth=3, color='steelblue', alpha=0.8)
                                
                                if not df_shocked_future.empty:
                                    ax.plot(df_shocked_future['Ano'], df_shocked_future['PIB_per_capita'], 
                                           '^--', label=f'{shock_indicator.replace("_", " ")} {shock_percentage:+}% (Persistente)', 
                                           linewidth=3, color='red', alpha=0.9)
                                
                                ax.set_title(f'Análise de Sensibilidade — {selected_country}', 
                                            fontsize=16, fontweight='bold')
                                ax.set_xlabel('Ano', fontsize=12)
                                ax.set_ylabel('PIB per capita (US$)', fontsize=12)
                                ax.legend(fontsize=11)
                                ax.grid(True, alpha=0.3)
                                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                                # Cálculo de impactos
                                if not df_base_future.empty and not df_shocked_future.empty:
                                    base_final = df_base_future['PIB_per_capita'].iloc[-1]
                                    shocked_final = df_shocked_future['PIB_per_capita'].iloc[-1]
                                    total_impact = ((shocked_final / base_final) - 1) * 100
                                    
                                    # Impacto médio ao longo do tempo
                                    df_comparison = pd.merge(
                                        df_base_future[['Ano', 'PIB_per_capita']], 
                                        df_shocked_future[['Ano', 'PIB_per_capita']], 
                                        on='Ano', suffixes=('_base', '_shocked')
                                    )
                                    df_comparison['impact_pct'] = ((df_comparison['PIB_per_capita_shocked'] / 
                                                                   df_comparison['PIB_per_capita_base']) - 1) * 100
                                    avg_impact = df_comparison['impact_pct'].mean()
                                    
                                    # Elasticidade
                                    elasticity = total_impact / shock_percentage if shock_percentage != 0 else 0
                                    
                                    # Métricas
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Impacto Final", f"{total_impact:+.2f}%", 
                                                 f"Ano {projection_year}")
                                    
                                    with col2:
                                        st.metric("Impacto Médio", f"{avg_impact:+.2f}%", 
                                                 "Ao longo da projeção")
                                    
                                    with col3:
                                        st.metric("Elasticidade", f"{elasticity:.3f}", 
                                                 "Impacto por 1% de mudança")
                                    
                                    with col4:
                                        # Significância do impacto
                                        if abs(total_impact) > 10:
                                            significance = "Alto"
                                            color = "error"
                                        elif abs(total_impact) > 5:
                                            significance = "Moderado"
                                            color = "warning"
                                        else:
                                            significance = "Baixo"
                                            color = "success"
                                        
                                        st.metric("Significância", significance)
                                    
                                    # Gráfico de evolução do impacto
                                    if len(df_comparison) > 1:
                                        fig2, ax2 = plt.subplots(figsize=(12, 5))
                                        ax2.plot(df_comparison['Ano'], df_comparison['impact_pct'], 
                                                'o-', color='purple', linewidth=2, markersize=5)
                                        ax2.set_title('Evolução do Impacto ao Longo do Tempo', fontweight='bold')
                                        ax2.set_xlabel('Ano')
                                        ax2.set_ylabel('Impacto no PIB (%)')
                                        ax2.grid(True, alpha=0.3)
                                        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                                        plt.tight_layout()
                                        st.pyplot(fig2)
                                        plt.close()
                                    
                                    # Interpretação avançada
                                    st.subheader("📋 Interpretação dos Resultados")
                                    
                                    if abs(elasticity) > 1.5:
                                        st.error(f"""
                                        🚨 **Elasticidade Muito Alta** ({elasticity:.3f})
                                        
                                        O indicador **{shock_indicator.replace('_', ' ')}** tem impacto 
                                        **desproporcional** no PIB. Uma mudança de 1% resulta em 
                                        {abs(elasticity):.2f}% de impacto no PIB per capita.
                                        
                                        **Recomendação**: Este é um indicador crítico que deve ser 
                                        prioridade máxima nas políticas econômicas.
                                        """)
                                    elif abs(elasticity) > 0.5:
                                        st.warning(f"""
                                        ⚠️ **Alta Sensibilidade** ({elasticity:.3f})
                                        
                                        O indicador **{shock_indicator.replace('_', ' ')}** tem 
                                        **grande impacto** no crescimento econômico.
                                        
                                        **Recomendação**: Políticas focadas neste indicador podem 
                                        gerar resultados significativos para o crescimento do PIB.
                                        """)
                                    elif abs(elasticity) > 0.2:
                                        st.info(f"""
                                        📊 **Sensibilidade Moderada** ({elasticity:.3f})
                                        
                                        O indicador **{shock_indicator.replace('_', ' ')}** tem 
                                        impacto **moderado** no PIB per capita.
                                        
                                        **Recomendação**: Importante para estratégias de médio prazo, 
                                        mas não é o fator mais crítico.
                                        """)
                                    else:
                                        st.success(f"""
                                        ✅ **Baixa Sensibilidade** ({elasticity:.3f})
                                        
                                        O indicador **{shock_indicator.replace('_', ' ')}** tem 
                                        **pouco impacto** direto no PIB per capita.
                                        
                                        **Recomendação**: Não é prioritário para políticas de 
                                        crescimento econômico, mas pode ter outros benefícios sociais.
                                        """)
                                else:
                                    st.warning("⚠️ Dados insuficientes para calcular impactos")
                    
                    st.success("✅ Análise concluída com sucesso!")
                    
                except Exception as e:
                    st.error(f"❌ Erro durante a análise: {str(e)}")
                    st.exception(e)
    
    # --- COMPARAÇÃO ENTRE PAÍSES ---
    st.header("🌎 Comparação Internacional")
    
    if st.checkbox("🔄 Ativar comparação entre países"):
        
        # Grupos predefinidos
        country_groups = {
            "Personalizado": [],
            "BRICS": BRICS,
            "Zona do Euro": ZONA_DO_EURO,
            "América do Sul": PAISES_SUL_AMERICA,
            "Sudeste Asiático": PAISES_SUDESTE_ASIATICO,
            "Tigres Asiáticos": TIGRES_ASIATICOS,
            "Economias Avançadas": ECONOMIAS_AVANCADAS
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_group = st.selectbox(
                "Selecionar grupo de países:",
                list(country_groups.keys()),
                help="Escolha um grupo predefinido ou selecione países personalizados"
            )
            
            if selected_group == "Personalizado":
                available_model_countries = sorted(df_model.reset_index()['País'].unique())
                comparison_countries = st.multiselect(
                    "Selecionar países (máximo 6):",
                    available_model_countries,
                    default=[selected_country] if selected_country in available_model_countries else [],
                    help="Escolha até 6 países para comparação"
                )
            else:
                # Filtrar países do grupo que estão disponíveis no modelo
                available_group_countries = [c for c in country_groups[selected_group] 
                                           if c in df_model.reset_index()['País'].values]
                comparison_countries = st.multiselect(
                    f"Países do grupo {selected_group}:",
                    available_group_countries,
                    default=available_group_countries[:5],  # Máximo 5 por padrão
                    help="Selecione os países do grupo para comparação"
                )
        
        with col2:
            comparison_year = st.selectbox(
                "Projetar até:",
                [2030, 2035, 2040],
                index=1,
                help="Ano final para comparação de projeções"
            )
            
            comparison_metric = st.selectbox(
                "Métrica de comparação:",
                ["Crescimento Anual Médio", "PIB Final Projetado", "Crescimento Total"],
                help="Escolha a métrica para ranking dos países"
            )
        
        # Limitar número de países
        if len(comparison_countries) > 6:
            st.warning("⚠️ Máximo 6 países permitidos para comparação.")
            comparison_countries = comparison_countries[:6]
        
        if len(comparison_countries) >= 2 and st.button("🔄 Executar Comparação Internacional"):
            
            with st.spinner("🌍 Analisando países selecionados..."):
                
                try:
                    fig, ax = plt.subplots(figsize=(16, 10))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_countries)))
                    
                    comparison_data = []
                    last_real_year = int(df_model.reset_index()['Ano'].max())
                    projection_engine = ProjectionEngine(df_model, selected_model, selected_model_name)
                    
                    for i, country in enumerate(comparison_countries):
                        try:
                            # Gerar projeção para o país
                            df_country = projection_engine.generate_scenario_projection(
                                country, comparison_year
                            )
                            
                            df_hist = df_country[df_country['Ano'] <= last_real_year]
                            df_proj = df_country[df_country['Ano'] > last_real_year]
                            
                            # Plotar dados históricos
                            if not df_hist.empty:
                                ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 
                                       'o-', color=colors[i], alpha=0.7, linewidth=2, markersize=4)
                            
                            # Plotar projeções
                            if not df_proj.empty:
                                ax.plot(df_proj['Ano'], df_proj['PIB_per_capita'], 
                                       's--', color=colors[i], alpha=0.9, linewidth=3, 
                                       label=country, markersize=5)
                            
                            # Calcular métricas
                            if not df_proj.empty and not df_hist.empty:
                                initial_gdp = df_hist['PIB_per_capita'].iloc[-1]
                                final_gdp = df_proj['PIB_per_capita'].iloc[-1]
                                projection_years = len(df_proj)
                                annual_growth = (((final_gdp / initial_gdp) ** (1/projection_years)) - 1) * 100
                                total_growth = ((final_gdp / initial_gdp) - 1) * 100
                                
                                comparison_data.append({
                                    'País': country,
                                    'PIB Atual (US$)': f"{initial_gdp:,.0f}",
                                    'PIB Projetado (US$)': f"{final_gdp:,.0f}",
                                    'Crescimento Anual Médio (%)': round(annual_growth, 2),
                                    'Crescimento Total (%)': round(total_growth, 1),
                                    'Anos para Dobrar': round(70 / annual_growth, 1) if annual_growth > 0 else float('inf'),
                                    'Modelo': selected_model_name,
                                    # Dados numéricos para ordenação
                                    '_annual_growth_num': annual_growth,
                                    '_final_gdp_num': final_gdp,
                                    '_total_growth_num': total_growth
                                })
                            
                            st.success(f"✅ {country} processado")
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao processar {country}: {str(e)}")
                            continue
                    
                    if comparison_data:
                        # Configurar gráfico
                        ax.set_title(f'Comparação Internacional de Crescimento — {selected_model_name}', 
                                    fontsize=16, fontweight='bold')
                        ax.set_xlabel('Ano', fontsize=12)
                        ax.set_ylabel('PIB per capita (US$)', fontsize=12)
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Tabela de comparação
                        st.subheader("📊 Ranking Internacional")
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        
                        # Ordenar por métrica selecionada
                        if comparison_metric == "Crescimento Anual Médio":
                            df_comparison = df_comparison.sort_values('_annual_growth_num', ascending=False)
                        elif comparison_metric == "PIB Final Projetado":
                            df_comparison = df_comparison.sort_values('_final_gdp_num', ascending=False)
                        else:  # Crescimento Total
                            df_comparison = df_comparison.sort_values('_total_growth_num', ascending=False)
                        
                        # Remover colunas auxiliares
                        display_cols = ['País', 'PIB Atual (US$)', 'PIB Projetado (US$)', 
                                       'Crescimento Anual Médio (%)', 'Crescimento Total (%)', 'Anos para Dobrar']
                        df_display = df_comparison[display_cols].copy()
                        
                        # Formatar coluna "Anos para Dobrar"
                        df_display['Anos para Dobrar'] = df_display['Anos para Dobrar'].apply(
                            lambda x: f"{x:.1f}" if x != float('inf') else "100+"
                        )
                        
                        st.dataframe(df_display, hide_index=True, use_container_width=True)
                        
                        # Destacar top performers
                        if len(df_comparison) > 1:
                            best_performer = df_comparison.iloc[0]
                            worst_performer = df_comparison.iloc[-1]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.success(f"""
                                🏆 **Melhor Performance**
                                
                                **{best_performer['País']}** lidera em {comparison_metric.lower()}
                                
                                • Crescimento anual: {best_performer['Crescimento Anual Médio (%)']}%
                                • PIB final: {best_performer['PIB Projetado (US$)']}
                                """)
                            
                            with col2:
                                st.info(f"""
                                📈 **Maior Potencial de Crescimento**
                                
                                Entre os países analisados, há uma diferença significativa 
                                de {best_performer['_annual_growth_num'] - worst_performer['_annual_growth_num']:.1f} 
                                pontos percentuais entre o maior e menor crescimento projetado.
                                """)
                        
                        # Download dos resultados
                        csv_data = df_display.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Baixar Comparação como CSV",
                            data=csv_data,
                            file_name=f"comparacao_internacional_{comparison_year}.csv",
                            mime='text/csv'
                        )
                    
                    else:
                        st.warning("❌ Nenhum país foi processado com sucesso.")
                        
                except Exception as e:
                    st.error(f"❌ Erro geral na comparação: {str(e)}")
    
    # --- DADOS E DOWNLOADS ---
    st.header("📋 Dados e Análises Complementares")
    
    # Dados filtrados do país selecionado
    st.subheader(f"📊 Dados Históricos — {selected_country}")
    
    if not df_filtered.empty:
        # Mostrar estatísticas resumidas
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Estatísticas Descritivas**")
            numeric_cols = [col for col in df_filtered.columns if col not in ['País', 'Ano']]
            stats_df = df_filtered[numeric_cols].describe().round(2)
            st.dataframe(stats_df)
        
        with col2:
            st.write("**📊 Correlações com PIB per capita**")
            if 'PIB_per_capita' in df_filtered.columns:
                correlations = df_filtered[numeric_cols].corr()['PIB_per_capita'].sort_values(ascending=False)
                correlations = correlations.drop('PIB_per_capita').head(10)
                
                # Gráfico de correlações
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['green' if x > 0 else 'red' for x in correlations.values]
                bars = ax.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
                ax.set_yticks(range(len(correlations)))
                ax.set_yticklabels([col.replace('_', ' ').title() for col in correlations.index])
                ax.set_xlabel('Correlação com PIB per capita')
                ax.set_title('Top 10 Correlações', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                ax.axvline(x=0, color='black', linewidth=0.8)
                
                # Adicionar valores nas barras
                for bar, value in zip(bars, correlations.values):
                    width = bar.get_width()
                    ax.text(width + (0.02 if width > 0 else -0.02), bar.get_y() + bar.get_height()/2, 
                           f'{value:.2f}', ha='left' if width > 0 else 'right', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # Tabela de dados completos
        st.write("**📋 Dados Completos do Período Selecionado**")
        
        # Formatar dados para visualização
        df_display = df_filtered.copy()
        
        # Formatar colunas monetárias
        money_cols = ['PIB_per_capita', 'Renda_Nacional_Bruta_per_Capita', 'Formacao_Bruta_Capital', 
                     'Valor_Exportacoes', 'Consumo_Familias', 'Investimento_Estrangeiro_Direto']
        
        for col in money_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        
        # Formatar colunas de percentual
        pct_cols = ['Alfabetizacao_Jovens', 'Participacao_Forca_Trabalho', 'Cobertura_Internet', 
                   'Acesso_Eletricidade', 'Desemprego', 'Conclusao_Ensino_Primario', 'Cobertura_Agua_Potavel',
                   'Gastos_Governamentais_Educacao', 'Gastos_Militares_perc_PIB', 'Exportacoes_Alta_Tecnologia_perc']
        
        for col in pct_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        st.dataframe(df_display, use_container_width=True)
        
        # Botões de download
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Dados como CSV",
                data=csv_data,
                file_name=f"{selected_country}_dados_historicos.csv",
                mime='text/csv'
            )
        
        with col2:
            # Criar relatório resumido
            report = f"""
RELATÓRIO ECONÔMICO - {selected_country}
Período: {year_start:.0f} - {year_end:.0f}
Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

INDICADORES PRINCIPAIS:
"""
            
            if 'PIB_per_capita' in df_filtered.columns:
                pib_inicial = df_filtered['PIB_per_capita'].iloc[0]
                pib_final = df_filtered['PIB_per_capita'].iloc[-1]
                crescimento_periodo = ((pib_final / pib_inicial) ** (1/(len(df_filtered)-1)) - 1) * 100
                
                report += f"""
• PIB per capita inicial: ${pib_inicial:,.0f}
• PIB per capita final: ${pib_final:,.0f}
• Crescimento anual médio: {crescimento_periodo:.2f}%

"""
            
            # Adicionar outros indicadores importantes
            key_indicators = ['Alfabetizacao_Jovens', 'Desemprego', 'Cobertura_Internet', 'Gini']
            for indicator in key_indicators:
                if indicator in df_filtered.columns:
                    initial_val = df_filtered[indicator].iloc[0]
                    final_val = df_filtered[indicator].iloc[-1]
                    change = final_val - initial_val
                    report += f"• {indicator.replace('_', ' ')}: {initial_val:.1f} → {final_val:.1f} ({change:+.1f})\n"
            
            report += f"""

MODELO UTILIZADO: {selected_model_name}
R² Score: {models_data['resultados'].iloc[0]['R²']:.4f}
RMSE: ${models_data['resultados'].iloc[0]['RMSE']:,.0f}

Relatório gerado pelo Sistema de Análise Econômica "Código da Riqueza"
"""
            
            st.download_button(
                label="📄 Baixar Relatório",
                data=report.encode('utf-8'),
                file_name=f"{selected_country}_relatorio_economico.txt",
                mime='text/plain'
            )
        
        with col3:
            # Exportar dados do modelo
            if st.button("📊 Exportar Dados do Modelo"):
                model_data = df_model.reset_index()
                model_csv = model_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dados Modelo",
                    data=model_csv,
                    file_name="dados_modelo_completo.csv",
                    mime='text/csv'
                )
    
    else:
        st.warning("⚠️ Nenhum dado disponível para o país e período selecionados")
    
    # --- INFORMAÇÕES TÉCNICAS ---
    with st.expander("🔍 Informações Técnicas e Metodologia"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Especificações do Dataset")
            
            st.write(f"""
            **Fonte dos Dados**: Banco Mundial (World Bank Open Data)
            
            **Período de Análise**: {df['Ano'].min():.0f} - {df['Ano'].max():.0f}
            
            **Países Incluídos**: {df['País'].nunique()} países
            
            **Total de Observações**: {len(df):,}
            
            **Indicadores Analisados**: {len([col for col in df.columns if col not in ['País', 'Ano']])}
            
            **Qualidade dos Dados**:
            - Dados originais faltantes: {quality_report['inicial']['missing_percentage']:.1f}%
            - Dados após imputação: {quality_report['final']['missing_percentage']:.1f}%
            """)
            
            st.subheader("🎯 Indicadores Incluídos")
            
            indicators_by_category = {
                "💰 Econômicos": [
                    "PIB_per_capita", "Formacao_Bruta_Capital", "Renda_Nacional_Bruta_per_Capita",
                    "Valor_Exportacoes", "Consumo_Familias", "Investimento_Estrangeiro_Direto",
                    "Inflacao_Anual_Consumidor", "Divida_Governo_Central_perc_PIB"
                ],
                "🎓 Capital Humano": [
                    "Alfabetizacao_Jovens", "Conclusao_Ensino_Primario", 
                    "Gastos_Governamentais_Educacao", "Expectativa_de_Vida"
                ],
                "💼 Trabalho": [
                    "Participacao_Forca_Trabalho", "Desemprego"
                ],
                "🏗️ Infraestrutura": [
                    "Cobertura_Internet", "Acesso_Eletricidade", "Cobertura_Agua_Potavel",
                    "Exportacoes_Alta_Tecnologia_perc"
                ],
                "⚖️ Governança": [
                    "Gini", "Qualidade_Regulatoria", "Gastos_Militares_perc_PIB"
                ]
            }
            
            for category, indicators in indicators_by_category.items():
                st.write(f"**{category}**")
                for indicator in indicators:
                    if indicator in df.columns:
                        # Calcular código do World Bank
                        wb_code = [k for k, v in INDICADORES.items() if v == indicator]
                        wb_code = wb_code[0] if wb_code else "N/A"
                        st.write(f"  • {indicator.replace('_', ' ')}")
        
        with col2:
            st.subheader("🤖 Metodologia de Machine Learning")
            
            st.write("""
            **Abordagem de Modelagem**:
            
            1. **Engenharia de Features**:
               - Variáveis lag (t-1 e t-2)
               - Taxas de crescimento
               - Normalização para modelos lineares
            
            2. **Modelos Testados**:
               - Regressão Linear
               - Ridge Regression (regularização L2)
               - Lasso Regression (regularização L1)
               - Árvore de Decisão
               - Random Forest
               - XGBoost
            
            3. **Métricas de Avaliação**:
               - R² Score (coeficiente de determinação)
               - RMSE (erro quadrático médio)
               - MAE (erro absoluto médio)
               - MAPE (erro percentual absoluto médio)
            
            4. **Validação**:
               - Validação in-sample
               - Análise de importância de features
               - Testes de sensibilidade
            """)
            
            st.subheader("🔮 Metodologia de Projeção")
            
            st.write("""
            **Estratégia Híbrida**:
            
            1. **Previsão por ML**: Utiliza o modelo treinado com variáveis lag
            
            2. **Tendência Histórica**: Calcula crescimento médio histórico
            
            3. **Combinação Ponderada**: 
               - Peso maior no modelo nos primeiros anos
               - Peso maior na tendência em anos distantes
               - Fator de decay temporal
            
            4. **Limitações e Controles**:
               - Crescimento anual limitado a ±10%
               - Limites específicos por tipo de indicador
               - Variação estocástica controlada
               - Aplicação de choques persistentes
            
            **Cenários Múltiplos**:
            - Realista: Projeção base
            - Otimista: +1.5% adicional por ano
            - Pessimista: -1% por ano
            """)
            
            st.subheader("⚠️ Limitações e Disclaimers")
            
            st.warning("""
            **IMPORTANTE - Limitações do Modelo**:
            
            • **Não captura choques externos**: Crises, pandemias, guerras
            • **Baseado em dados históricos**: Padrões passados podem não se repetir
            • **Não inclui políticas futuras**: Mudanças regulatórias não previstas
            • **Intervalo de confiança**: ±10-20% nas projeções
            • **Horizonte recomendado**: Máximo 10-15 anos
            
            **Este sistema é uma ferramenta de apoio à decisão, não substituindo análise especializada em economia.**
            """)
    
    # --- FOOTER ---
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <b>🏛️ Sistema de Análise Econômica "Código da Riqueza"</b><br>
        Desenvolvido com dados do Banco Mundial e técnicas avançadas de Machine Learning<br>
        Versão Otimizada Final | © 2024
    </div>
    """, unsafe_allow_html=True)

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Erro crítico na aplicação: {str(e)}")
        st.exception(e)
        st.info("Por favor, recarregue a página ou entre em contato com o suporte técnico.")
else:
    # Código para execução em linha de comando
    print("=" * 80)
    print("🏛️  CÓDIGO DA RIQUEZA - SISTEMA DE ANÁLISE ECONÔMICA AVANÇADA")
    print("=" * 80)
    print()
    print("🚀 Iniciando sistema...")
    
    try:
        # Carregamento de dados
        print("📡 Conectando ao Banco Mundial...")
        loader = WorldBankDataLoader()
        df_raw = loader.load_with_cache_and_retry()
        
        if df_raw is not None:
            print("✅ Dados carregados com sucesso!")
            
            # Processamento
            print("🔧 Processando e limpando dados...")
            processor = DataProcessor()
            df, df_model = processor.process_data(df_raw)
            
            if df is not None and df_model is not None:
                print(f"✅ Processamento concluído!")
                print(f"   • Países: {df['País'].nunique()}")
                print(f"   • Observações: {len(df):,}")
                print(f"   • Período: {df['Ano'].min():.0f}-{df['Ano'].max():.0f}")
                
                # Treinamento de modelos
                print("🤖 Treinando modelos de machine learning...")
                trainer = ModelTrainer()
                models_data = trainer.train_and_evaluate(df_model)
                
                print("🏆 Resultados dos modelos:")
                print(models_data['resultados'].to_string(index=False))
                
                # Exportação
                print("💾 Exportando resultados...")
                
                # Dados processados
                df_export = df_model.reset_index()
                df_export.to_csv("dados_modelo_codigo_riqueza.csv", index=False)
                print("   ✅ dados_modelo_codigo_riqueza.csv")
                
                # Comparação de modelos
                models_data['resultados'].to_csv("comparacao_modelos_codigo_riqueza.csv", index=False)
                print("   ✅ comparacao_modelos_codigo_riqueza.csv")
                
                # Relatório de qualidade
                quality_report = processor.get_quality_report()
                with open("relatorio_qualidade_dados.txt", "w", encoding="utf-8") as f:
                    f.write("RELATÓRIO DE QUALIDADE DOS DADOS\n")
                    f.write("=" * 40 + "\n\n")
                    for stage, data in quality_report.items():
                        f.write(f"{stage.upper()}:\n")
                        for key, value in data.items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")
                print("   ✅ relatorio_qualidade_dados.txt")
                
                print()
                print("🎯 SISTEMA PRONTO!")
                print("💻 Execute: streamlit run codigo_riqueza_otimizado.py")
                print()
                print("🔧 RECURSOS DISPONÍVEIS:")
                print("   ✅ Sistema robusto de cache e retry")
                print("   ✅ Estratégia multinível de imputação")
                print("   ✅ 6 modelos de ML comparados")
                print("   ✅ Análise de sensibilidade avançada")
                print("   ✅ Projeções com múltiplos cenários")
                print("   ✅ Comparação internacional")
                print("   ✅ Interface interativa completa")
                
            else:
                print("❌ Erro no processamento dos dados")
        else:
            print("❌ Erro ao carregar dados do Banco Mundial")
            
    except Exception as e:
        print(f"❌ Erro na execução: {str(e)}")
        print("💡 Sugestão: Execute com 'streamlit run codigo_riqueza_otimizado.py'")
    
    print()
    print("=" * 80)
