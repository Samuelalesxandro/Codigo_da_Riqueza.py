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
        
        # Renomear colunas do índice
        if 'country' in df.columns:
            df.rename(columns={'country': 'País'}, inplace=True)
        if 'date' in df.columns:
            df.rename(columns={'date': 'Ano'}, inplace=True)
            df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce')
        
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
            group = group.copy()  # Evitar SettingWithCopyWarning
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
            
            result = country_data.reset_index()
            result['País'] = group['País'].iloc[0]  # Adicionar país de volta
            return result
        
        # Aplicar processamento por país
        df_processed_list = []
        for pais, group in df.groupby('País'):
            processed_group = process_country_group(group)
            df_processed_list.append(processed_group)
        
        df_processed = pd.concat(df_processed_list, ignore_index=True)
        
        # Como último recurso, preenchimento com mediana regional
        df_processed = self._fill_with_regional_median(df_processed)
        
        # Remover outliers extremos
        df_processed = self._remove_extreme_outliers(df_processed)
        
        return df_processed
    
    def _fill_with_regional_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche dados faltantes com mediana regional - VERSÃO CORRIGIDA"""
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
            # Verificar se há valores faltantes nesta coluna
            if df[col].isnull().sum() == 0:
                continue
                
            # Calcular mediana por região e ano - CORRIGIDO
            try:
                # Agrupar e calcular mediana de forma mais robusta
                regional_medians = df.groupby(['Região', 'Ano'])[col].median()
                
                # Preencher valores faltantes
                mask = df[col].isnull()
                
                for idx in df[mask].index:
                    region = df.loc[idx, 'Região']
                    year = df.loc[idx, 'Ano']
                    
                    # Tentar usar mediana regional por ano
                    try:
                        if pd.notna(regional_medians.loc[(region, year)]):
                            df.loc[idx, col] = regional_medians.loc[(region, year)]
                            continue
                    except (KeyError, TypeError):
                        pass
                    
                    # Fallback: mediana regional geral
                    try:
                        regional_median = df[df['Região'] == region][col].median()
                        if pd.notna(regional_median):
                            df.loc[idx, col] = regional_median
                            continue
                    except:
                        pass
                    
                    # Último recurso: mediana global
                    try:
                        global_median = df[col].median()
                        if pd.notna(global_median):
                            df.loc[idx, col] = global_median
                        else:
                            df.loc[idx, col] = 0
                    except:
                        df.loc[idx, col] = 0
                        
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar coluna {col}: {e}")
                # Como fallback final, preencher com mediana global ou zero
                df[col].fillna(df[col].median() if pd.notna(df[col].median()) else 0, inplace=True)
        
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
        try:
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
            
        except Exception as e:
            st.error(f"❌ Erro na preparação dos dados para modelagem: {e}")
            return None
    
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

class EconomicProjectionSystem:
    """Sistema avançado de projeções econômicas com múltiplos cenários"""
    
    def __init__(self, df_model: pd.DataFrame, trained_models: Dict, models_results: pd.DataFrame):
        self.df_model = df_model
        self.trained_models = trained_models
        self.models_results = models_results
        self.base_indicators = self._get_base_indicators()
        
    def _get_base_indicators(self) -> List[str]:
        """Identifica indicadores base (sem lag/growth)"""
        all_cols = [col for col in self.df_model.columns if col != 'PIB_per_capita']
        base_indicators = []
        
        for col in all_cols:
            base_name = col.replace('_lag1', '').replace('_lag2', '').replace('_growth_lag1', '').replace('_growth', '')
            if base_name not in base_indicators and base_name in self.df_model.columns:
                base_indicators.append(base_name)
        return base_indicators

class EconomicProjectionSystem:
    """Sistema avançado de projeções econômicas com múltiplos cenários - VERSÃO CORRIGIDA"""
    
    def __init__(self, df_model: pd.DataFrame, trained_models: Dict, models_results: pd.DataFrame):
        self.df_model = df_model
        self.trained_models = trained_models
        self.models_results = models_results
        self.base_indicators = self._get_base_indicators()
        
    def _get_base_indicators(self) -> List[str]:
        """Identifica indicadores base (sem lag/growth)"""
        all_cols = [col for col in self.df_model.columns if col != 'PIB_per_capita']
        base_indicators = []
        
        for col in all_cols:
            base_name = col.replace('_lag1', '').replace('_lag2', '').replace('_growth_lag1', '').replace('_growth', '')
            if base_name not in base_indicators and base_name in self.df_model.columns:
                base_indicators.append(base_name)
        return base_indicators

    def create_projection_interface(self):
        """Cria interface Streamlit para projeções econômicas"""
        st.header("🔮 Projeções Econômicas - Cenários Futuros")
        
        # Verificação segura de dados
        if not hasattr(self, 'df_model') or self.df_model is None:
            st.error("❌ Dados do modelo não disponíveis")
            return
        
        if not hasattr(self, 'trained_models') or 'modelos' not in self.trained_models:
            st.error("❌ Modelos treinados não disponíveis")
            return
            
        # Interface de seleção de país
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Obter países disponíveis
            try:
                available_countries = sorted(self.df_model.reset_index()['País'].unique())
                selected_country = st.selectbox(
                    "Selecione o país:",
                    options=available_countries,
                    index=0 if available_countries else None,
                    help="Escolha o país para análise de projeção"
                )
            except Exception as e:
                st.error(f"❌ Erro ao obter países: {e}")
                return
        
        with col2:
            # Seleção do modelo
            model_names = list(self.trained_models['modelos'].keys())
            selected_model = st.selectbox(
                "Modelo:",
                options=model_names,
                index=0,
                help="Modelo para gerar as projeções"
            )
        
        with col3:
            # Anos para projeção
            projection_years = st.slider(
                "Anos para projetar:",
                min_value=1,
                max_value=10,
                value=5,
                help="Número de anos futuros para projeção"
            )
        
        # Obter dados mais recentes do país selecionado
        try:
            country_data = self._get_country_time_series(selected_country)
            
            if country_data is None or country_data.empty:
                st.error(f"❌ Não foi possível obter dados para {selected_country}")
                return
                
        except Exception as e:
            st.error(f"❌ Erro ao obter dados do país: {e}")
            return
        
        # Configuração de cenário
        st.subheader("🌐 Configurar Cenários")
        
        scenarios = {
            "Status Quo": "Manter tendências históricas",
            "Otimista": "Melhorias graduais nos indicadores",
            "Pessimista": "Deterioração gradual dos indicadores",
            "Personalizado": "Definir crescimento customizado"
        }
        
        selected_scenario = st.radio(
            "Selecione o cenário:",
            options=list(scenarios.keys()),
            horizontal=True,
            help="Tipo de cenário econômico para projeção"
        )
        
        # Mostrar descrição do cenário
        st.info(f"📋 **{selected_scenario}**: {scenarios[selected_scenario]}")
        
        # Configurações específicas do cenário
        scenario_params = self._configure_scenario(selected_scenario, country_data)
        
        # Executar projeção
        if st.button("▶️ Executar Projeção", type="primary"):
            with st.spinner(f"📊 Gerando projeção para {selected_country}..."):
                try:
                    projections = self._generate_projections_fixed(
                        country=selected_country,
                        historical_data=country_data,
                        model_name=selected_model,
                        years=projection_years,
                        scenario=selected_scenario,
                        scenario_params=scenario_params
                    )
                    
                    if projections is not None:
                        self._display_projection_results(projections, selected_country, selected_scenario)
                    else:
                        st.error("❌ Erro ao gerar projeções")
                        
                except Exception as e:
                    st.error(f"❌ Erro durante a projeção: {e}")
                    st.exception(e)

    def _get_country_time_series(self, country: str) -> Optional[pd.DataFrame]:
        """Obtém série histórica completa para um país específico"""
        try:
            country_data = self.df_model.reset_index()
            country_df = country_data[country_data['País'] == country].copy()
            
            if country_df.empty:
                st.error(f"❌ País {country} não encontrado nos dados")
                return None
            
            # Ordenar por ano
            country_df = country_df.sort_values('Ano')
            
            return country_df
            
        except Exception as e:
            st.error(f"❌ Erro ao obter dados para {country}: {str(e)}")
            return None

    def _configure_scenario(self, scenario: str, country_data: pd.DataFrame) -> Dict:
        """Configura parâmetros do cenário"""
        
        # Indicadores principais para configuração
        key_indicators = [
            'Formacao_Bruta_Capital', 'Cobertura_Internet', 'Alfabetizacao_Jovens',
            'Desemprego', 'Inflacao_Anual_Consumidor', 'Gini'
        ]
        
        available_indicators = [ind for ind in key_indicators if ind in country_data.columns]
        
        scenario_params = {}
        
        if scenario == "Personalizado" and available_indicators:
            st.info("🔧 Configure as taxas de crescimento anual para os indicadores:")
            
            for indicator in available_indicators:
                # Para indicadores "negativos" (onde crescimento é ruim)
                if indicator in ['Desemprego', 'Inflacao_Anual_Consumidor', 'Gini']:
                    default_growth = -2.0  # Redução de 2% ao ano
                    help_text = f"Taxa de mudança anual para {indicator.replace('_', ' ')} (negativo = melhoria)"
                else:
                    default_growth = 3.0   # Crescimento de 3% ao ano
                    help_text = f"Taxa de crescimento anual para {indicator.replace('_', ' ')}"
                
                scenario_params[indicator] = st.slider(
                    f"{indicator.replace('_', ' ')} (% ao ano):",
                    min_value=-10.0,
                    max_value=15.0,
                    value=default_growth,
                    step=0.5,
                    help=help_text,
                    key=f"growth_{indicator}"
                )
        else:
            # Cenários pré-definidos
            for indicator in available_indicators:
                if scenario == "Otimista":
                    if indicator in ['Desemprego', 'Inflacao_Anual_Consumidor', 'Gini']:
                        scenario_params[indicator] = -3.0  # Redução de 3% ao ano
                    else:
                        scenario_params[indicator] = 4.0   # Crescimento de 4% ao ano
                
                elif scenario == "Pessimista":
                    if indicator in ['Desemprego', 'Inflacao_Anual_Consumidor', 'Gini']:
                        scenario_params[indicator] = 2.0   # Aumento de 2% ao ano
                    else:
                        scenario_params[indicator] = -1.5  # Queda de 1.5% ao ano
                
                else:  # Status Quo
                    # Calcular tendência histórica dos últimos 5 anos
                    recent_data = country_data.tail(5) if len(country_data) >= 5 else country_data
                    if len(recent_data) >= 2 and indicator in recent_data.columns:
                        values = recent_data[indicator].dropna()
                        if len(values) >= 2:
                            # Calcular taxa de crescimento histórica
                            initial = values.iloc[0]
                            final = values.iloc[-1]
                            years = len(values) - 1
                            if initial > 0 and years > 0:
                                historical_growth = ((final / initial) ** (1/years) - 1) * 100
                                scenario_params[indicator] = historical_growth
                            else:
                                scenario_params[indicator] = 0.0
                        else:
                            scenario_params[indicator] = 0.0
                    else:
                        scenario_params[indicator] = 0.0
        
        return scenario_params

    def _generate_projections_fixed(self, country: str, historical_data: pd.DataFrame, 
                                  model_name: str, years: int, scenario: str, 
                                  scenario_params: Dict) -> Optional[pd.DataFrame]:
        """Gera projeções com evolução dinâmica dos indicadores - VERSÃO CORRIGIDA"""
        
        try:
            # Obter o modelo treinado
            model = self.trained_models['modelos'].get(model_name)
            if model is None:
                st.error(f"❌ Modelo {model_name} não encontrado")
                return None
            
            # Obter dados mais recentes
            latest_data = historical_data.iloc[-1].to_dict()
            current_year = int(latest_data['Ano'])
            
            # Preparar estrutura para projeções
            projections = []
            current_values = latest_data.copy()
            
            # Debug info
            st.info(f"🔍 Dados base para {country}: Ano {current_year}, PIB inicial: ${latest_data.get('PIB_per_capita', 0):,.0f}")
            
            for year_offset in range(1, years + 1):
                projection_year = current_year + year_offset
                
                # Atualizar indicadores base com crescimento do cenário
                for indicator, growth_rate in scenario_params.items():
                    if indicator in current_values:
                        current_value = current_values[indicator]
                        if pd.notna(current_value) and current_value != 0:
                            # Aplicar crescimento composto anual
                            new_value = current_value * ((1 + growth_rate/100) ** year_offset)
                            current_values[indicator] = new_value
                
                # Atualizar ano
                current_values['Ano'] = projection_year
                
                # Preparar features para predição
                features_dict = self._build_features_for_prediction(current_values, historical_data, year_offset)
                
                if features_dict is None:
                    st.warning(f"⚠️ Não foi possível preparar features para {projection_year}")
                    # Usar crescimento linear simples como fallback
                    if 'PIB_per_capita' in current_values:
                        # Crescimento baseado na média do cenário
                        avg_growth = np.mean(list(scenario_params.values()))
                        if scenario == "Status Quo":
                            avg_growth = 1.0  # 1% de crescimento padrão
                        current_values['PIB_per_capita'] = latest_data['PIB_per_capita'] * ((1 + avg_growth/100) ** year_offset)
                else:
                    # Fazer predição usando o modelo
                    try:
                        features_array = np.array(list(features_dict.values())).reshape(1, -1)
                        
                        # Verificar se precisa normalizar
                        if model_name in ["Regressão Linear", "Ridge Regression", "Lasso Regression"]:
                            # Usar normalização
                            X_mean = self.trained_models.get('X_normalized', self.trained_models['X']).mean()
                            X_std = self.trained_models.get('X_normalized', self.trained_models['X']).std()
                            features_array = (features_array - X_mean.values) / X_std.values
                        
                        prediction = model.predict(features_array)[0]
                        
                        # Validar predição
                        if np.isfinite(prediction) and prediction > 0:
                            current_values['PIB_per_capita'] = prediction
                        else:
                            # Fallback para crescimento linear
                            avg_growth = np.mean(list(scenario_params.values())) if scenario_params else 1.0
                            current_values['PIB_per_capita'] = latest_data['PIB_per_capita'] * ((1 + avg_growth/100) ** year_offset)
                            
                    except Exception as e:
                        st.warning(f"⚠️ Erro na predição para {projection_year}: {e}")
                        # Fallback
                        avg_growth = 1.0
                        current_values['PIB_per_capita'] = latest_data['PIB_per_capita'] * ((1 + avg_growth/100) ** year_offset)
                
                # Adicionar informações de contexto
                projection_row = current_values.copy()
                projection_row['Cenário'] = scenario
                projection_row['Fonte'] = 'Projeção'
                projection_row['Ano_Offset'] = year_offset
                
                projections.append(projection_row)
            
            # Criar DataFrame com projeções
            projections_df = pd.DataFrame(projections)
            
            # Preparar dados históricos
            historical_prepared = historical_data.copy()
            historical_prepared['Cenário'] = 'Histórico'
            historical_prepared['Fonte'] = 'Dados'
            historical_prepared['Ano_Offset'] = 0
            
            # Combinar dados
            full_data = pd.concat([historical_prepared, projections_df], ignore_index=True)
            full_data = full_data.sort_values('Ano')
            
            # Debug: mostrar primeiras projeções
            if len(projections_df) > 0:
                first_proj = projections_df.iloc[0]
                last_proj = projections_df.iloc[-1]
                st.success(f"✅ Projeção gerada: {first_proj['Ano']:.0f} (${first_proj['PIB_per_capita']:,.0f}) → {last_proj['Ano']:.0f} (${last_proj['PIB_per_capita']:,.0f})")
            
            return full_data
            
        except Exception as e:
            st.error(f"❌ Erro ao gerar projeções: {str(e)}")
            st.exception(e)
            return None

    def _build_features_for_prediction(self, current_data: Dict, historical_data: pd.DataFrame, year_offset: int) -> Optional[Dict]:
        """Constrói features necessárias para predição do modelo"""
        
        try:
            features = {}
            
            # Obter lista de preditores do modelo
            predictors = self.trained_models.get('predictors', [])
            
            if not predictors:
                return None
            
            # Para cada preditor necessário
            for predictor in predictors:
                if '_lag1' in predictor:
                    # Feature de lag 1
                    base_var = predictor.replace('_lag1', '')
                    if base_var in current_data:
                        features[predictor] = current_data[base_var]
                    else:
                        features[predictor] = 0
                        
                elif '_lag2' in predictor:
                    # Feature de lag 2
                    base_var = predictor.replace('_lag2', '')
                    # Usar valor de 2 anos atrás (ou valor atual como aproximação)
                    if len(historical_data) >= 2 and base_var in historical_data.columns:
                        lag2_value = historical_data.iloc[-2][base_var] if pd.notna(historical_data.iloc[-2][base_var]) else current_data.get(base_var, 0)
                        features[predictor] = lag2_value
                    else:
                        features[predictor] = current_data.get(base_var, 0)
                        
                elif '_growth_lag1' in predictor:
                    # Feature de crescimento com lag
                    base_var = predictor.replace('_growth_lag1', '')
                    if base_var in current_data and len(historical_data) >= 1:
                        current_val = current_data[base_var]
                        prev_val = historical_data.iloc[-1][base_var] if pd.notna(historical_data.iloc[-1][base_var]) else current_val
                        
                        if prev_val != 0:
                            growth = (current_val - prev_val) / prev_val
                        else:
                            growth = 0
                        features[predictor] = growth
                    else:
                        features[predictor] = 0
                        
                elif '_growth' in predictor and '_growth_lag1' not in predictor:
                    # Feature de crescimento simples
                    base_var = predictor.replace('_growth', '')
                    features[predictor] = 0.02  # Assume 2% de crescimento padrão
                    
                else:
                    # Feature direta
                    if predictor in current_data:
                        features[predictor] = current_data[predictor]
                    else:
                        features[predictor] = 0
            
            return features
            
        except Exception as e:
            st.warning(f"⚠️ Erro ao construir features: {e}")
            return None

    def _display_projection_results(self, projections: pd.DataFrame, country: str, scenario: str):
        """Exibe os resultados da projeção - VERSÃO MELHORADA"""
        
        try:
            # Separar dados históricos e projetados
            historical = projections[projections['Fonte'] == 'Dados']
            projected = projections[projections['Fonte'] == 'Projeção']
            
            if historical.empty or projected.empty:
                st.error("❌ Dados insuficientes para exibir projeção")
                return
            
            st.success(f"✅ Projeção concluída para {country} - Cenário {scenario}")
            
            # Gráfico principal
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Dados históricos
            hist_years = historical['Ano'].values
            hist_pib = historical['PIB_per_capita'].values
            
            # Dados projetados
            proj_years = projected['Ano'].values
            proj_pib = projected['PIB_per_capita'].values
            
            # Linhas principais
            ax.plot(hist_years, hist_pib, 'o-', color='steelblue', linewidth=3, 
                   markersize=8, label='Dados Históricos', alpha=0.9)
            
            ax.plot(proj_years, proj_pib, 's--', color='green', linewidth=3, 
                   markersize=8, label=f'Projeção ({scenario})', alpha=0.9)
            
            # Conectar último ponto histórico com primeiro projetado
            if len(hist_years) > 0 and len(proj_years) > 0:
                ax.plot([hist_years[-1], proj_years[0]], [hist_pib[-1], proj_pib[0]], 
                       ':', color='gray', linewidth=2, alpha=0.7)
            
            # Área de incerteza para projeções
            uncertainty_factor = 0.15  # ±15%
            proj_upper = proj_pib * (1 + uncertainty_factor)
            proj_lower = proj_pib * (1 - uncertainty_factor)
            
            ax.fill_between(proj_years, proj_lower, proj_upper, 
                          color='green', alpha=0.2, label='Intervalo de Incerteza (±15%)')
            
            # Configurações do gráfico
            ax.set_title(f'Projeção de PIB per capita - {country}\nCenário: {scenario}', 
                        fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Ano', fontsize=14, fontweight='bold')
            ax.set_ylabel('PIB per capita (US$)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12, loc='best')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Formatação dos eixos
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Anotações importantes
            try:
                if len(hist_pib) > 0 and len(proj_pib) > 0:
                    # Último valor histórico
                    ax.annotate(f'Último dado\n{hist_years[-1]:.0f}: ${hist_pib[-1]:,.0f}',
                               xy=(hist_years[-1], hist_pib[-1]),
                               xytext=(20, 20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', lw=1.5),
                               fontsize=11, fontweight='bold')
                    
                    # Última projeção
                    final_year = proj_years[-1]
                    final_pib = proj_pib[-1]
                    
                    # Calcular taxa de crescimento anual
                    years_diff = len(proj_pib)
                    if hist_pib[-1] > 0 and years_diff > 0:
                        total_growth = ((final_pib / hist_pib[-1]) ** (1/years_diff) - 1) * 100
                    else:
                        total_growth = 0
                    
                    ax.annotate(f'Projeção {final_year:.0f}\n${final_pib:,.0f}\n({total_growth:+.1f}% a.a.)',
                               xy=(final_year, final_pib),
                               xytext=(-80, 20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', lw=1.5),
                               fontsize=11, fontweight='bold')
            except Exception as e:
                st.warning(f"⚠️ Erro ao adicionar anotações: {e}")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Métricas resumidas
            self._display_projection_metrics(historical, projected)
            
            # Tabela de valores
            self._display_projection_table(projected, country, scenario)
            
        except Exception as e:
            st.error(f"❌ Erro ao exibir resultados: {e}")
            st.exception(e)

    def _display_projection_metrics(self, historical: pd.DataFrame, projected: pd.DataFrame):
        """Exibe métricas resumidas da projeção"""
        
        st.subheader("📊 Resumo da Projeção")
        
        try:
            initial_pib = historical['PIB_per_capita'].iloc[-1]
            final_pib = projected['PIB_per_capita'].iloc[-1]
            years = len(projected)
            
            # Cálculos
            absolute_change = final_pib - initial_pib
            if initial_pib > 0 and years > 0:
                annual_growth = ((final_pib / initial_pib) ** (1/years) - 1) * 100
                total_growth = ((final_pib / initial_pib) - 1) * 100
            else:
                annual_growth = 0
                total_growth = 0
            
            # Exibir métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("PIB per capita Base", f"${initial_pib:,.0f}")
            
            with col2:
                st.metric("PIB per capita Final", f"${final_pib:,.0f}", 
                         delta=f"${absolute_change:+,.0f}")
            
            with col3:
                st.metric("Crescimento Total", f"{total_growth:+.1f}%")
            
            with col4:
                st.metric("Crescimento Anual", f"{annual_growth:+.1f}%")
            
            # Análise contextual
            if annual_growth > 3:
                st.success(f"🚀 **Crescimento acelerado**: {annual_growth:.1f}% ao ano indica expansão econômica robusta")
            elif annual_growth > 1:
                st.info(f"📈 **Crescimento moderado**: {annual_growth:.1f}% ao ano representa desenvolvimento estável")
            elif annual_growth > -1:
                st.warning(f"📊 **Crescimento baixo**: {annual_growth:.1f}% ao ano indica estagnação econômica")
            else:
                st.error(f"📉 **Declínio econômico**: {annual_growth:.1f}% ao ano representa contração significativa")
                
        except Exception as e:
            st.error(f"❌ Erro ao calcular métricas: {e}")

    def _display_projection_table(self, projected: pd.DataFrame, country: str, scenario: str):
        """Exibe tabela com valores projetados"""
        
        st.subheader("📋 Valores Projetados Detalhados")
        
        try:
            # Preparar dados para exibição
            display_cols = ['Ano', 'PIB_per_capita']
            
            # Adicionar indicadores principais se disponíveis
            extra_cols = ['Formacao_Bruta_Capital', 'Cobertura_Internet', 'Alfabetizacao_Jovens', 
                         'Desemprego', 'Inflacao_Anual_Consumidor']
            
            for col in extra_cols:
                if col in projected.columns:
                    display_cols.append(col)
            
            table_data = projected[display_cols].copy()
            
            # Formatação
            money_cols = ['PIB_per_capita', 'Formacao_Bruta_Capital']
            pct_cols = ['Cobertura_Internet', 'Alfabetizacao_Jovens', 'Desemprego', 'Inflacao_Anual_Consumidor']
            
            for col in money_cols:
                if col in table_data.columns:
                    table_data[col] = table_data[col].apply(lambda x: f"${x:,.0f}")
            
            for col in pct_cols:
                if col in table_data.columns:
                    table_data[col] = table_data[col].apply(lambda x: f"{x:.1f}%")
            
            # Renomear colunas para exibição
            table_data.columns = [col.replace('_', ' ').title() for col in table_data.columns]
            
            st.dataframe(table_data.set_index('Ano'), use_container_width=True)
            
            # Download
            csv_data = projected[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Projeção Completa (CSV)",
                data=csv_data,
                file_name=f"projecao_{country}_{scenario.lower().replace(' ', '_')}.csv",
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"❌ Erro ao exibir tabela: {e}")


# Função para integrar ao sistema principal
def add_projections_to_main():
    """Adiciona funcionalidade de projeções ao sistema principal - VERSÃO CORRIGIDA"""
    
    st.markdown("---")
    
    # Verificar se os modelos foram treinados
    if 'models_data' in st.session_state and 'df_model' in st.session_state:
        try:
            projection_system = EconomicProjectionSystem(
                df_model=st.session_state.df_model,
                trained_models=st.session_state.models_data,
                models_results=st.session_state.models_data['resultados']
            )
            projection_system.create_projection_interface()
        except Exception as e:
            st.error(f"❌ Erro ao inicializar sistema de projeções: {e}")
            st.info("🔄 Execute primeiro a seção de treinamento de modelos para habilitar as projeções.")
    else:
        st.info("🔄 Execute primeiro a seção de treinamento de modelos para habilitar as projeções.")
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
    - ✅ **Sistema de projeções econômicas corrigido**
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
        elif model_info['R²'] >= 0.6:
            performance_level = "Bom"
        else:
            performance_level = "Limitado"
        
        if performance_level == "Excelente":
            st.success(f"""
            **Performance: {performance_level}**
            
            Este modelo tem poder explicativo **{performance_level.lower()}** para prever o PIB per capita.
            Com erro médio de ${model_info['MAE']:,.0f}, as projeções têm precisão de 
            ±{model_info['MAPE']:.1f}% em média.
            """)
        elif performance_level == "Bom":
            st.info(f"""
            **Performance: {performance_level}**
            
            Este modelo tem poder explicativo **{performance_level.lower()}** para prever o PIB per capita.
            Com erro médio de ${model_info['MAE']:,.0f}, as projeções têm precisão de 
            ±{model_info['MAPE']:.1f}% em média.
            """)
        else:
            st.warning(f"""
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
            help="Define o intervalo de anos para análise histórica"
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
        
        # Adicionar anotações para valores máximo e mínimo - VERSÃO CORRIGIDA
        try:
            if len(y_data) > 0 and not y_data.empty:
                # Encontrar índices válidos
                max_idx = y_data.idxmax()
                min_idx = y_data.idxmin()
                
                # Verificar se os índices existem em ambas as séries
                if max_idx in x_data.index and max_idx in y_data.index:
                    max_x = x_data.loc[max_idx]
                    max_y = y_data.loc[max_idx]
                    ax.annotate(f'Máx: {max_y:,.0f}', 
                               xy=(max_x, max_y),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                if min_idx in x_data.index and min_idx in y_data.index:
                    min_x = x_data.loc[min_idx]
                    min_y = y_data.loc[min_idx]
                    ax.annotate(f'Mín: {min_y:,.0f}', 
                               xy=(min_x, min_y),
                               xytext=(10, -20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        except Exception as e:
            # Se houver erro nas anotações, continua sem elas
            st.warning(f"⚠️ Não foi possível adicionar anotações de máx/mín: {e}")
            pass
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Estatísticas resumidas - VERSÃO CORRIGIDA
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                current_value = y_data.iloc[-1] if len(y_data) > 0 else 0
                st.metric("Valor Atual", f"{current_value:,.0f}")
            except:
                st.metric("Valor Atual", "N/A")
                
        with col2:
            try:
                if len(y_data) > 1:
                    initial_value = y_data.iloc[0]
                    final_value = y_data.iloc[-1]
                    if initial_value > 0:
                        growth_rate = ((final_value / initial_value) ** (1/(len(y_data)-1)) - 1) * 100
                    else:
                        growth_rate = 0
                else:
                    growth_rate = 0
                st.metric("Crescimento Anual Médio", f"{growth_rate:+.1f}%")
            except:
                st.metric("Crescimento Anual Médio", "N/A")
                
        with col3:
            try:
                max_value = y_data.max() if len(y_data) > 0 else 0
                st.metric("Máximo Histórico", f"{max_value:,.0f}")
            except:
                st.metric("Máximo Histórico", "N/A")
                
        with col4:
            try:
                min_value = y_data.min() if len(y_data) > 0 else 0
                st.metric("Mínimo Histórico", f"{min_value:,.0f}")
            except:
                st.metric("Mínimo Histórico", "N/A")
    
    else:
        st.warning("⚠️ Nenhum dado disponível para os filtros selecionados")
    
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
            try:
                if 'PIB_per_capita' in df_filtered.columns and len(df_filtered) > 1:
                    # Verificar se há dados suficientes
                    pib_data = df_filtered['PIB_per_capita'].dropna()
                    if len(pib_data) > 1:
                        correlations = df_filtered[numeric_cols].corr()['PIB_per_capita'].sort_values(ascending=False)
                        correlations = correlations.drop('PIB_per_capita', errors='ignore').head(10)
                        
                        if len(correlations) > 0:
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
                        else:
                            st.info("📊 Não há correlações significativas para exibir")
                    else:
                        st.info("📊 Dados insuficientes para análise de correlação")
                else:
                    st.info("📊 PIB per capita não disponível para análise de correlação")
                    
            except Exception as e:
                st.warning(f"⚠️ Erro ao calcular correlações: {e}")
                st.info("📊 Análise de correlação não disponível para este país/período")
        
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
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Dados como CSV",
                data=csv_data,
                file_name=f"{selected_country}_dados_historicos.csv",
                mime='text/csv'
            )
        
        with col2:
            # Criar relatório resumido - VERSÃO CORRIGIDA
            report = f"""
RELATÓRIO ECONÔMICO - {selected_country}
Período: {year_start:.0f} - {year_end:.0f}
Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

INDICADORES PRINCIPAIS:
"""
            
            try:
                if 'PIB_per_capita' in df_filtered.columns and len(df_filtered) > 0:
                    pib_data = df_filtered['PIB_per_capita'].dropna()
                    if len(pib_data) > 1:
                        pib_inicial = pib_data.iloc[0]
                        pib_final = pib_data.iloc[-1]
                        if pib_inicial > 0:
                            crescimento_periodo = ((pib_final / pib_inicial) ** (1/(len(pib_data)-1)) - 1) * 100
                        else:
                            crescimento_periodo = 0
                        
                        report += f"""
• PIB per capita inicial: ${pib_inicial:,.0f}
• PIB per capita final: ${pib_final:,.0f}
• Crescimento anual médio: {crescimento_periodo:.2f}%

"""
                    elif len(pib_data) == 1:
                        report += f"""
• PIB per capita: ${pib_data.iloc[0]:,.0f}
• Dados disponíveis apenas para um ano

"""
                
                # Adicionar outros indicadores importantes - COM VERIFICAÇÃO
                key_indicators = ['Alfabetizacao_Jovens', 'Desemprego', 'Cobertura_Internet', 'Gini']
                for indicator in key_indicators:
                    if indicator in df_filtered.columns:
                        indicator_data = df_filtered[indicator].dropna()
                        if len(indicator_data) >= 2:
                            initial_val = indicator_data.iloc[0]
                            final_val = indicator_data.iloc[-1]
                            change = final_val - initial_val
                            report += f"• {indicator.replace('_', ' ')}: {initial_val:.1f} → {final_val:.1f} ({change:+.1f})\n"
                        elif len(indicator_data) == 1:
                            report += f"• {indicator.replace('_', ' ')}: {indicator_data.iloc[0]:.1f}\n"
                
            except Exception as e:
                report += f"\n⚠️ Erro ao processar alguns indicadores: {str(e)}\n"
            
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
    
    else:
        st.warning("⚠️ Nenhum dado disponível para o país e período selecionados")
    
    # --- PROJEÇÕES ECONÔMICAS ---
    add_projections_to_main()

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
            
            5. **Sistema de Projeções**:
               - Cenários econômicos (Otimista, Pessimista, Status Quo)
               - Configuração personalizada de indicadores
               - Projeções de 1-10 anos
               - Intervalos de confiança
            """)
            
            st.subheader("⚠️ Limitações e Disclaimers")
            
            st.warning("""
            **IMPORTANTE - Limitações do Modelo**:
            
            • **Não captura choques externos**: Crises, pandemias, guerras
            • **Baseado em dados históricos**: Padrões passados podem não se repetir
            • **Não inclui políticas futuras**: Mudanças regulatórias não previstas
            • **Intervalo de confiança**: ±10-20% nas projeções
            • **Dependência de dados**: Qualidade limitada pela fonte
            
            **Este sistema é uma ferramenta de apoio à decisão, não substituindo análise especializada em economia.**
            """)
    
    # --- FOOTER ---
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <b>🏛️ Sistema de Análise Econômica "Código da Riqueza"</b><br>
        Desenvolvido com dados do Banco Mundial e técnicas avançadas de Machine Learning<br>
        <b>✅ Versão Corrigida com Sistema de Projeções Funcional</b> | © 2024
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
                print("💻 Execute: streamlit run codigo_riqueza_corrigido.py")
                print()
                print("🔧 RECURSOS DISPONÍVEIS:")
                print("   ✅ Sistema robusto de cache e retry")
                print("   ✅ Estratégia multinível de imputação")
                print("   ✅ 6 modelos de ML comparados")
                print("   ✅ Análise de importância de variáveis")
                print("   ✅ Sistema de projeções econômicas CORRIGIDO")
                print("   ✅ Interface interativa completa")
                
            else:
                print("❌ Erro no processamento dos dados")
        else:
            print("❌ Erro ao carregar dados do Banco Mundial")
            
    except Exception as e:
        print(f"❌ Erro na execução: {str(e)}")
        print("💡 Sugestão: Execute com 'streamlit run codigo_riqueza_corrigido.py'")
    
    print()
    print("=" * 80)
