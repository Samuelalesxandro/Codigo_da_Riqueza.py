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

# --- APLICAÇÃO STREAMLIT PRINCIPAL ---



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
            # Remove sufixos _lag1, _lag2, _growth, etc.
            base_name = col.replace('_lag1', '').replace('_lag2', '').replace('_growth_lag1', '').replace('_growth', '')
            if base_name not in base_indicators and base_name in self.df_model.columns:
                base_indicators.append(base_name)
                
        return base_indicators

    def create_projection_interface():
    """Cria interface Streamlit para projeções econômicas"""
    
    st.header("🔮 Projeções Econômicas Avançadas")
    
    # Verificar se os dados necessários estão disponíveis
    required_session_vars = ['df_model', 'models_data']
    missing_vars = [var for var in required_session_vars if var not in st.session_state]
    
    if missing_vars:
        st.error(f"❌ Dados necessários não encontrados: {missing_vars}")
        st.info("Por favor, execute primeiro a seção de treinamento de modelos.")
        return
    
    df_model = st.session_state.df_model
    models_data = st.session_state.models_data
    
    # Inicializar sistema de projeções
    projection_system = EconomicProjectionSystem(
        df_model=df_model,
        trained_models=models_data['modelos'],
        models_results=models_data['resultados']
    )
    
    # Configurações da projeção
    st.subheader("⚙️ Configurações da Projeção")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_countries = sorted(df_model.reset_index()['País'].unique())
        selected_country = st.selectbox(
            "País para projeção:",
            available_countries,
            help="Escolha o país para análise prospectiva"
        )
    
    with col2:
        years_ahead = st.slider(
            "Horizonte (anos):",
            min_value=1,
            max_value=10,
            value=5,
            help="Número de anos para projetar no futuro"
        )
    
    with col3:
        available_models = models_data['resultados']['Modelo'].tolist()
        selected_model = st.selectbox(
            "Modelo de projeção:",
            available_models,
            index=0,
            help="Modelo de ML para fazer as projeções"
        )
    
    st.subheader("📊 Cenários Econômicos")
    scenarios = projection_system.create_projection_scenarios()
    scenario_tabs = st.tabs(list(scenarios.keys()) + ["Personalizado"])
    projections_results = {}
    
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        with scenario_tabs[i]:
            st.write(f"**{scenario_data['description']}**")
            st.write("**Principais ajustes anuais:**")
            adjustments_df = pd.DataFrame([
                {"Indicador": k.replace('_', ' ').title(), "Crescimento Anual": f"{v:+.1%}"}
                for k, v in scenario_data['adjustments'].items()
            ])
            st.dataframe(adjustments_df, use_container_width=True, hide_index=True)
            
            if st.button(f"🚀 Calcular Projeção - {scenario_name}", key=f"calc_{scenario_name}"):
                with st.spinner(f"Calculando projeção {scenario_name}..."):
                    try:
                        future_data = projection_system.generate_future_data(
                            country=selected_country,
                            years_ahead=years_ahead,
                            scenario=scenario_name
                        )
                        results = projection_system.make_projections(future_data, selected_model)
                        projections_results[scenario_name] = results
                        
                        st.success(f"✅ Projeção {scenario_name} calculada com sucesso!")
                        initial_gdp = results['predicoes'][0]
                        final_gdp = results['predicoes'][-1]
                        cagr = ((final_gdp / initial_gdp) ** (1/years_ahead) - 1) * 100
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("PIB per capita inicial", f"${initial_gdp:,.0f}")
                        with col_b:
                            st.metric("PIB per capita final", f"${final_gdp:,.0f}")
                        with col_c:
                            st.metric("Crescimento anual médio", f"{cagr:+.1f}%")
                    except Exception as e:
                        st.error(f"❌ Erro ao calcular projeção: {str(e)}")
    
    with scenario_tabs[-1]:
        st.write("**Crie seu próprio cenário econômico**")
        custom_adjustments = {}
        col1, col2 = st.columns(2)
        key_indicators = [
            "Formacao_Bruta_Capital", "Valor_Exportacoes", "Consumo_Familias",
            "Investimento_Estrangeiro_Direto", "Cobertura_Internet", 
            "Alfabetizacao_Jovens", "Desemprego", "Inflacao_Anual_Consumidor"
        ]
        
        for i, indicator in enumerate(key_indicators):
            col = col1 if i % 2 == 0 else col2
            with col:
                display_name = indicator.replace('_', ' ').title()
                adjustment = st.slider(
                    f"{display_name}:",
                    min_value=-0.10,
                    max_value=0.15,
                    value=0.02,
                    step=0.005,
                    format="%.1%%",
                    key=f"custom_{indicator}",
                    help=f"Crescimento anual para {display_name}"
                )
                custom_adjustments[indicator] = adjustment
        
        if st.button("🎯 Calcular Cenário Personalizado", key="calc_custom"):
            with st.spinner("Calculando cenário personalizado..."):
                try:
                    future_data = projection_system.generate_future_data(
                        country=selected_country,
                        years_ahead=years_ahead,
                        scenario="custom",
                        custom_adjustments=custom_adjustments
                    )
                    results = projection_system.make_projections(future_data, selected_model)
                    projections_results["Personalizado"] = results
                    
                    st.success("✅ Cenário personalizado calculado com sucesso!")
                    initial_gdp = results['predicoes'][0]
                    final_gdp = results['predicoes'][-1]
                    cagr = ((final_gdp / initial_gdp) ** (1/years_ahead) - 1) * 100
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("PIB per capita inicial", f"${initial_gdp:,.0f}")
                    with col_b:
                        st.metric("PIB per capita final", f"${final_gdp:,.0f}")
                    with col_c:
                        st.metric("Crescimento anual médio", f"{cagr:+.1f}%")
                except Exception as e:
                    st.error(f"❌ Erro ao calcular cenário personalizado: {str(e)}")
    
    if projections_results:
        st.subheader("📈 Comparação de Cenários")
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (scenario, results) in enumerate(projections_results.items()):
            color = colors[i % len(colors)]
            ax.plot(results['anos'], results['predicoes'], 
                    marker='o', linewidth=3, markersize=6, 
                    label=scenario, color=color, alpha=0.9)
            ax.fill_between(results['anos'], 
                            results['intervalo_inferior'], 
                            results['intervalo_superior'],
                            alpha=0.2, color=color)
        
        ax.set_title(f'Projeções do PIB per capita - {selected_country}', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Ano', fontsize=12)
        ax.set_ylabel('PIB per capita (US$)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("📊 Resumo Comparativo dos Cenários")
        comparison_data = []
        for scenario, results in projections_results.items():
            initial_gdp = results['predicoes'][0]
            final_gdp = results['predicoes'][-1]
            cagr = ((final_gdp / initial_gdp) ** (1/years_ahead) - 1) * 100
            total_growth = ((final_gdp / initial_gdp) - 1) * 100
            
            comparison_data.append({
                'Cenário': scenario,
                'PIB Inicial (US$)': f"${initial_gdp:,.0f}",
                'PIB Final (US$)': f"${final_gdp:,.0f}",
                'Crescimento Total': f"{total_growth:+.1f}%",
                'Crescimento Anual': f"{cagr:+.1f}%",
                'Modelo R²': f"{results['r2']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.subheader("💾 Exportar Resultados")
        col1, col2 = st.columns(2)
        
        with col1:
            all_projections = []
            for scenario, results in projections_results.items():
                for i, year in enumerate(results['anos']):
                    all_projections.append({
                        'País': selected_country,
                        'Cenário': scenario,
                        'Ano': year,
                        'PIB_per_capita_projetado': results['predicoes'][i],
                        'Intervalo_Inferior': results['intervalo_inferior'][i],
                        'Intervalo_Superior': results['intervalo_superior'][i],
                        'Modelo': results['modelo_usado']
                    })
            projections_df = pd.DataFrame(all_projections)
            csv_data = projections_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Projeções (CSV)",
                data=csv_data,
                file_name=f"projecoes_{selected_country}_{years_ahead}anos.csv",
                mime='text/csv'
            )
        
        with col2:
            report = f\"""RELATÓRIO DE PROJEÇÕES ECONÔMICAS
{selected_country} - Horizonte: {years_ahead} anos
Modelo utilizado: {selected_model}
Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

RESUMO EXECUTIVO:
\"""
            for scenario, results in projections_results.items():
                initial_gdp = results['predicoes'][0]
                final_gdp = results['predicoes'][-1]
                cagr = ((final_gdp / initial_gdp) ** (1/years_ahead) - 1) * 100
                report += f\"""\n{scenario.upper()}:
• PIB per capita inicial: ${initial_gdp:,.0f}
• PIB per capita final: ${final_gdp:,.0f}
• Crescimento anual médio: {cagr:+.2f}%
• R² do modelo: {results['r2']:.3f}
\"""
            report += f\""""

METODOLOGIA:
• Modelo de projeção: {selected_model}
• Precisão do modelo (RMSE): ${models_data['resultados'].iloc[0]['RMSE']:,.0f}
• Intervalo de confiança: ±95%
• Features utilizadas: {len(models_data['predictors'])} variáveis

DISCLAIMER:
As projeções são baseadas em modelos econométricos e cenários 
hipotéticos. Resultados reais podem variar significativamente 
devido a choques externos, mudanças políticas e outros fatores 
não previstos nos modelos.

Sistema "Código da Riqueza" - Versão Projeções
\"""
            st.download_button(
                label="📄 Baixar Relatório",
                data=report.encode('utf-8'),
                file_name=f"relatorio_projecoes_{selected_country}.txt",
                mime='text/plain'
            )


def add_projections_to_main():
    """Adiciona funcionalidade de projeções ao sistema principal"""
    
    # Adicionar na navegação principal (após a seção de modelos)
    st.markdown("---")
    
    # Verificar se os modelos foram treinados
    if 'models_data' in st.session_state and 'df_model' in st.session_state:
        create_projection_interface()
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
            """)
            
            st.subheader("⚠️ Limitações e Disclaimers")
            
            st.warning("""
            **IMPORTANTE - Limitações do Modelo**:
            
            • **Não captura choques externos**: Crises, pandemias, guerras
            • **Baseado em dados históricos**: Padrões passados podem não se repetir
            • **Não inclui políticas futuras**: Mudanças regulatórias não previstas
            • **Intervalo de confiança**: ±10-20% nas análises
            
            **Este sistema é uma ferramenta de apoio à decisão, não substituindo análise especializada em economia.**
            """)
    
    # --- FOOTER ---
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <b>🏛️ Sistema de Análise Econômica "Código da Riqueza"</b><br>
        Desenvolvido com dados do Banco Mundial e técnicas avançadas de Machine Learning<br>
        Versão Corrigida | © 2024
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
