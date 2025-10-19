import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ML
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional packages (SHAP)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Banco Mundial (wbdata) - leave installed if using automatic download
try:
    import wbdata
    WBDATA_AVAILABLE = True
except Exception:
    WBDATA_AVAILABLE = False

# -------------------- Config --------------------
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

_cached_data = None
_cached_models = None

# -------------------- Data utils --------------------
def carregar_dados_banco_mundial(force_refresh=False):
    global _cached_data
    if _cached_data is not None and not force_refresh:
        return _cached_data
    if not WBDATA_AVAILABLE:
        st.error("wbdata não instalado. Forneça um DataFrame ou instale wbdata.")
        return None
    try:
        df_raw = wbdata.get_dataframe(indicators=INDICADORES, country=TODOS_PAISES, date=(DATA_INICIO, DATA_FIM))
        _cached_data = pd.DataFrame(df_raw).reset_index()
        return _cached_data
    except Exception as e:
        st.error(f"Erro ao baixar dados do Banco Mundial: {e}")
        return None

def processar_dados(df_raw):
    if df_raw is None:
        return None, None
    df = df_raw.copy().reset_index(drop=True)
    if 'country' in df.columns:
        df.rename(columns={'country': 'País'}, inplace=True)
    if 'date' in df.columns:
        df.rename(columns={'date': 'Ano'}, inplace=True)
    if 'País' not in df.columns and hasattr(df_raw, 'index') and hasattr(df_raw.index, 'get_level_values'):
        try:
            df['País'] = df_raw.index.get_level_values('country')
        except Exception:
            pass
    if 'Ano' not in df.columns and hasattr(df_raw, 'index') and hasattr(df_raw.index, 'get_level_values'):
        try:
            df['Ano'] = df_raw.index.get_level_values('date')
        except Exception:
            pass
    if 'País' not in df.columns or 'Ano' not in df.columns:
        st.error("Colunas 'País' ou 'Ano' ausentes.")
        return None, None
    df = df.sort_values(['País','Ano'])
    df = df.groupby('País', group_keys=False).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
    df = df.dropna()
    df_model = df.copy().set_index(['País','Ano'])
    for col in df_model.columns:
        if col != 'PIB_per_capita':
            df_model[f"{col}_lag1"] = df_model.groupby('País')[col].shift(1)
    df_model = df_model.dropna()
    return df, df_model.reset_index()

# -------------------- Modeling --------------------
def treinar_todos_modelos(df_model):
    global _cached_models
    if _cached_models is not None:
        return _cached_models
    TARGET = 'PIB_per_capita'
    PREDICTORS = [c for c in df_model.columns if c.endswith('_lag1')]
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
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        resultados.append({'Modelo':nome,'R²':round(r2,4),'RMSE':round(rmse,2),'MAE':round(mae,2)})
        modelos_treinados[nome] = modelo
        if hasattr(modelo, 'feature_importances_'):
            modelos_treinados[f"{nome}_importance"] = pd.Series(modelo.feature_importances_, index=PREDICTORS).sort_values(ascending=False)
    _cached_models = {'resultados':pd.DataFrame(resultados).sort_values('R²', ascending=False), 'modelos':modelos_treinados, 'X':X, 'y':y, 'predictors':PREDICTORS}
    return _cached_models

# -------------------- Improved projections --------------------
def gerar_projecao_realista_improved(df_model, pais, modelo, ano_final=2035, uncertainty=True):
    df = df_model.copy()
    dfp = df[df['País']==pais].sort_values('Ano').reset_index(drop=True)
    if dfp.empty:
        raise ValueError("Dados insuficientes para o país")
    ultimo = dfp.iloc[-1].copy()
    ultimo_ano = int(ultimo['Ano'])
    target_col = 'PIB_per_capita'
    predictors = [c for c in df.columns if c.endswith('_lag1')]
    hist = dfp.tail(6)[target_col].astype(float)
    if len(hist) >= 2:
        growth_rates = hist.pct_change().dropna()+1
        geom = growth_rates.prod()**(1/len(growth_rates))
        trend_rate = geom - 1
        trend_rate = max(-0.05, min(0.05, trend_rate))
    else:
        trend_rate = 0.02
    anos_fut = list(range(ultimo_ano+1, int(ano_final)+1))
    rows = []
    base_row = ultimo.copy()
    X_all = df[[c for c in df.columns if c.endswith('_lag1')]]
    y_all = df[target_col]
    try:
        preds_all = modelo.predict(X_all)
        resid = (y_all - preds_all)
        resid_std = float(np.nanstd(resid))
    except Exception:
        resid_std = max(1.0, float(np.nanstd(y_all)) * 0.05)
    weight_model = 0.7
    for i, ano in enumerate(anos_fut):
        new = base_row.copy()
        new['Ano'] = int(ano)
        for p in predictors:
            base_name = p.replace('_lag1','')
            if base_name in base_row.index:
                new[p] = base_row[base_name]
        X_in = np.array([new[p] for p in predictors]).reshape(1,-1)
        try:
            pred_model = float(modelo.predict(X_in)[0])
        except Exception:
            pred_model = float(new[target_col] * (1 + trend_rate))
        years_ahead = i+1
        decay = 0.92 ** years_ahead
        wm = weight_model * decay
        wt = 1 - wm
        pred_blend = wm * pred_model + wt * (new[target_col] * (1+trend_rate)**years_ahead)
        prev = float(base_row[target_col])
        change = (pred_blend / prev) - 1
        change = max(-0.10, min(0.10, change))
        final = prev * (1 + change)
        new[target_col] = final
        for col in df.columns:
            if col not in ['País','Ano','PIB_per_capita'] and not col.endswith('_lag1'):
                recent = dfp.tail(3)[col].dropna()
                if len(recent)>0:
                    mean_recent = float(recent.mean())
                    new[col] = float(base_row[col]*0.85 + mean_recent*0.15)
        rows.append(new)
        base_row = new
    df_proj = pd.concat([dfp, pd.DataFrame(rows)], ignore_index=True)
    df_proj['Ano'] = df_proj['Ano'].astype(int)
    if uncertainty and len(rows)>0:
        sims = 100
        sims_matrix = np.zeros((sims, len(rows)))
        rng = np.random.default_rng(42)
        for s in range(sims):
            base_row = ultimo.copy()
            path = []
            for i, ano in enumerate(anos_fut):
                new = base_row.copy()
                new['Ano'] = int(ano)
                for p in predictors:
                    base_name = p.replace('_lag1','')
                    if base_name in base_row.index:
                        new[p] = base_row[base_name]
                X_in = np.array([new[p] for p in predictors]).reshape(1,-1)
                try:
                    pred_model = float(modelo.predict(X_in)[0])
                except Exception:
                    pred_model = float(new[target_col] * (1 + trend_rate))
                decay = 0.92 ** (i+1)
                wm = weight_model * decay
                wt = 1 - wm
                pred_blend = wm * pred_model + wt * (new[target_col] * (1+trend_rate)**(i+1))
                prev = float(base_row[target_col])
                noise = rng.normal(0, resid_std)
                change = (pred_blend + noise) / prev - 1
                change = max(-0.12, min(0.12, change))
                final = prev * (1 + change)
                new[target_col] = final
                for col in df.columns:
                    if col not in ['País','Ano','PIB_per_capita'] and not col.endswith('_lag1'):
                        recent = dfp.tail(3)[col].dropna()
                        if len(recent)>0:
                            mean_recent = float(recent.mean())
                            new[col] = float(base_row[col]*0.85 + mean_recent*0.15)
                path.append(new[target_col])
                base_row = new
            sims_matrix[s,:] = path
        lower = np.percentile(sims_matrix, 5, axis=0)
        upper = np.percentile(sims_matrix, 95, axis=0)
        median = np.percentile(sims_matrix, 50, axis=0)
        df_proj = df_proj.reset_index(drop=True)
        hist_len = len(dfp)
        df_proj.loc[hist_len:hist_len+len(rows)-1,'PIB_lo'] = lower
        df_proj.loc[hist_len:hist_len+len(rows)-1,'PIB_hi'] = upper
        df_proj.loc[hist_len:hist_len+len(rows)-1,'PIB_med'] = median
    return df_proj

# -------------------- Improved sensitivity --------------------
def analise_sensibilidade_improved(df_model, pais, modelo, indicador_base, pct_variation=10, ano_horizon=2030):
    dfm = df_model.copy().reset_index()
    if pais not in dfm['País'].unique():
        raise ValueError("País não encontrado")
    base_proj = gerar_projecao_realista_improved(dfm, pais, modelo, ano_final=ano_horizon, uncertainty=False)
    lag_col = f"{indicador_base}_lag1" if not indicador_base.endswith('_lag1') else indicador_base
    df_shocked = dfm.copy()
    idx_last = df_shocked[(df_shocked['País']==pais)].index.max()
    if pd.isna(idx_last):
        raise ValueError("Dados insuficientes para o país")
    df_shocked.loc[idx_last, lag_col] = df_shocked.loc[idx_last, lag_col] * (1 + pct_variation/100.0)
    shocked_proj = gerar_projecao_realista_improved(df_shocked, pais, modelo, ano_final=ano_horizon, uncertainty=False)
    base_final = float(base_proj[base_proj['País']==pais].iloc[-1]['PIB_per_capita'])
    shocked_final = float(shocked_proj[shocked_proj['País']==pais].iloc[-1]['PIB_per_capita'])
    impact_pct = (shocked_final/base_final - 1) * 100
    elasticity = impact_pct / pct_variation if pct_variation != 0 else np.nan
    return {'base':base_proj, 'shocked':shocked_proj, 'impact_pct':impact_pct, 'elasticity':elasticity}

# -------------------- Figures (TCC & SHAP) --------------------
def fig_importancia_variaveis(modelo_xgboost, feature_names, out_path="figuras_tcc/Figura1_Importancia_Variaveis.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    importancias = getattr(modelo_xgboost, 'feature_importances_', None)
    if importancias is None:
        return None
    df_imp = pd.DataFrame({'Variável':[f.replace('_lag1','').replace('_',' ') for f in feature_names], 'Importancia':importancias*100}).sort_values('Importancia')
    fig, ax = plt.subplots(figsize=(10,8))
    colors = plt.cm.RdYlGn(np.linspace(0.3,0.9,len(df_imp)))
    ax.barh(df_imp['Variável'], df_imp['Importancia'], color=colors)
    for i,row in enumerate(df_imp.itertuples()):
        ax.text(row.Importancia+0.2, i, f"{row.Importancia:.2f}%", va='center', fontsize=9)
    ax.set_xlabel('Importância relativa (%)'); ax.set_title('Figura 1 - Importância das variáveis (XGBoost)')
    plt.tight_layout(); fig.savefig(out_path, dpi=300); plt.close()
    return out_path

def fig_comparacao_modelos(resultados_df, out_path="figuras_tcc/Figura2_Comparacao_Modelos.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1,3, figsize=(16,5))
    cores = ['#2ecc71' if i==0 else '#3498db' if i<3 else '#e74c3c' for i in range(len(resultados_df))]
    axes[0].barh(resultados_df['Modelo'], resultados_df['R²'], color=cores); axes[0].set_xlabel('R²'); axes[0].set_title('(a) R²')
    axes[1].barh(resultados_df['Modelo'], resultados_df['RMSE'], color=cores); axes[1].set_xlabel('RMSE'); axes[1].set_title('(b) RMSE')
    axes[2].barh(resultados_df['Modelo'], resultados_df['MAE'], color=cores); axes[2].set_xlabel('MAE'); axes[2].set_title('(c) MAE')
    plt.tight_layout(); fig.savefig(out_path, dpi=300); plt.close()
    return out_path

def fig_validacao_temporal(df_model, modelo, feature_names, out_path="figuras_tcc/Figura3_Validacao_Temporal.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = df_model.copy(); df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce')
    df_treino = df[df['Ano']<=2018]; df_teste = df[df['Ano']>2018]
    predictors = [c for c in df.columns if c.endswith('_lag1')]
    X_t = df_treino[predictors]; y_t = df_treino['PIB_per_capita']
    X_te = df_teste[predictors]; y_te = df_teste['PIB_per_capita']
    model_temp = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    model_temp.fit(X_t, y_t)
    ypt = model_temp.predict(X_t); ypte = model_temp.predict(X_te)
    r2_t = r2_score(y_t, ypt); r2_te = r2_score(y_te, ypte)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(y_t, ypt, alpha=0.4, s=30, label=f'Treino R²={r2_t:.4f}'); ax.scatter(y_te, ypte, alpha=0.7, s=50, marker='s', label=f'Teste R²={r2_te:.4f}')
    mn = min(y_t.min(), y_te.min()); mx = max(y_t.max(), y_te.max()); ax.plot([mn,mx],[mn,mx],'k--', linewidth=1)
    ax.set_xlabel('PIB real'); ax.set_ylabel('PIB previsto'); ax.set_title('Figura 3 - Validação Temporal')
    plt.tight_layout(); fig.savefig(out_path, dpi=300); plt.close()
    return out_path

def fig_cenarios_china(df_model, modelo, out_path="figuras_tcc/Figura4_Cenarios_China.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        cen = gerar_projecao_realista_improved(df_model, 'China', modelo, ano_final=2030, uncertainty=True)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(14,7))
    ultimo_ano_real = int(df_model['Ano'].astype(int).max())
    df_hist = cen[cen['Ano']<=ultimo_ano_real]
    ax.plot(df_hist['Ano'], df_hist['PIB_per_capita'], 'o-', color='black', linewidth=2, label='Histórico')
    mask = cen['Ano']>ultimo_ano_real
    if mask.any():
        ax.plot(cen.loc[mask,'Ano'], cen.loc[mask,'PIB_med'], 's--', label='Mediana (Simulação)')
        ax.fill_between(cen.loc[mask,'Ano'], cen.loc[mask,'PIB_lo'], cen.loc[mask,'PIB_hi'], alpha=0.2, label='90% intervalo')
    ax.set_title('Figura 4 - Cenários China (2025-2030)'); ax.set_xlabel('Ano'); ax.set_ylabel('PIB per capita (US$)'); ax.legend(); plt.tight_layout(); fig.savefig(out_path, dpi=300); plt.close()
    return out_path

def fig_ranking_crescimento(df_model, modelo, top_n=10, out_path="figuras_tcc/Figura5_Ranking_Crescimento.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cres = []
    # Use the countries available in the *input* df_model
    # Filter for countries that exist in the dataset
    available_countries = df_model['País'].unique()
    countries_to_process = [pais for pais in TODOS_PAISES if pais in available_countries]
    for pais in countries_to_process: # Iterate only over countries present in the data
        try:
            proj = gerar_projecao_realista_improved(df_model, pais, modelo, ano_final=2035, uncertainty=False)
            proj = proj.reset_index(drop=True)
            df_hist = proj[proj['Ano']<=2024]
            df_fut = proj[proj['Ano']>2024]
            if len(df_hist)>0 and len(df_fut)>0:
                pib_i = float(df_hist.iloc[-1]['PIB_per_capita'])
                pib_f = float(df_fut.iloc[-1]['PIB_per_capita'])
                anos = len(df_fut)
                taxa = (((pib_f/pib_i)**(1/anos))-1)*100
                # Ensure the column name matches the one used in sort_values
                cres.append({'País': pais, 'Taxa Anual (%)': taxa})
        except Exception:
            # If an error occurs for a specific country, just skip it and continue
            continue

    # Check if the cres list is empty before creating the DataFrame
    if not cres: # If cres is empty
        print(f"Debug: No data generated for ranking in fig_ranking_crescimento. Available countries: {list(available_countries)}")
        return None # Or return an empty plot if preferred

    df_rank = pd.DataFrame(cres).sort_values('Taxa Anual (%)', ascending=False).head(top_n)

    # Check if df_rank is empty after sorting and slicing
    if df_rank.empty:
        print("Debug: df_rank is empty after sorting and slicing.")
        return None # Or return an empty plot if preferred

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_rank)))
    ax.barh(df_rank['País'], df_rank['Taxa Anual (%)'], color=colors)
    for i, row in enumerate(df_rank.itertuples()):
        # Use getattr or row._2; row._1 is the index
        ax.text(row._2 + 0.1, i, f"{row._2:.1f}%", va='center', fontsize=9)
    ax.set_xlabel('Taxa anual (%)')
    ax.set_title('Figura 5 - Ranking de crescimento projetado (2025-2035)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def fig_shap_summary(modelo, X, feature_names, out_path="figuras_shap/Figura6_SHAP_Summary.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not SHAP_AVAILABLE:
        return None
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(12,10))
    shap.summary_plot(shap_values, X, feature_names=[f.replace('_lag1','').replace('_',' ') for f in feature_names], show=False, max_display=10)
    plt.title('Figura 6 - SHAP Summary'); plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    return out_path, explainer, shap_values

def fig_shap_dependence(X, shap_values, feature_names, out_path="figuras_shap/Figura7_SHAP_Dependence.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not SHAP_AVAILABLE:
        return None
    top_idx = list(range(min(4, X.shape[1])))
    fig, axes = plt.subplots(2,2, figsize=(16,12)); axes = axes.flatten()
    fnames = [f.replace('_lag1','').replace('_',' ') for f in feature_names]
    Xr = X.copy(); Xr.columns = fnames
    pairs = []
    if len(fnames)>=2:
        pairs = [(fnames[0], fnames[1]), (fnames[2%len(fnames)], fnames[1%len(fnames)]), (fnames[3%len(fnames)], fnames[0%len(fnames)]), (fnames[1%len(fnames)], fnames[2%len(fnames)])]
    for i,(a,b) in enumerate(pairs):
        axes[i].scatter(Xr[a], shap_values[:, i if i<shap_values.shape[1] else 0], c=Xr[b], alpha=0.6, s=20)
        axes[i].set_xlabel(a); axes[i].set_ylabel(f"SHAP {a}"); axes[i].set_title(f"Interação: {a} x {b}")
    plt.tight_layout(); fig.savefig(out_path, dpi=300); plt.close()
    return out_path

def fig_shap_casos_extremos(df_model, modelo, feature_names, out_path="figuras_shap/Figura8_Casos_Extremos.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = df_model.copy()
    predictors = feature_names
    X = df[predictors]; y = df['PIB_per_capita']; y_pred = modelo.predict(X)
    df['erro'] = y - y_pred; df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce')
    countries = df['País'].unique()[:5]
    fig, ax = plt.subplots(2,1, figsize=(14,10))
    for pais in countries:
        sub = df[df['País']==pais]
        if sub.empty: continue
        ax[0].plot(sub['Ano'], sub['erro'], marker='o', label=pais)
        ax[1].plot(sub['Ano'], (sub['erro']/sub['PIB_per_capita'])*100, marker='s', label=pais)
    ax[0].axhline(0, color='k', linestyle='--'); ax[0].set_ylabel('Erro (US$)'); ax[1].set_ylabel('Erro (%)')
    ax[0].legend(); ax[1].legend(); plt.tight_layout(); fig.savefig(out_path, dpi=300); plt.close()
    return out_path

# -------------------- Streamlit UI --------------------
def aba_geracao_figuras(df_model, models_data):
    st.header("🎨 Geração Automática de Figuras — TCC e SHAP")
    st.write("Gera e salva figuras do TCC (1–5) e SHAP (6–8).")
    modelo_xgboost = models_data['modelos'].get('XGBoost')
    feature_names = models_data['predictors']
    resultados_df = models_data['resultados']
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Gerar Figuras do TCC (1–5)"):
            with st.spinner("Gerando figuras TCC..."):
                out1 = fig_importancia_variaveis(modelo_xgboost, feature_names)
                out2 = fig_comparacao_modelos(resultados_df)
                out3 = fig_validacao_temporal(df_model, modelo_xgboost, feature_names)
                out4 = fig_cenarios_china(df_model, modelo_xgboost)
                out5 = fig_ranking_crescimento(df_model, modelo_xgboost)
                st.success("✅ Figuras do TCC (1–5) geradas com sucesso!")
                for i, out in enumerate([out1, out2, out3, out4, out5], 1):
                    if out:
                        st.write(f"Figura {i} salva em: `{out}`")
                    else:
                        st.warning(f"⚠️ Figura {i} não pôde ser gerada.")

    with col2:
        if SHAP_AVAILABLE and st.button("🔍 Gerar Figuras SHAP (6–8)"):
            with st.spinner("Gerando figuras SHAP..."):
                try:
                    out6, explainer, shap_vals = fig_shap_summary(modelo_xgboost, models_data['X'], feature_names)
                    out7 = fig_shap_dependence(models_data['X'], shap_vals, feature_names)
                    out8 = fig_shap_casos_extremos(df_model, modelo_xgboost, feature_names)
                    st.success("✅ Figuras SHAP (6–8) geradas com sucesso!")
                    for i, out in enumerate([out6, out7, out8], 6):
                        if out:
                            st.write(f"Figura {i} salva em: `{out}`")
                        else:
                            st.warning(f"⚠️ Figura {i} não pôde ser gerada.")
                except Exception as e:
                    st.error(f"Erro ao gerar figuras SHAP: {e}")
        elif not SHAP_AVAILABLE:
            st.info("📦 Pacote `shap` não disponível. Instale com: `pip install shap`")

def aba_projecao_pais(df_model, models_data):
    st.header("🌍 Projeção por País")
    paises = sorted(df_model['País'].unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        pais = st.selectbox("Selecione o país", paises)
    with col2:
        modelo_nome = st.selectbox("Modelo", list(models_data['modelos'].keys()))
    with col3:
        ano_final = st.slider("Ano final da projeção", 2026, 2040, 2035)
    modelo = models_data['modelos'][modelo_nome]
    if st.button("🚀 Gerar Projeção"):
        try:
            with st.spinner("Gerando projeção..."):
                proj_df = gerar_projecao_realista_improved(df_model, pais, modelo, ano_final=ano_final, uncertainty=True)
            st.subheader(f"Projeção para {pais} até {ano_final}")
            st.dataframe(proj_df[['Ano','PIB_per_capita'] + (['PIB_lo','PIB_med','PIB_hi'] if 'PIB_lo' in proj_df.columns else [])].tail(15))
            fig, ax = plt.subplots(figsize=(12,6))
            ultimo_ano_real = int(df_model[df_model['País']==pais]['Ano'].max())
            hist = proj_df[proj_df['Ano'] <= ultimo_ano_real]
            fut = proj_df[proj_df['Ano'] > ultimo_ano_real]
            ax.plot(hist['Ano'], hist['PIB_per_capita'], 'o-', color='black', label='Histórico')
            if not fut.empty:
                ax.plot(fut['Ano'], fut['PIB_med'], 's--', color='red', label='Mediana (Projeção)')
                ax.fill_between(fut['Ano'], fut['PIB_lo'], fut['PIB_hi'], color='red', alpha=0.2, label='Intervalo 90%')
            ax.set_title(f"Projeção de PIB per capita – {pais}")
            ax.set_xlabel("Ano"); ax.set_ylabel("PIB per capita (US$)")
            ax.legend(); st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro na projeção: {e}")

def aba_sensibilidade(df_model, models_data):
    st.header("📊 Análise de Sensibilidade")
    paises = sorted(df_model['País'].unique())
    indicadores = [c.replace('_lag1','') for c in models_data['predictors']]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pais = st.selectbox("País", paises, key="sens_pais")
    with col2:
        indicador = st.selectbox("Indicador", indicadores, key="sens_ind")
    with col3:
        pct = st.number_input("Variação (%)", -30.0, 30.0, 10.0, key="sens_pct")
    with col4:
        ano_horizon = st.number_input("Horizonte (ano)", 2026, 2040, 2030, key="sens_ano")
    modelo = models_data['modelos']['XGBoost']
    if st.button("🔬 Executar Análise de Sensibilidade"):
        try:
            with st.spinner("Executando análise..."):
                res = analise_sensibilidade_improved(df_model, pais, modelo, indicador, pct_variation=pct, ano_horizon=ano_horizon)
            st.success(f"Um choque de **{pct}%** em **{indicador}** leva a um impacto de **{res['impact_pct']:.2f}%** no PIB per capita em {ano_horizon}.")
            st.metric("Elasticidade", f"{res['elasticity']:.3f}")
            fig, ax = plt.subplots(figsize=(12,6))
            base = res['base']; shocked = res['shocked']
            ax.plot(base['Ano'], base['PIB_per_capita'], 'k-', label='Base')
            ax.plot(shocked['Ano'], shocked['PIB_per_capita'], 'r--', label=f'Choque (+{pct}%)')
            ax.axvline(x=ano_horizon, color='gray', linestyle=':', label=f'Horizonte ({ano_horizon})')
            ax.set_title(f"Sensibilidade: {indicador} → PIB per capita ({pais})")
            ax.set_xlabel("Ano"); ax.set_ylabel("PIB per capita (US$)"); ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro na análise: {e}")

def main():
    st.set_page_config(page_title="Código da Riqueza — Análise Econômica com ML", layout="wide")
    st.title("💰 Código da Riqueza")
    st.markdown("### Análise preditiva do PIB per capita com Machine Learning e dados do Banco Mundial")

    # Load data
    with st.spinner("Carregando dados do Banco Mundial..."):
        df_raw = carregar_dados_banco_mundial()
    if df_raw is None:
        st.stop()
    df, df_model = processar_dados(df_raw)
    if df_model is None:
        st.error("Falha no processamento dos dados.")
        st.stop()

    # Train models
    with st.spinner("Treinando modelos..."):
        models_data = treinar_todos_modelos(df_model)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparação de Modelos", "🌍 Projeção por País", "🔬 Sensibilidade", "🎨 Geração de Figuras"])

    with tab1:
        st.subheader("Tabela de Desempenho dos Modelos")
        st.dataframe(models_data['resultados'], use_container_width=True)
        st.markdown("**Melhor modelo:** " + models_data['resultados'].iloc[0]['Modelo'])

    with tab2:
        aba_projecao_pais(df_model, models_data)

    with tab3:
        aba_sensibilidade(df_model, models_data)

    with tab4:
        aba_geracao_figuras(df_model, models_data)

if __name__ == "__main__":
    main()
