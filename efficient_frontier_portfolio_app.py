# efficient_frontier_portfolio_app_v7_2.py

import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf

# ==============================================================================
# ⚙️ CONFIGURAÇÃO INSTITUCIONAL (UX TOTALMENTE VISÍVEL)
# ==============================================================================
st.set_page_config(page_title="Grilli Research | Global Asset Standard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetric"] { background-color: rgba(28, 131, 225, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(28, 131, 225, 0.1); }
    .stButton>button { background-color: #1e3a8a; color: white; font-weight: bold; width: 100%; height: 3.5em; border-radius: 5px; }
    .config-section { padding: 20px; border-radius: 10px; background-color: rgba(150, 150, 150, 0.05); border: 1px solid rgba(150, 150, 150, 0.1); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Terminal Quantitativo v7.2: Global Asset Standard")
st.write("Black-Litterman Framework | Rolling Walk-Forward Optimization")
st.write("---")

# ==============================================================================
# 🔢 CORE FUNCTIONS
# ==============================================================================

def get_market_weights(prices):
    n = len(prices.columns)
    return np.array([1.0 / n] * n)

def black_litterman_full(mu_prior, cov, views_dict, confidences=None, tau=0.05):
    n = len(mu_prior)
    if not views_dict:
        return mu_prior, cov
    
    P = np.zeros((len(views_dict), n))
    Q = np.zeros(len(views_dict))
    assets = list(mu_prior.index)
    omega_diag = []

    for i, (asset, view_val) in enumerate(views_dict.items()):
        if asset in assets:
            idx = assets.index(asset)
            P[i, idx] = 1
            Q[i] = view_val
            conf = confidences.get(asset, 0.5) if confidences else 0.5
            variance = P[i] @ (tau * cov) @ P[i].T
            omega_diag.append(variance / conf)

    omega = np.diag(omega_diag)
    inv_prior = np.linalg.inv(tau * cov)
    inv_omega = np.linalg.inv(omega)
    term1 = np.linalg.inv(inv_prior + P.T @ inv_omega @ P)
    term2 = inv_prior @ mu_prior + P.T @ inv_omega @ Q
    mu_bl = term1 @ term2
    
    return pd.Series(mu_bl, index=assets), cov

def rolling_bl_backtest(prices, rf, views, confidences, t_cost=0.001, window=252, rebalance=21):
    rets = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    last_w = np.array([1/n_assets]*n_assets)
    port_rets, dates, weights_history = [], [], []
    
    for i in range(window, len(rets)-rebalance, rebalance):
        train = rets.iloc[i-window:i]
        test = rets.iloc[i:i+rebalance]
        lw = LedoitWolf().fit(train)
        cov = lw.covariance_ * 252
        w_mkt = get_market_weights(train)
        pi = 3.0 * (cov @ w_mkt)
        pi = pd.Series(pi, index=prices.columns)
        mu_bl, cov_bl = black_litterman_full(pi, cov, views, confidences)
        
        def obj(w):
            r = np.sum(mu_bl * w)
            v = np.sqrt(w.T @ cov_bl @ w)
            cost = t_cost * np.sum(np.abs(w - last_w))
            return -(r - rf - cost) / v
        
        res = sco.minimize(obj, last_w, bounds=tuple((0,0.5) for _ in range(n_assets)),
                           constraints=({'type':'eq','fun':lambda x: np.sum(x)-1}))
        
        if res.success:
            last_w = res.x
            weights_history.append(last_w.copy())
            port_rets.extend(test.dot(last_w).tolist())
            dates.extend(test.index.tolist())
    
    return pd.Series(port_rets, index=dates), weights_history

# ==============================================================================
# 🎛️ INTERFACE PRINCIPAL (SEM SIDEBAR)
# ==============================================================================

# 1. Definição do Universo e Mandato
st.subheader("1. Definição do Universo e Mandato")
col_tickers, col_dates = st.columns([2, 1])

with col_tickers:
    tickers_input = st.text_input("Ativos (tickers separados por vírgula):", "VALE3.SA, ITUB4.SA, PETR4.SA, AAPL, MSFT, BTC-USD")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    st.caption("Brasil: .SA (PETR4.SA) | Internacional: Ticker puro (AAPL)")

with col_dates:
    s_date = st.date_input("Início da Série Histórica:", date(2020, 1, 1))

# 2. Parametrização de Risco e Custos
st.markdown("<div class='config-section'>", unsafe_allow_html=True)
st.subheader("2. Parametrização de Risco e Custos")
c_rf, c_cost, c_window = st.columns(3)

with c_rf:
    rf_rate = st.number_input("Risk-Free (Anual %)", 0.0, 20.0, 10.75, help="Taxa livre de risco para cálculo de excesso de retorno.") / 100
with c_cost:
    t_cost = st.slider("Custo Transacional (bps)", 0, 100, 10, help="Fricção de mercado aplicada no rebalanceamento.") / 10000
with c_window:
    window_val = st.selectbox("Janela de Rolling (Dias)", options=[126, 252, 504], index=1, help="Período de treino para cada rebalanceamento.")
st.markdown("</div>", unsafe_allow_html=True)

# 3. Convicções (Views) de Black-Litterman
with st.expander("3. Convicções de Black-Litterman (Opcional)", expanded=False):
    st.write("Insira suas expectativas de retorno e nível de confiança para ativos específicos.")
    views, confidences = {}, {}
    cols_views = st.columns(len(tickers) if len(tickers) < 5 else 4)
    
    for i, t in enumerate(tickers):
        with cols_views[i % len(cols_views)]:
            v = st.number_input(f"Retorno {t} (%)", -50, 100, 0, key=f"v_{t}")
            c = st.slider(f"Confiança {t}", 0.1, 1.0, 0.5, key=f"c_{t}")
            if v != 0:
                views[t], confidences[t] = v/100, c

# Botão de Execução
if st.button("GERAR RELATÓRIO QUANT COMPLETO"):
    with st.spinner("Processando Walk-Forward Optimization e Estimadores de Ledoit-Wolf..."):
        bench = "^BVSP" if any(".SA" in t for t in tickers) else "^GSPC"
        try:
            df = Ticker(tickers + [bench]).history(start=s_date.isoformat())
            prices = df.reset_index().pivot(index='date', columns='symbol', values='adjclose').ffill().dropna()
            bench_rets = prices[bench].pct_change().dropna()
            asset_prices = prices.drop(columns=[bench])
            
            port_rets, weights_hist = rolling_bl_backtest(asset_prices, rf_rate, views, confidences, t_cost, window=window_val)
            
            # --- DASHBOARD DE RESULTADOS ---
            st.write("---")
            st.header("Análise de Performance Out-of-Sample")
            
            cum = (1 + port_rets).cumprod() * 10000
            bench_cum = (1 + bench_rets.loc[port_rets.index]).cumprod() * 10000
            mdd = ((cum - cum.cummax()) / cum.cummax()).min()
            ann_ret = port_rets.mean() * 252
            ann_vol = port_rets.std() * np.sqrt(252)
            sharpe = (ann_ret - rf_rate) / ann_vol
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retorno Anualizado", f"{ann_ret:.2%}")
            c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c3.metric("Max Drawdown", f"{mdd:.2%}")
            c4.metric("Information Ratio", f"{(ann_ret - bench_rets.loc[port_rets.index].mean()*252) / (np.std(port_rets - bench_rets.loc[port_rets.index])*np.sqrt(252)):.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Estratégia Robusta", line=dict(color='#1e3a8a', width=3)))
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name=f"Benchmark ({bench})", line=dict(color='gray', dash='dot')))
            fig.update_layout(title="Equity Curve: Validação Walk-Forward (Net of Costs)", template="plotly_white", height=500, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Última Alocação Sugerida")
            st.table(pd.DataFrame({'Peso (%)': (weights_hist[-1]*100).round(2)}, index=asset_prices.columns).T)
            
        except Exception as e:
            st.error(f"Erro no processamento: {e}")

st.write("---")
st.caption("© 2026 Rafael Grilli - Grilli Research | Dados via Yahoo Finance")
