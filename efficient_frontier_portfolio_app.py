# efficient_frontier_portfolio_app_v7_1.py

import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf

# ==============================================================================
# ⚙️ CONFIGURAÇÃO INSTITUCIONAL
# ==============================================================================
st.set_page_config(page_title="Grilli Research | Global Asset Standard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetric"] { background-color: rgba(28, 131, 225, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(28, 131, 225, 0.1); }
    .stButton>button { background-color: #1e3a8a; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("Terminal Quantitativo v7.1: Global Asset Standard")
st.write("Black-Litterman Framework | Rolling Walk-Forward Optimization")
st.write("---")

# ==============================================================================
# 🔢 CORE FUNCTIONS (RIGOR 10/10)
# ==============================================================================

def get_market_weights(prices):
    """Ponto de neutralidade institucional (Equal Weight) para o Prior de Equilíbrio."""
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
            # Matriz Omega de Incerteza (He & Litterman)
            conf = confidences.get(asset, 0.5) if confidences else 0.5
            variance = P[i] @ (tau * cov) @ P[i].T
            omega_diag.append(variance / conf)

    omega = np.diag(omega_diag)
    inv_prior = np.linalg.inv(tau * cov)
    inv_omega = np.linalg.inv(omega)
    
    # Posterior Return (Fórmula de Theil)
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
        
        # Estimadores Robustos
        lw = LedoitWolf().fit(train)
        cov = lw.covariance_ * 252
        w_mkt = get_market_weights(train)
        pi = 3.0 * (cov @ w_mkt) # 3.0 = Coeficiente de Aversão ao Risco Implícito
        pi = pd.Series(pi, index=prices.columns)
        
        mu_bl, cov_bl = black_litterman_full(pi, cov, views, confidences)
        
        # Otimização com Fricção de Mercado (Custos de Transação)
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
# 🚀 EXECUÇÃO & DASHBOARD
# ==============================================================================

with st.sidebar:
    st.header("Mandato & Convicções")
    tickers_input = st.text_input("Universo (Tickers):", "VALE3.SA, ITUB4.SA, PETR4.SA, AAPL, MSFT, BTC-USD")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    rf_rate = st.number_input("Risk-Free (Anual %)", 0.0, 20.0, 10.75) / 100
    t_cost = st.slider("Custo Transacional (bps)", 0, 100, 10) / 10000
    
    views, confidences = {}, {}
    for t in tickers:
        v = st.number_input(f"View {t} (%)", -50, 100, 0)
        c = st.slider(f"Confiança {t}", 0.1, 1.0, 0.5)
        if v != 0:
            views[t], confidences[t] = v/100, c

if st.button("GERAR DOSSIÊ QUANT 10/10"):
    with st.spinner("Computando Walk-Forward Optimization..."):
        bench = "^BVSP" if any(".SA" in t for t in tickers) else "^GSPC"
        df = Ticker(tickers + [bench]).history(start="2020-01-01")
        prices = df.reset_index().pivot(index='date', columns='symbol', values='adjclose').ffill().dropna()
        bench_rets = prices[bench].pct_change().dropna()
        asset_prices = prices.drop(columns=[bench])
        
        port_rets, weights_hist = rolling_bl_backtest(asset_prices, rf_rate, views, confidences, t_cost)
        
        # Estatísticas de Rigor
        cum = (1 + port_rets).cumprod() * 10000
        bench_cum = (1 + bench_rets.loc[port_rets.index]).cumprod() * 10000
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        ann_ret = port_rets.mean() * 252
        ann_vol = port_rets.std() * np.sqrt(252)
        sharpe = (ann_ret - rf_rate) / ann_vol
        
        st.header("Performance Out-of-Sample (Net of Costs)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno Anualizado", f"{ann_ret:.2%}")
        c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c3.metric("Max Drawdown", f"{mdd:.2%}")
        c4.metric("Information Ratio", f"{(ann_ret - bench_rets.loc[port_rets.index].mean()*252) / (np.std(port_rets - bench_rets.loc[port_rets.index])*np.sqrt(252)):.2f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Grilli Strategy", line=dict(color='#1e3a8a', width=3)))
        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name=f"Market ({bench})", line=dict(color='gray', dash='dot')))
        fig.update_layout(title="Equity Curve: Walk-Forward Validation", template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Composição Atual (Black-Litterman)")
        st.table(pd.DataFrame({'Alocação (%)': (weights_hist[-1]*100).round(2)}, index=asset_prices.columns).T)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Rafael Grilli - Grilli Research")
