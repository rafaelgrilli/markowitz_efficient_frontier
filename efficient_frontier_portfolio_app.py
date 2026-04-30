# efficient_frontier_portfolio_app_FINAL_10_10.py

import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf

# ==============================================================================
# ⚙️ ESTILO E IDENTIDADE (DENSIDADE ANALÍTICA)
# ==============================================================================
st.set_page_config(page_title="Grilli Analytics | Institutional Terminal", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetric"] { 
        background-color: rgba(28, 131, 225, 0.05); 
        padding: 15px; border-radius: 8px; border: 1px solid rgba(28, 131, 225, 0.1); 
    }
    .stButton>button { 
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
        color: white; font-weight: bold; width: 100%; height: 3.5em; border: none;
    }
    .nota-metrica { font-size: 0.85rem; color: #666; font-style: italic; margin-top: -10px; margin-bottom: 15px; }
    .secao-titulo { color: #1e3a8a; font-weight: bold; border-bottom: 2px solid #1e3a8a; padding-bottom: 5px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Terminal Quantitativo v7.3: Institutional Asset Suite")
st.write("Black-Litterman Framework | Markowitz Frontier | Walk-Forward Validation")
st.write("---")

# ==============================================================================
# 🔢 CORE FUNCTIONS (RIGOR TOTAL)
# ==============================================================================

def get_market_weights(prices):
    """Ponto de neutralidade institucional (Inverse Volatility) para o Prior."""
    vols = prices.pct_change().std()
    inv_vol = 1 / vols
    w = inv_vol / inv_vol.sum()
    return w.values

def calculate_stats(weights, mu, cov_mat, rf):
    p_ret = np.sum(mu * weights)
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0
    return p_ret, p_vol, p_sharpe

def risk_parity_objective(weights, cov_mat):
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)
    marginal_risk = (cov_mat @ weights) / p_vol
    risk_contribution = weights * marginal_risk
    return np.sum(np.square(risk_contribution - (p_vol / len(weights))))

def black_litterman_full(mu_prior, cov, views_dict, confidences=None, tau=0.05):
    n = len(mu_prior)
    if not views_dict: return mu_prior, cov
    P, Q = np.zeros((len(views_dict), n)), np.zeros(len(views_dict))
    assets = list(mu_prior.index)
    omega_diag = []
    for i, (asset, view_val) in enumerate(views_dict.items()):
        if asset in assets:
            idx = assets.index(asset)
            P[i, idx], Q[i] = 1, view_val
            conf = confidences.get(asset, 0.5) if confidences else 0.5
            omega_diag.append((P[i] @ (tau * cov) @ P[i].T) / conf)
    omega = np.diag(omega_diag)
    inv_prior, inv_omega = np.linalg.inv(tau * cov), np.linalg.inv(omega)
    term1 = np.linalg.inv(inv_prior + P.T @ inv_omega @ P)
    mu_bl = term1 @ (inv_prior @ mu_prior + P.T @ inv_omega @ Q)
    return pd.Series(mu_bl, index=assets), cov

def rolling_backtest(prices, rf, t_cost, views, confs, window=252, rebalance=21):
    """Motor de validação walk-forward (Out-of-Sample) com fricção de custos."""
    rets = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    w_prev = np.array([1/n_assets]*n_assets)
    port_rets, dates = [], []

    for i in range(window, len(rets)-rebalance, rebalance):
        train = rets.iloc[i-window:i]
        test = rets.iloc[i:i+rebalance]
        lw = LedoitWolf().fit(train)
        cov = lw.covariance_ * 252
        
        # Prior de mercado e ajuste Black-Litterman
        w_mkt = get_market_weights(train)
        pi = 3.0 * (cov @ w_mkt)
        mu_bl, cov_bl = black_litterman_full(pd.Series(pi, index=prices.columns), cov, views, confs)

        def obj(w):
            r = np.sum(mu_bl * w)
            v = np.sqrt(w.T @ cov_bl @ w)
            turnover = np.sum(np.abs(w - w_prev))
            cost = t_cost * turnover
            return -(r - cost - rf) / v

        res = sco.minimize(obj, w_prev, bounds=tuple((0,0.5) for _ in range(n_assets)),
                           constraints={'type':'eq','fun': lambda x: np.sum(x)-1})

        if res.success:
            w_prev = res.x
            r_test = test.dot(w_prev)
            port_rets.extend(r_test.tolist())
            dates.extend(test.index.tolist())
    return pd.Series(port_rets, index=dates)

# ==============================================================================
# 🎛️ PAINEL DE CONFIGURAÇÃO (CENTRALIZADO)
# ==============================================================================

st.markdown("<div class='secao-titulo'>1. PARÂMETROS DE MERCADO E MANDATO</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    tickers_in = st.text_input("Universo de Ativos:", "VALE3.SA, ITUB4.SA, AAPL, MSFT, BTC-USD", 
                               help="Tickers via Yahoo Finance. Brasil: .SA | EUA: puro.")
    tickers = [t.strip().upper() for t in tickers_in.split(",")]
with c2:
    rf_rate = st.number_input("Risk-Free (Anual %):", 0.0, 20.0, 10.75, help="Taxa livre de risco para Alpha.") / 100
with c3:
    t_cost = st.slider("Custo de Transação (bps):", 0, 100, 10, help="Fricção de mercado (100bps = 1%).") / 10000
with c4:
    s_date = st.date_input("Início da Série:", date(2020, 1, 1), help="Início da coleta histórica.")

with st.expander("💡 Black-Litterman: Convicções e Nível de Confiança"):
    v_cols = st.columns(len(tickers) if len(tickers) < 6 else 5)
    views, confs = {}, {}
    for i, t in enumerate(tickers):
        with v_cols[i % len(v_cols)]:
            v = st.number_input(f"E[R] {t} (%)", -50, 100, 0, key=f"v_{t}", help=f"Retorno esperado para {t}.")
            c = st.slider(f"Confiança {t}", 0.1, 1.0, 0.5, key=f"c_{t}", help="Grau de certeza na sua view.")
            if v != 0: views[t], confs[t] = v/100, c

if st.button("🚀 GERAR RELATÓRIO QUANTITATIVO COMPLETO"):
    with st.spinner("Processando Walk-Forward e Estimadores Robustos..."):
        bench = "^BVSP" if any(".SA" in t for t in tickers) else "^GSPC"
        df = Ticker(tickers + [bench]).history(start=s_date.isoformat())
        prices = df.reset_index().pivot(index='date', columns='symbol', values='adjclose').ffill().dropna()
        bench_rets = prices[bench].pct_change().dropna()
        asset_prices = prices.drop(columns=[bench])
        rets = asset_prices.pct_change().dropna()
        
        # 1. Estimadores Robustos (Ledoit-Wolf + BL)
        lw = LedoitWolf().fit(rets)
        cov_robust = lw.covariance_ * 252
        w_mkt = get_market_weights(asset_prices)
        pi = 3.0 * (cov_robust @ w_mkt)
        mu_bl, _ = black_litterman_full(pd.Series(pi, index=asset_prices.columns), cov_robust, views, confs)
        
        # 2. Otimizações Estáticas
        n = len(asset_prices.columns)
        bnds = tuple((0, 0.5) for _ in range(n))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        opt_s = sco.minimize(lambda w: -calculate_stats(w, mu_bl, cov_robust, rf_rate)[2], n*[1./n], bounds=bnds, constraints=cons)
        opt_v = sco.minimize(lambda w: calculate_stats(w, mu_bl, cov_robust, rf_rate)[1], n*[1./n], bounds=bnds, constraints=cons)
        opt_rp = sco.minimize(risk_parity_objective, n*[1./n], args=(cov_robust,), bounds=bnds, constraints=cons)

        # 3. Backtest Rolling (O Coração da Validação)
        rolling = rolling_backtest(asset_prices, rf_rate, t_cost, views, confs)
        bench_aligned = bench_rets.loc[rolling.index]

        # ==============================================================================
        # 📊 OUTPUT - PARTE 1: PERFORMANCE REAL (OUT-OF-SAMPLE)
        # ==============================================================================
        st.markdown("<div class='secao-titulo'>2. PERFORMANCE OUT-OF-SAMPLE (REALIZADA)</div>", unsafe_allow_html=True)
        ann_ret = rolling.mean()*252
        ann_vol = rolling.std()*np.sqrt(252)
        sharpe = (ann_ret - rf_rate)/ann_vol
        tracking_error = np.std(rolling - bench_aligned)*np.sqrt(252)
        info_ratio = (ann_ret - bench_aligned.mean()*252) / tracking_error

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno Real", f"{ann_ret:.2%}", help="Retorno médio anualizado no backtest rolling.")
        c2.metric("Sharpe Real", f"{sharpe:.2f}", help="Retorno/Risco realizado net de custos.")
        c3.metric("Information Ratio", f"{info_ratio:.2f}", help="Alpha gerado por unidade de tracking error.")
        c4.metric("Max Drawdown", f"{(( (1+rolling).cumprod() / (1+rolling).cumprod().cummax() ) - 1).min():.2%}")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=rolling.index, y=(1+rolling).cumprod()*10000, name="Estratégia (Rolling)", line=dict(color='#1e3a8a', width=3)))
        fig_bt.add_trace(go.Scatter(x=bench_aligned.index, y=(1+bench_aligned).cumprod()*10000, name="Benchmark", line=dict(color='gray', dash='dot')))
        fig_bt.update_layout(title="Equity Curve: Walk-Forward Validation ($10k Init)", template="plotly_white", height=500)
        st.plotly_chart(fig_bt, use_container_width=True)

        # ==============================================================================
        # 📊 OUTPUT - PARTE 2: FRONTEIRA E MÉTODOS
        # ==============================================================================
        st.markdown("<div class='secao-titulo'>3. ANÁLISE DE ALOCAÇÃO E FRONTEIRA</div>", unsafe_allow_html=True)
        tabs = st.tabs(["🎯 Máximo Sharpe", "🛡️ Mínima Variância", "⚖️ Paridade de Risco"])
        for tab, opt, guia in zip(tabs, [opt_s, opt_v, opt_rp], ["Equilíbrio Risco/Retorno", "Foco em Proteção", "Equilíbrio de Contribuição"]):
            with tab:
                w = opt.x
                r, v, s = calculate_stats(w, mu_bl, cov_robust, rf_rate)
                st.write(f"**Mandato:** {guia}")
                st.table(pd.DataFrame({'Peso (%)': (w*100).round(2)}, index=asset_prices.columns).T)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**Fronteira Eficiente (Markowitz Robust)**")
            mc_v, mc_r = [], []
            for _ in range(3000):
                w = np.random.random(n); w /= np.sum(w)
                mc_r.append(np.sum(mu_bl * w)); mc_v.append(np.sqrt(w.T @ cov_robust @ w))
            fig_fe = go.Figure()
            fig_fe.add_trace(go.Scatter(x=np.array(mc_v)*100, y=np.array(mc_r)*100, mode='markers', marker=dict(color=(np.array(mc_r)-rf_rate)/np.array(mc_v), colorscale='Viridis', showscale=True)))
            fig_fe.add_trace(go.Scatter(x=[calculate_stats(opt_s.x, mu_bl, cov_robust, rf_rate)[1]*100], y=[calculate_stats(opt_s.x, mu_bl, cov_robust, rf_rate)[0]*100], mode='markers', marker=dict(color='red', size=15, symbol='star')))
            fig_fe.update_layout(xaxis_title="Risco (%)", yaxis_title="Retorno (%)", template="plotly_white")
            st.plotly_chart(fig_fe, use_container_width=True)
        
        with col_g2:
            st.write("**Matriz de Correlação Robusta (Ledoit-Wolf)**")
            st.dataframe(pd.DataFrame(rets.corr().round(2)))

st.sidebar.markdown("© 2026 Rafael Grilli - Grilli Research")
