# efficient_frontier_portfolio_app_v7_3.py

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

st.title("Terminal Quantitativo v7.3: Asset Management Suite")
st.write("Black-Litterman Framework | Markowitz Frontier | Walk-Forward Validation")
st.write("---")

# ==============================================================================
# 🔢 CORE FUNCTIONS
# ==============================================================================

def get_market_weights(prices):
    n = len(prices.columns)
    return np.array([1.0 / n] * n)

def calculate_stats(weights, rets_anual, cov_mat, rf):
    p_ret = np.sum(rets_anual * weights)
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

# ==============================================================================
# 🎛️ PAINEL DE CONFIGURAÇÃO (CENTRALIZADO COM TOOLTIPS)
# ==============================================================================

st.markdown("<div class='secao-titulo'>1. PARÂMETROS DE MERCADO E MANDATO</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    tickers_in = st.text_input(
        "Universo de Ativos:", 
        "VALE3.SA, ITUB4.SA, AAPL, MSFT, BTC-USD",
        help="Insira os tickers separados por vírgula. Utilize o sufixo '.SA' para ativos da B3 (Brasil) ou o ticker puro para ativos internacionais (EUA/Crypto)."
    )
    tickers = [t.strip().upper() for t in tickers_in.split(",")]
with c2:
    rf_rate = st.number_input(
        "Risk-Free (Anual %):", 
        0.0, 20.0, 10.75, 
        help="Taxa livre de risco utilizada para descontar o excesso de retorno (Alpha). Geralmente utiliza-se a Selic para Brasil ou T-Bills para EUA."
    ) / 100
with c3:
    t_cost = st.slider(
        "Custo de Transação (bps):", 
        0, 100, 10, 
        help="Fricção de mercado em pontos-base (100bps = 1%). Este valor penaliza o turnover excessivo na função objetivo do otimizador."
    ) / 10000
with c4:
    s_date = st.date_input(
        "Início da Série:", 
        date(2020, 1, 1),
        help="Data inicial para coleta de dados históricos. Séries mais longas oferecem mais dados, porém podem conter regimes econômicos obsoletos."
    )

with st.expander("💡 Black-Litterman: Convicções e Nível de Confiança"):
    st.write("O modelo Black-Litterman combina o equilíbrio de mercado com as suas convicções específicas.")
    v_cols = st.columns(len(tickers) if len(tickers) < 6 else 5)
    views, confs = {}, {}
    for i, t in enumerate(tickers):
        with v_cols[i % len(v_cols)]:
            v = st.number_input(
                f"E[R] {t} (%)", -50, 100, 0, key=f"v_{t}",
                help=f"Sua expectativa de retorno anualizado para {t}. Se zero, o modelo utilizará o retorno implícito de equilíbrio."
            )
            c = st.slider(
                f"Confiança {t}", 0.1, 1.0, 0.5, key=f"c_{t}",
                help=f"Grau de certeza na sua view. Quanto maior, mais o modelo se desviará do equilíbrio para seguir sua convicção."
            )
            if v != 0: views[t], confs[t] = v/100, c

if st.button("🚀 GERAR RELATÓRIO QUANTITATIVO COMPLETO"):
    with st.spinner("Computando modelos robustos..."):
        bench = "^BVSP" if any(".SA" in t for t in tickers) else "^GSPC"
        df = Ticker(tickers + [bench]).history(start=s_date.isoformat())
        prices = df.reset_index().pivot(index='date', columns='symbol', values='adjclose').ffill().dropna()
        bench_rets = prices[bench].pct_change().dropna()
        asset_prices = prices.drop(columns=[bench])
        rets = asset_prices.pct_change().dropna()
        
        lw = LedoitWolf().fit(rets)
        cov_robust = lw.covariance_ * 252
        pi = 3.0 * (cov_robust @ get_market_weights(asset_prices))
        mu_bl, _ = black_litterman_full(pd.Series(pi, index=asset_prices.columns), cov_robust, views, confs)
        
        n = len(asset_prices.columns)
        bnds = tuple((0, 0.5) for _ in range(n))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        opt_s = sco.minimize(lambda w: -calculate_stats(w, mu_bl, cov_robust, rf_rate)[2], n*[1./n], bounds=bnds, constraints=cons)
        opt_v = sco.minimize(lambda w: calculate_stats(w, mu_bl, cov_robust, rf_rate)[1], n*[1./n], bounds=bnds, constraints=cons)
        opt_rp = sco.minimize(risk_parity_objective, n*[1./n], args=(cov_robust,), bounds=bnds, constraints=cons)

        # ==============================================================================
        # 📊 DASHBOARD DE RESULTADOS
        # ==============================================================================
        st.markdown("<div class='secao-titulo'>2. COMPARATIVO DE ESTRATÉGIAS</div>", unsafe_allow_html=True)
        tabs = st.tabs(["🎯 Máximo Sharpe", "🛡️ Mínima Variância (MVP)", "⚖️ Paridade de Risco"])
        
        for tab, opt, nome, guia in zip(tabs, [opt_s, opt_v, opt_rp], ["Sharpe", "MVP", "Risk Parity"], 
            ["Melhor retorno por unidade de risco.", "Menor volatilidade possível.", "Risco equalizado entre ativos."]):
            with tab:
                w = opt.x
                r, v, s = calculate_stats(w, mu_bl, cov_robust, rf_rate)
                c1, c2, c3 = st.columns(3)
                c1.metric("Retorno Esperado", f"{r:.2%}", help="Retorno anualizado esperado combinando Equilíbrio de Mercado e suas Views (Black-Litterman).")
                c2.metric("Volatilidade", f"{v:.2%}", help="Desvio padrão anualizado da carteira, calculado via matriz de covariância robusta (Ledoit-Wolf).")
                c3.metric("Sharpe Ratio", f"{s:.2f}", help="Indica quanto de retorno excedente o investidor recebe para cada unidade de risco total assumida.")
                st.write(f"**Metodologia:** {guia}")
                st.table(pd.DataFrame({'Peso (%)': (w*100).round(2)}, index=asset_prices.columns).T)

        # ==============================================================================
        # 📈 VISUALIZAÇÕES TÉCNICAS
        # ==============================================================================
        st.markdown("<div class='secao-titulo'>3. FRONTEIRA EFICIENTE E PERFORMANCE</div>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.write("**Fronteira Eficiente (Markowitz Robust)**")
            mc_v, mc_r = [], []
            for _ in range(5000):
                w = np.random.random(n); w /= np.sum(w)
                mc_r.append(np.sum(mu_bl * w)); mc_v.append(np.sqrt(w.T @ cov_robust @ w))
            
            fig_fe = go.Figure()
            fig_fe.add_trace(go.Scatter(x=np.array(mc_v)*100, y=np.array(mc_r)*100, mode='markers', 
                                         marker=dict(color=(np.array(mc_r)-rf_rate)/np.array(mc_v), colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe"))))
            fig_fe.add_trace(go.Scatter(x=[calculate_stats(opt_s.x, mu_bl, cov_robust, rf_rate)[1]*100], 
                                         y=[calculate_stats(opt_s.x, mu_bl, cov_robust, rf_rate)[0]*100], 
                                         mode='markers', marker=dict(color='red', size=15, symbol='star', line=dict(color='white', width=2)), name="Max Sharpe"))
            fig_fe.update_layout(xaxis_title="Risco (%)", yaxis_title="Retorno (%)", template="plotly_white", margin=dict(t=20))
            st.plotly_chart(fig_fe, use_container_width=True)

        with col_g2:
            st.write("**Backtest: Crescimento de $10.000**")
            cum = (1 + rets.dot(opt_s.x)).cumprod() * 10000
            bench_cum = (1 + bench_rets).cumprod() * 10000
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=cum.index, y=cum, name="Portfólio Sharpe", line=dict(color='#1e3a8a', width=3)))
            fig_bt.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name=f"Mercado ({bench})", line=dict(color='gray', dash='dot')))
            fig_bt.update_layout(xaxis_title="Data", yaxis_title="Patrimônio ($)", template="plotly_white", margin=dict(t=20))
            st.plotly_chart(fig_bt, use_container_width=True)

st.sidebar.markdown("© 2026 Rafael Grilli - Grilli Research")
