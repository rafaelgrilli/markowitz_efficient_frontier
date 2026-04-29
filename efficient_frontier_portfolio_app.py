# efficient_frontier_portfolio_app.py

import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import io
import scipy.optimize as sco

# ==============================================================================
# ⚙️ CONFIGURAÇÃO E ESTILO (CONTRASTE DINÂMICO)
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Analytics - Grilli Research", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS que se adapta ao Modo Claro/Escuro do navegador
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(150, 150, 150, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(150, 150, 150, 0.2);
    }
    .instrucao-ticker { 
        padding: 12px; 
        border-radius: 5px;
        border-left: 5px solid #2e7d32; 
        background-color: rgba(46, 125, 50, 0.1);
        font-size: 0.95rem; 
        margin-bottom: 20px;
    }
    /* Garante que o texto das tabelas acompanhe o tema */
    .stTable { color: inherit; }
    </style>
    """, unsafe_allow_html=True)

st.title("Simulador de Portfólio: Markowitz & Risk Parity")
st.write("---")

# ==============================================================================
# 🔢 FUNÇÕES ANALÍTICAS
# ==============================================================================

def calculate_stats(weights, rets_anual, cov_mat, rf):
    p_ret = np.sum(rets_anual * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 1e-6 else 0
    return p_ret, p_vol, p_sharpe

def calculate_advanced_metrics(rets_series, weights, rf_diario):
    p_rets = rets_series.dot(weights)
    downside_rets = p_rets[p_rets < 0]
    down_std = np.std(downside_rets) * np.sqrt(252)
    p_ret_anual = p_rets.mean() * 252
    sortino = (p_ret_anual - (rf_diario * 252)) / down_std if down_std > 0 else 0
    cum_rets = (1 + p_rets).cumprod()
    peak = cum_rets.cummax()
    max_dd = ((cum_rets - peak) / peak).min()
    sorted_rets = np.sort(p_rets)
    idx = int(0.05 * len(sorted_rets))
    var_95 = -sorted_rets[idx] * np.sqrt(252)
    cvar_95 = -sorted_rets[:idx].mean() * np.sqrt(252)
    return sortino, max_dd, var_95, cvar_95

def risk_parity_objective(weights, cov_mat):
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    marginal_risk = np.dot(cov_mat, weights) / p_vol
    risk_contribution = weights * marginal_risk
    target_risk = p_vol / len(weights)
    return np.sum(np.square(risk_contribution - target_risk))

# ==============================================================================
# 📥 GESTÃO DE DADOS
# ==============================================================================

@st.cache_data(ttl=3600)
def get_consolidated_data(tickers, start, end):
    try:
        bench = "^BVSP" if any(".SA" in t.upper() for t in tickers) else "^GSPC"
        full_list = list(set(tickers + [bench]))
        t_obj = Ticker(full_list)
        df = t_obj.history(start=start, end=end)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty) or isinstance(df, str):
            return None, None, None
        df = df.reset_index()
        col = 'adjclose' if 'adjclose' in df.columns else 'close'
        prices = df.pivot(index='date', columns='symbol', values=col).ffill().dropna()
        if bench not in prices.columns: return prices, None, None
        return prices.drop(columns=[bench]), prices[bench], bench
    except: return None, None, None

# ==============================================================================
# 🎛️ ENTRADA DE DADOS
# ==============================================================================

col_main, col_side = st.columns([2, 1])
with col_main:
    tickers_in = st.text_input("Ativos para análise:", "VALE3.SA, ITUB4.SA, AAPL, MSFT")
    st.markdown("""<div class='instrucao-ticker'>💡 <b>Sintaxe:</b> Brasil (ex: PETR4.SA) | Internacional (ex: AAPL)</div>""", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("Início:", date(2021, 1, 1))
    with c2: e_date = st.date_input("Fim:", date.today())
    with c3: n_sim = st.slider("Simulações Monte Carlo:", 1000, 20000, 5000)

with col_side:
    rf_rate = st.number_input("Taxa Livre de Risco (Anual %):", 0.0, 0.20, 0.1075, step=0.0025, format="%.4f")
    allow_short = st.checkbox("Venda a Descoberto (Short)", value=False)

st.subheader("Restrições de Peso")
c_min, c_max = st.columns(2)
with c_min: min_w = st.number_input("Mínimo por ativo:", -1.0 if allow_short else 0.0, 1.0, 0.0)
with c_max: max_w = st.number_input("Máximo por ativo:", 0.0, 2.0, 1.0)

# ==============================================================================
# 🚀 PROCESSAMENTO E DASHBOARD
# ==============================================================================

if st.button("Executar Análise de Portfólio", use_container_width=True):
    t_list = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    
    with st.spinner("Analisando mercado..."):
        prices, bench_prices, bench_ticker = get_consolidated_data(t_list, s_date.isoformat(), e_date.isoformat())
        
        if prices is not None:
            rets = prices.pct_change().dropna()
            rets_a = rets.mean() * 252
            cov_a = rets.cov() * 252
            n_assets = len(prices.columns)
            rf_d = rf_rate / 252

            bnds = tuple((min_w, max_w) for _ in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init = n_assets * [1./n_assets]
            
            opt_sharpe = sco.minimize(lambda w: -calculate_stats(w, rets_a, cov_a, rf_rate)[2], init, method='SLSQP', bounds=bnds, constraints=cons)
            opt_mvp = sco.minimize(lambda w: calculate_stats(w, rets_a, cov_a, rf_rate)[1], init, method='SLSQP', bounds=bnds, constraints=cons)
            opt_rp = sco.minimize(risk_parity_objective, init, args=(cov_a,), method='SLSQP', bounds=bnds, constraints=cons)

            st.header("1. Comparativo de Estratégias")
            estrategias = [
                ("🎯 Máximo Sharpe", opt_sharpe.x, "Melhor equilíbrio risco/retorno."),
                ("🛡️ Mínima Variância (MVP)", opt_mvp.x, "Menor volatilidade possível."),
                ("⚖️ Paridade de Risco", opt_rp.x, "Contribuição de risco igualitária.")
            ]
            
            cols = st.columns(3)
            for i, (nome, pesos, desc) in enumerate(estrategias):
                with cols[i]:
                    st.subheader(nome)
                    r, v, s = calculate_stats(pesos, rets_a, cov_a, rf_rate)
                    sortino, mdd, var, cvar = calculate_advanced_metrics(rets, pesos, rf_d)
                    
                    st.metric("Retorno Esperado", f"{r:.2%}", help="Retorno médio anualizado.")
                    st.metric("Volatilidade Anual", f"{v:.2%}", help="Oscilação total da carteira.")
                    st.metric("Índice de Sharpe", f"{s:.2f}", help="Retorno excedente vs risco total.")
                    st.metric("Índice de Sortino", f"{sortino:.2f}", help="Retorno excedente vs risco de queda.")
                    st.metric("Max Drawdown", f"{mdd:.2%}", help="Maior queda histórica pico-a-vale.")
                    
                    st.write(f"**Risco de Cauda (95%):**")
                    st.write(f"VaR: `{var:.2%}` | CVaR: `{cvar:.2%}`")
                    st.write("**Composição:**")
                    st.table(pd.DataFrame({'Peso (%)': (pesos*100).round(2)}, index=prices.columns))

            # --- GRÁFICOS (ALTO CONTRASTE) ---
            st.header("2. Fronteira Eficiente e Performance")
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                mc_r, mc_v = [], []
                for _ in range(n_sim):
                    w = np.random.random(n_assets); w /= np.sum(w)
                    mc_r.append(np.sum(rets_a * w)); mc_v.append(np.sqrt(np.dot(w.T, np.dot(cov_a, w))))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.array(mc_v)*100, y=np.array(mc_r)*100, mode='markers', 
                                         marker=dict(color=(np.array(mc_r)-rf_rate)/np.array(mc_v), colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")), 
                                         name="Simulações"))
                fig.add_trace(go.Scatter(x=[v*100 for _, v, _ in [calculate_stats(opt_sharpe.x, rets_a, cov_a, rf_rate)]], 
                                         y=[r*100 for r, _, _ in [calculate_stats(opt_sharpe.x, rets_a, cov_a, rf_rate)]], 
                                         mode='markers', marker=dict(color='#FFD700', size=15, symbol='star', line=dict(width=2, color='white')), name="Max Sharpe"))
                fig.add_trace(go.Scatter(x=[v*100 for _, v, _ in [calculate_stats(opt_mvp.x, rets_a, cov_a, rf_rate)]], 
                                         y=[r*100 for r, _, _ in [calculate_stats(opt_mvp.x, rets_a, cov_a, rf_rate)]], 
                                         mode='markers', marker=dict(color='#00FFFF', size=12, symbol='diamond', line=dict(width=2, color='white')), name="MVP"))
                fig.update_layout(xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)", legend=dict(orientation="h", y=-0.3, xanchor="center", x=0.5), margin=dict(r=80, b=100), template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with col_g2:
                if bench_prices is not None:
                    cum_port = (1 + rets.dot(opt_sharpe.x)).cumprod() * 10000
                    cum_bench = (1 + bench_prices.pct_change().dropna()).cumprod() * 10000
                    fig_b = go.Figure()
                    # RoyalBlue e Crimson têm excelente contraste em fundos claros e escuros
                    fig_b.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Max Sharpe", line=dict(color='#4169E1', width=3)))
                    fig_b.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=bench_ticker, line=dict(color='#DC143C', width=2, dash='dot')))
                    fig_b.update_layout(title="Crescimento de $10.000 (Backtest)", xaxis_title="Data", yaxis_title="Patrimônio", legend=dict(orientation="h", y=-0.3, xanchor="center", x=0.5), template="plotly_dark")
                    st.plotly_chart(fig_b, use_container_width=True)

            st.subheader("3. Matriz de Correlação")
            st.dataframe(rets.corr().round(2), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Grilli Research")
