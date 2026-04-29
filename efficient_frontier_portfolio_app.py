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
# ⚙️ CONFIGURAÇÃO E ESTILO
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Analytics - Grilli Research", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(28, 131, 225, 0.1);
    }
    .instrucao-ticker { 
        padding: 12px; 
        border-radius: 5px;
        border-left: 5px solid #1e3a8a; 
        background-color: rgba(30, 58, 138, 0.05);
        font-size: 0.95rem; 
        color: #1e3a8a; 
        margin-bottom: 20px;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
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
    p_sharpe = (p_ret - rf) / p_vol if p_vol != 0 else 0
    return p_ret, p_vol, p_sharpe

def calculate_advanced_metrics(rets_series, weights, rf_diario):
    p_rets = rets_series.dot(weights)
    
    # Sortino Ratio (Downside Deviation)
    downside_rets = p_rets[p_rets < 0]
    down_std = np.std(downside_rets) * np.sqrt(252)
    p_ret_anual = p_rets.mean() * 252
    sortino = (p_ret_anual - (rf_diario * 252)) / down_std if down_std != 0 else 0
    
    # Max Drawdown
    cum_rets = (1 + p_rets).cumprod()
    peak = cum_rets.cummax()
    max_dd = ((cum_rets - peak) / peak).min()
    
    # VaR e CVaR (95% confiança)
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
    st.markdown("""<div class='instrucao-ticker'>💡 <b>Uso dos Tickers:</b> Brasil (ticker + <b>.SA</b>) | Internacional (ticker puro)</div>""", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("Início da análise:", date(2021, 1, 1))
    with c2: e_date = st.date_input("Fim da análise:", date.today())
    with c3: n_sim = st.slider("Simulações Monte Carlo:", 1000, 20000, 5000)

with col_side:
    rf_rate = st.number_input("Taxa Livre de Risco (Anual %):", 0.0, 0.20, 0.1075, step=0.0025, format="%.4f")
    allow_short = st.checkbox("Venda a Descoberto (Short)", value=False)

st.subheader("Restrições de Peso")
c_min, c_max = st.columns(2)
with c_min: min_w = st.number_input("Peso Mínimo:", -1.0 if allow_short else 0.0, 1.0, 0.0)
with c_max: max_w = st.number_input("Peso Máximo:", 0.0, 2.0, 1.0)

# ==============================================================================
# 🚀 PROCESSAMENTO E DASHBOARD CONSOLIDADO
# ==============================================================================

if st.button("Executar Análise de Portfólio", use_container_width=True):
    t_list = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    
    with st.spinner("Sincronizando dados e rodando otimizadores..."):
        prices, bench_prices, bench_ticker = get_consolidated_data(t_list, s_date.isoformat(), e_date.isoformat())
        
        if prices is not None:
            rets = prices.pct_change().dropna()
            rets_a = rets.mean() * 252
            cov_a = rets.cov() * 252
            n_assets = len(prices.columns)
            rf_d = rf_rate / 252

            # Otimizadores
            bnds = tuple((min_w, max_w) for _ in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init = n_assets * [1./n_assets]
            
            opt_sharpe = sco.minimize(lambda w: -calculate_stats(w, rets_a, cov_a, rf_rate)[2], init, method='SLSQP', bounds=bnds, constraints=cons)
            opt_mvp = sco.minimize(lambda w: calculate_stats(w, rets_a, cov_a, rf_rate)[1], init, method='SLSQP', bounds=bnds, constraints=cons)
            opt_rp = sco.minimize(risk_parity_objective, init, args=(cov_a,), method='SLSQP', bounds=bnds, constraints=cons)

            # --- SEÇÃO CONSOLIDADA DE RESULTADOS ---
            st.header("1. Comparativo de Estratégias")
            
            estrategias = [
                ("🎯 Máximo Sharpe", opt_sharpe.x, "Busca o melhor retorno por unidade de risco total."),
                ("🛡️ Mínima Variância (MVP)", opt_mvp.x, "Busca a carteira com o menor risco absoluto."),
                ("⚖️ Paridade de Risco", opt_rp.x, "Equaliza a contribuição de risco entre os ativos.")
            ]
            
            # Exibição Lado a Lado
            cols = st.columns(3)
            for i, (nome, pesos, desc) in enumerate(estrategias):
                with cols[i]:
                    st.subheader(nome)
                    st.caption(desc)
                    r, v, s = calculate_stats(pesos, rets_a, cov_a, rf_rate)
                    sortino, mdd, var, cvar = calculate_advanced_metrics(rets, pesos, rf_d)
                    
                    st.metric("Retorno Esperado", f"{r:.2%}", help="Retorno médio anualizado baseado no histórico.")
                    st.metric("Volatilidade Anual", f"{v:.2%}", help="Indica a oscilação total da carteira.")
                    st.metric("Índice de Sharpe", f"{s:.2f}", help="Retorno excedente em relação ao risco total. Quanto maior, melhor.")
                    st.metric("Índice de Sortino", f"{sortino:.2f}", help="Retorno excedente penalizando apenas a volatilidade negativa (risco real de perda).")
                    st.metric("Max Drawdown", f"{mdd:.2%}", help="Maior queda histórica pico-a-vale registrada no período.")
                    
                    st.write(f"**Risco de Cauda (95%):**")
                    st.write(f"VaR: `{var:.2%}` | CVaR: `{cvar:.2%}`")
                    st.markdown(f"*(VaR: perda máx. provável | CVaR: perda média no pior cenário)*", help="Value at Risk e Conditional Value at Risk.")
                    
                    st.write("**Composição:**")
                    st.table(pd.DataFrame({'Peso (%)': (pesos*100).round(2)}, index=prices.columns))

            # --- GRÁFICOS ---
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
                                         name="Simulações", hovertemplate='Risco: %{x:.2f}%<br>Retorno: %{y:.2f}%'))
                fig.add_trace(go.Scatter(x=[v*100 for _, v, _ in [calculate_stats(opt_sharpe.x, rets_a, cov_a, rf_rate)]], 
                                         y=[r*100 for r, _, _ in [calculate_stats(opt_sharpe.x, rets_a, cov_a, rf_rate)]], 
                                         mode='markers', marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='white')), name="Max Sharpe"))
                fig.add_trace(go.Scatter(x=[v*100 for _, v, _ in [calculate_stats(opt_mvp.x, rets_a, cov_a, rf_rate)]], 
                                         y=[r*100 for r, _, _ in [calculate_stats(opt_mvp.x, rets_a, cov_a, rf_rate)]], 
                                         mode='markers', marker=dict(color='blue', size=12, symbol='diamond', line=dict(width=2, color='white')), name="MVP"))
                fig.update_layout(xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)", legend=dict(orientation="h", y=-0.35, xanchor="center", x=0.5), margin=dict(r=80), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with col_g2:
                if bench_prices is not None:
                    cum_port = (1 + rets.dot(opt_sharpe.x)).cumprod() * 10000
                    cum_bench = (1 + bench_prices.pct_change().dropna()).cumprod() * 10000
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Max Sharpe", line=dict(color='black', width=3)))
                    fig_b.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=bench_ticker, line=dict(color='gray', dash='dash')))
                    fig_b.update_layout(title="Backtest: Crescimento de $10.000", xaxis_title="Data", yaxis_title="Patrimônio ($)", legend=dict(orientation="h", y=-0.35, xanchor="center", x=0.5), template="plotly_white")
                    st.plotly_chart(fig_b, use_container_width=True)

            st.subheader("3. Matriz de Correlação")
            st.dataframe(rets.corr().round(2), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Grilli Research")
