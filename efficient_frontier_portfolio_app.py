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
    .nota-explicativa { font-size: 0.88rem; color: #555; font-style: italic; margin-bottom: 20px; line-height: 1.4; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 5px; border: 1px solid #eee; }
    h1, h2, h3 { color: #000000; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
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
    downside_rets = p_rets[p_rets < 0]
    down_std = np.std(downside_rets) * np.sqrt(252)
    p_ret_anual = p_rets.mean() * 252
    sortino = (p_ret_anual - (rf_diario * 252)) / down_std if down_std != 0 else 0
    cum_rets = (1 + p_rets).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
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
# 📥 DADOS E BENCHMARK
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
        if bench not in prices.columns:
            return prices, None, None
        bench_data = prices[bench]
        assets_data = prices.drop(columns=[bench])
        return assets_data, bench_data, bench
    except:
        return None, None, None

# ==============================================================================
# 🎛️ WIDGETS
# ==============================================================================

with st.expander("Metodologia e Indicadores", expanded=False):
    st.markdown("""
    * **Índice de Sharpe:** Retorno excedente por unidade de risco total.
    * **MVP (Minimum Variance Portfolio):** O portfólio com a menor volatilidade possível na fronteira eficiente.
    * **Risk Parity:** Alocação onde cada ativo contribui igualmente para o risco total.
    * **VaR (95%):** Perda máxima anualizada esperada (95% de confiança).
    * **Max Drawdown:** Queda máxima histórica do pico ao vale.
    """)

col_main, col_side = st.columns([2, 1])
with col_main:
    tickers_in = st.text_input("Ativos (separados por vírgula):", "VALE3.SA, ITUB4.SA, AAPL, MSFT")
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("Data inicial:", date(2021, 1, 1))
    with c2: e_date = st.date_input("Data final:", date.today())
    with c3: n_sim = st.slider("Simulações Monte Carlo:", 1000, 20000, 5000)

with col_side:
    rf_rate = st.number_input("Taxa livre de risco (Anual decimal):", 0.0, 0.20, 0.1075, step=0.0025, format="%.4f")
    allow_short = st.checkbox("Permitir Venda a Descoberto", value=False)

st.subheader("Configurações de Peso")
c_min, c_max = st.columns(2)
with c_min: min_w = st.number_input("Peso Mínimo:", -1.0 if allow_short else 0.0, 1.0, 0.0)
with c_max: max_w = st.number_input("Peso Máximo:", 0.0, 2.0, 1.0)

# ==============================================================================
# 🚀 PROCESSAMENTO
# ==============================================================================

if st.button("Executar Análise de Portfólio", use_container_width=True):
    t_list = [t.strip().upper() for t in t_input = tickers_in.split(",") if t.strip()]
    
    with st.spinner("Processando..."):
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
            
            # Otimizadores
            opt_sharpe = sco.minimize(lambda w: -calculate_stats(w, rets_a, cov_a, rf_rate)[2], init, method='SLSQP', bounds=bnds, constraints=cons)
            opt_mvp = sco.minimize(lambda w: calculate_stats(w, rets_a, cov_a, rf_rate)[1], init, method='SLSQP', bounds=bnds, constraints=cons)
            opt_rp = sco.minimize(risk_parity_objective, init, args=(cov_a,), method='SLSQP', bounds=bnds, constraints=cons)

            # --- RESULTADOS ---
            st.header("Análise de Alocação e Risco")
            tab_s, tab_v, tab_r = st.tabs(["Máximo Sharpe", "Mínima Variância (MVP)", "Paridade de Risco"])
            
            with tab_s:
                ws = opt_sharpe.x
                rs, vs, ss = calculate_stats(ws, rets_a, cov_a, rf_rate)
                sort_s, dd_s, var_s, cvs = calculate_advanced_metrics(rets, ws, rf_d)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno", f"{rs:.2%}"); c2.metric("Volatilidade", f"{vs:.2%}"); c3.metric("Sortino", f"{sort_s:.2f}"); c4.metric("Max DD", f"{dd_s:.2%}")
                st.write(f"**VaR (95%):** {var_s:.2%} | **CVaR:** {cvs:.2%}")
                st.table(pd.DataFrame({'Peso': ws}, index=prices.columns).style.format("{:.2%}"))

            with tab_v:
                wv = opt_mvp.x
                rv, vv, sv = calculate_stats(wv, rets_a, cov_a, rf_rate)
                sort_v, dd_v, var_v, cvv = calculate_advanced_metrics(rets, wv, rf_d)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno", f"{rv:.2%}"); c2.metric("Volatilidade", f"{vv:.2%}"); c3.metric("Sortino", f"{sort_v:.2f}"); c4.metric("Max DD", f"{dd_v:.2%}")
                st.write(f"**VaR (95%):** {var_v:.2%} | **CVaR:** {cvv:.2%}")
                st.table(pd.DataFrame({'Peso': wv}, index=prices.columns).style.format("{:.2%}"))

            with tab_r:
                wr = opt_rp.x
                rr, vr, sr = calculate_stats(wr, rets_a, cov_a, rf_rate)
                sort_r, dd_r, var_r, cvr = calculate_advanced_metrics(rets, wr, rf_d)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno", f"{rr:.2%}"); c2.metric("Volatilidade", f"{vr:.2%}"); c3.metric("Sortino", f"{sort_r:.2f}"); c4.metric("Max DD", f"{dd_r:.2%}")
                st.write(f"**VaR (95%):** {var_r:.2%} | **CVaR:** {cvr:.2%}")
                st.table(pd.DataFrame({'Peso': wr}, index=prices.columns).style.format("{:.2%}"))

            # --- VISUALIZAÇÕES ---
            st.header("Fronteira Eficiente")
            mc_r, mc_v = [], []
            for _ in range(n_sim):
                w = np.random.random(n_assets); w /= np.sum(w)
                mc_r.append(np.sum(rets_a * w)); mc_v.append(np.sqrt(np.dot(w.T, np.dot(cov_a, w))))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.array(mc_v)*100, y=np.array(mc_r)*100, mode='markers', marker=dict(color=(np.array(mc_r)-rf_rate)/np.array(mc_v), colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")), name="Simulações"))
            fig.add_trace(go.Scatter(x=[vs*100], y=[rs*100], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
            fig.add_trace(go.Scatter(x=[vv*100], y=[rv*100], mode='markers', marker=dict(color='blue', size=12, symbol='diamond'), name="MVP"))
            fig.update_layout(xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)", legend=dict(orientation="h", y=-0.4, x=0.5, xanchor="center"), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            if bench_prices is not None:
                st.header(f"Backtesting vs {bench_ticker}")
                cum_port = (1 + rets.dot(ws)).cumprod() * 10000
                cum_bench = (1 + bench_prices.pct_change().dropna()).cumprod() * 10000
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Max Sharpe", line=dict(color='black', width=2)))
                fig_b.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=bench_ticker, line=dict(color='gray', dash='dash')))
                st.plotly_chart(fig_b, use_container_width=True)

            st.subheader("Matriz de Correlação")
            st.dataframe(rets.corr().round(2))

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Grilli Research")
