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
# ⚙️ CONFIGURAÇÃO E ESTILO (CONTRASTE PARA MODO ESCURO/CLARO)
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Analytics Pro - Grilli Research", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(100, 100, 100, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(150, 150, 150, 0.2);
    }
    .instrucao-ticker { 
        padding: 12px; 
        border-radius: 5px;
        border-left: 5px solid #ffcc00; 
        background-color: rgba(255, 204, 0, 0.1);
        font-size: 0.95rem; 
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Simulador de Portfólio: Markowitz, Risk Parity & Atribuição de Risco")
st.write("---")

# ==============================================================================
# 🔢 FUNÇÕES ANALÍTICAS AVANÇADAS
# ==============================================================================

def calculate_stats(weights, rets_anual, cov_mat, rf):
    p_ret = np.sum(rets_anual * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0
    return p_ret, p_vol, p_sharpe

def calculate_risk_contribution(weights, cov_mat):
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    marginal_risk = np.dot(cov_mat, weights) / p_vol
    risk_contribution = weights * marginal_risk
    return risk_contribution / p_vol 

def calculate_advanced_metrics(rets_series, weights, rf_diario, bench_rets):
    p_rets = rets_series.dot(weights)
    
    # Sortino Ratio
    downside_rets = p_rets[p_rets < 0]
    down_std = np.std(downside_rets) * np.sqrt(252)
    sortino = (p_rets.mean() * 252 - (rf_diario * 252)) / down_std if down_std > 0 else 0
    
    # Beta e Índice de Treynor
    covariance = np.cov(p_rets, bench_rets)[0, 1]
    bench_variance = np.var(bench_rets)
    beta = covariance / bench_variance if bench_variance > 0 else 1
    treynor = (p_rets.mean() * 252 - (rf_diario * 252)) / beta if beta != 0 else 0
    
    # Drawdown
    cum_rets = (1 + p_rets).cumprod()
    max_dd = ((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min()
    
    return sortino, beta, treynor, max_dd

# ==============================================================================
# 📥 GESTÃO DE DADOS
# ==============================================================================

@st.cache_data(ttl=3600)
def get_market_data(tickers, start, end):
    try:
        bench = "^BVSP" if any(".SA" in t.upper() for t in tickers) else "^GSPC"
        full_list = list(set(tickers + [bench]))
        t_obj = Ticker(full_list)
        df = t_obj.history(start=start, end=end)
        if df is None or df.empty: return None, None, None
        df = df.reset_index()
        col = 'adjclose' if 'adjclose' in df.columns else 'close'
        prices = df.pivot(index='date', columns='symbol', values=col).ffill().dropna()
        return prices.drop(columns=[bench]), prices[bench], bench
    except: return None, None, None

# ==============================================================================
# 🎛️ ENTRADA E PARÂMETROS
# ==============================================================================

col_in, col_opt = st.columns([2, 1])
with col_in:
    tickers_in = st.text_input("Ativos:", "VALE3.SA, ITUB4.SA, PETR4.SA, AAPL, MSFT, BTC-USD")
    st.markdown("<div class='instrucao-ticker'>💡 <b>Sintaxe:</b> Brasil (.SA) | EUA (ticker puro) | Benchmark detectado automaticamente.</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("Início:", date(2021, 1, 1))
    with c2: e_date = st.date_input("Fim:", date.today())
    with c3: n_sim = st.slider("Monte Carlo:", 1000, 10000, 5000)

with col_opt:
    rf_rate = st.number_input("Taxa Selic/RF (Anual %):", 0.0, 0.2, 0.1075, step=0.0025, format="%.4f")
    allow_short = st.checkbox("Venda a Descoberto", value=False)
    min_w = st.number_input("Peso Mín:", -1.0 if allow_short else 0.0, 1.0, 0.0)
    max_w = st.number_input("Peso Máx:", 0.0, 2.0, 1.0)

# ==============================================================================
# 🚀 EXECUÇÃO E DASHBOARD
# ==============================================================================

if st.button("Executar Relatório de Atribuição de Risco", use_container_width=True):
    t_list = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    
    with st.spinner("Sincronizando dados..."):
        prices, bench_prices, bench_name = get_market_data(t_list, s_date.isoformat(), e_date.isoformat())
        
        if prices is not None:
            rets = prices.pct_change().dropna()
            bench_rets_series = bench_prices.pct_change().dropna()
            
            # Alinhamento de datas
            common_idx = rets.index.intersection(bench_rets_series.index)
            rets = rets.loc[common_idx]
            bench_rets_series = bench_rets_series.loc[common_idx]
            
            rets_a, cov_a = rets.mean() * 252, rets.cov() * 252
            n_assets = len(prices.columns)
            
            # Otimizadores
            bnds = tuple((min_w, max_w) for _ in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init = n_assets * [1./n_assets]
            
            opt_s = sco.minimize(lambda w: -calculate_stats(w, rets_a, cov_a, rf_rate)[2], init, method='SLSQP', bounds=bnds, constraints=cons)
            opt_v = sco.minimize(lambda w: calculate_stats(w, rets_a, cov_a, rf_rate)[1], init, method='SLSQP', bounds=bnds, constraints=cons)

            # --- RESULTADOS ---
            st.header("1. Performance e Métricas de Treynor/Sortino")
            
            tabs = st.tabs(["🎯 Máximo Sharpe", "🛡️ Mínima Variância (MVP)"])
            for tab, weights, label in zip(tabs, [opt_s.x, opt_v.x], ["Max Sharpe", "MVP"]):
                with tab:
                    r, v, s = calculate_stats(weights, rets_a, cov_a, rf_rate)
                    sortino, beta, treynor, mdd = calculate_advanced_metrics(rets, weights, rf_rate/252, bench_rets_series)
                    risk_contrib = calculate_risk_contribution(weights, cov_a)
                    
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Retorno", f"{r:.2%}")
                    c2.metric("Sharpe", f"{s:.2f}")
                    c3.metric("Sortino", f"{sortino:.2f}", help="Retorno/Risco de Queda")
                    c4.metric("Beta", f"{beta:.2f}", help=f"Sensibilidade ao {bench_name}")
                    c5.metric("Treynor", f"{treynor:.2f}", help="Retorno excedente por unidade de Beta")

                    st.subheader("Atribuição de Risco vs Alocação de Capital")
                    df_attrib = pd.DataFrame({
                        'Peso Capital (%)': (weights * 100).round(2),
                        'Contribuição Risco (%)': (risk_contrib * 100).round(2)
                    }, index=prices.columns)
                    
                    col_t, col_p = st.columns([1, 1])
                    with col_t:
                        st.table(df_attrib)
                    with col_p:
                        fig_attr = go.Figure(data=[
                            go.Bar(name='Peso Capital', x=prices.columns, y=weights, marker_color='#4169E1'),
                            go.Bar(name='Contrib. Risco', x=prices.columns, y=risk_contribution, marker_color='#FF4500')
                        ])
                        fig_attr.update_layout(barmode='group', height=350, template="plotly_dark", margin=dict(t=20, b=20))
                        st.plotly_chart(fig_attr, use_container_width=True)

            # --- GRÁFICOS ---
            st.header("2. Fronteira Eficiente e Backtest Acumulado")
            cg1, cg2 = st.columns(2)
            
            with cg1:
                mc_v, mc_r = [], []
                for _ in range(n_sim):
                    w = np.random.random(n_assets); w /= np.sum(w)
                    ret_sim, vol_sim, _ = calculate_stats(w, rets_a, cov_a, rf_rate)
                    mc_v.append(vol_sim); mc_r.append(ret_sim)
                
                fig_fe = go.Figure()
                fig_fe.add_trace(go.Scatter(x=np.array(mc_v)*100, y=np.array(mc_r)*100, mode='markers', marker=dict(color=(np.array(mc_r)-rf_rate)/np.array(mc_v), colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe"))))
                fig_fe.add_trace(go.Scatter(x=[calculate_stats(opt_s.x, rets_a, cov_a, rf_rate)[1]*100], y=[calculate_stats(opt_s.x, rets_a, cov_a, rf_rate)[0]*100], mode='markers', marker=dict(color='#FFD700', size=15, symbol='star', line=dict(color='white', width=2)), name="Max Sharpe"))
                fig_fe.update_layout(xaxis_title="Risco (%)", yaxis_title="Retorno (%)", template="plotly_dark", legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_fe, use_container_width=True)

            with cg2:
                cum_port = (1 + rets.dot(opt_s.x)).cumprod() * 10000
                cum_bench = (1 + bench_rets_series).cumprod() * 10000
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfólio (Max Sharpe)", line=dict(color='#00FFFF', width=3)))
                fig_bt.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=f"Benchmark ({bench_name})", line=dict(color='#FF4500', dash='dot', width=2)))
                fig_bt.update_layout(title="Crescimento de Capital ($10k)", template="plotly_dark", legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_bt, use_container_width=True)

            st.subheader("3. Matriz de Correlação")
            st.dataframe(rets.corr().round(2), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Grilli Research")
