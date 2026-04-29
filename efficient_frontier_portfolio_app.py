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
    .nota-metrica { font-size: 0.8rem; color: #666; margin-top: -10px; margin-bottom: 15px; }
    .instrucao-ticker { font-size: 0.9rem; color: #1e3a8a; font-weight: bold; margin-top: 5px; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 5px; border: 1px solid #eee; }
    h1, h2, h3 { color: #000000; font-family: 'Segoe UI', sans-serif; }
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
# 🎛️ INTERFACE DE ENTRADA
# ==============================================================================

col_main, col_side = st.columns([2, 1])
with col_main:
    tickers_in = st.text_input(
        "Ativos para análise (separados por vírgula):", 
        "VALE3.SA, ITUB4.SA, AAPL, MSFT",
        help="Insira os tickers como no Yahoo Finance."
    )
    st.markdown("<p class='instrucao-ticker'>💡 Brasil: use o ticker + .SA (ex: PETR4.SA) | EUA: use o ticker puro (ex: AAPL)</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: s_date = st.date_input("Início do período:", date(2021, 1, 1))
    with c2: e_date = st.date_input("Fim do período:", date.today())
    with c3: n_sim = st.slider("Simulações de Portfólios:", 1000, 20000, 5000)

with col_side:
    rf_rate = st.number_input("Taxa Selic/Livre de Risco (Anual %):", 0.0, 0.20, 0.1075, step=0.0025, format="%.4f")
    allow_short = st.checkbox("Venda a Descoberto (Short)", value=False, help="Permite que os pesos dos ativos sejam negativos.")

st.subheader("Configurações de Alocação")
c_min, c_max = st.columns(2)
with c_min: min_w = st.number_input("Peso Mínimo por Ativo:", -1.0 if allow_short else 0.0, 1.0, 0.0)
with c_max: max_w = st.number_input("Peso Máximo por Ativo:", 0.0, 2.0, 1.0)

# ==============================================================================
# 🚀 PROCESSAMENTO E DASHBOARD
# ==============================================================================

if st.button("Gerar Relatório de Otimização", use_container_width=True):
    t_list = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    
    with st.spinner("Sincronizando dados com mercado financeiro..."):
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

            st.header("Análise de Resultados")
            tabs = st.tabs(["🎯 Máximo Sharpe", "🛡️ Mínima Variância (MVP)", "⚖️ Paridade de Risco"])
            
            for tab, opt_res, desc in zip(tabs, [opt_sharpe, opt_mvp, opt_rp], 
                                           ["Busca o ponto de maior retorno ajustado pelo risco total.", 
                                            "Busca a carteira com o menor nível de oscilação possível.", 
                                            "Iguala a contribuição de risco de cada ativo na carteira."]):
                with tab:
                    w = opt_res.x
                    r, v, s = calculate_stats(w, rets_a, cov_a, rf_rate)
                    sortino, mdd, var, cvar = calculate_advanced_metrics(rets, w, rf_d)
                    
                    st.write(f"**Estratégia:** {desc}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Retorno Esperado", f"{r:.2%}", help="Retorno médio anualizado baseado no histórico.")
                    c2.metric("Volatilidade", f"{v:.2%}", help="O desvio padrão anualizado. Indica a intensidade da variação de preços.")
                    c3.metric("Sortino Ratio", f"{sortino:.2f}", help="Indica o retorno excedente por unidade de risco de queda.")
                    c4.metric("Max Drawdown", f"{mdd:.2%}", help="Maior queda histórica sofrida pelo portfólio no período.")
                    
                    st.write(f"**VaR Anualizado (95%):** {var:.2%} | **CVaR:** {cvar:.2%}")
                    st.markdown(f"<p class='nota-metrica'>Interpretação: Com 95% de confiança, a perda anual não deve exceder {var:.2%}. O CVaR indica a perda média se o VaR for ultrapassado.</p>", unsafe_allow_html=True)
                    st.table(pd.DataFrame({'Peso (%)': (w*100).round(2)}, index=prices.columns))

            # --- FRONTEIRA EFICIENTE ---
            st.header("Fronteira Eficiente")
            st.markdown("""
            O gráfico abaixo representa a **Fronteira Eficiente**, um conceito da Teoria de Markowitz que mostra o conjunto de portfólios que oferecem o 
            **maior retorno esperado para cada nível de risco**. Cada ponto colorido é uma combinação aleatória de ativos.
            """)
            
            mc_r, mc_v = [], []
            for _ in range(n_sim):
                w = np.random.random(n_assets); w /= np.sum(w)
                mc_r.append(np.sum(rets_a * w)); mc_v.append(np.sqrt(np.dot(w.T, np.dot(cov_a, w))))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.array(mc_v)*100, y=np.array(mc_r)*100, mode='markers', 
                                     marker=dict(color=(np.array(mc_r)-rf_rate)/np.array(mc_v), colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")), name="Portfólios Simulados"))
            fig.add_trace(go.Scatter(x=[calculate_stats(opt_sharpe.x, rets_a, cov_a, rf_rate)[1]*100], y=[calculate_stats(opt_sharpe.x, rets_a, cov_a, rf_rate)[0]*100], mode='markers', marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='white')), name="Max Sharpe"))
            fig.add_trace(go.Scatter(x=[calculate_stats(opt_mvp.x, rets_a, cov_a, rf_rate)[1]*100], y=[calculate_stats(opt_mvp.x, rets_a, cov_a, rf_rate)[0]*100], mode='markers', marker=dict(color='blue', size=12, symbol='diamond', line=dict(width=2, color='white')), name="MVP"))
            
            fig.update_layout(xaxis_title="Risco (Volatilidade %)", yaxis_title="Retorno Anual Esperado (%)", 
                              legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5), 
                              margin=dict(r=100), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # --- BACKTESTING ---
            if bench_prices is not None:
                st.header(f"Backtesting Acumulado vs {bench_ticker}")
                st.markdown(f"""
                O **Backtest** é uma simulação histórica que testa como o portfólio de **Máximo Sharpe** teria se comportado no passado comparado ao benchmark do mercado. 
                Aqui, mostramos o crescimento hipotético de **$10.000** investidos no início do período selecionado.
                """)
                
                cum_port = (1 + rets.dot(opt_sharpe.x)).cumprod() * 10000
                cum_bench = (1 + bench_prices.pct_change().dropna()).cumprod() * 10000
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Meu Portfólio (Max Sharpe)", line=dict(color='black', width=3)))
                fig_b.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=f"Referência ({bench_ticker})", line=dict(color='gray', dash='dash')))
                fig_b.update_layout(xaxis_title="Período", yaxis_title="Capital Acumulado ($)", legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5), template="plotly_white")
                st.plotly_chart(fig_b, use_container_width=True)

            st.subheader("Matriz de Correlação")
            st.markdown("A correlação indica como os ativos se movem uns em relação aos outros. Valores perto de 1 indicam movimentos iguais; perto de 0 ou negativos indicam diversificação eficiente.")
            st.dataframe(rets.corr().round(2))

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Grilli Research")
