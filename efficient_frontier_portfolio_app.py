# streamlit_app.py

import streamlit as st          
from yahooquery import Ticker   
import pandas as pd             
import numpy as np              
import plotly.graph_objects as go 
from datetime import date       
import io                       
import scipy.optimize as sco    

# ==============================================================================
# ⚙️ CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Optimization Pro",      
    layout="wide",                             
    initial_sidebar_state="collapsed",         
)

st.title("📊 Simulador de Portfólio Markowitz Pro") 
st.write("⚠️ Ferramenta educativa para análise de fronteira eficiente e métricas de risco.")

# ==============================================================================
# 🔢 FUNÇÕES DE CÁLCULO
# ==============================================================================

def calculate_portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate):
    p_ret = np.sum(expected_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol != 0 else 0
    return p_ret, p_vol, p_sharpe

def calculate_var_cvar(returns_series, weights, confidence_level=0.95):
    portfolio_returns = returns_series.dot(weights)
    sorted_returns = np.sort(portfolio_returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var_diario = sorted_returns[var_index]
    cvar_diario = sorted_returns[:var_index].mean()
    return -var_diario * np.sqrt(252), -cvar_diario * np.sqrt(252)

def neg_sharpe(weights, returns, cov, rf):
    return -calculate_portfolio_performance(weights, returns, cov, rf)[2]

def get_vol(weights, returns, cov, rf):
    return calculate_portfolio_performance(weights, returns, cov, rf)[1]

# ==============================================================================
# 📥 BUSCA DE DADOS
# ==============================================================================

@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    try:
        t_obj = Ticker(tickers)
        data = t_obj.history(start=start, end=end, interval="1d")
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or isinstance(data, str):
            return None, None
        
        df = data.reset_index()
        if 'symbol' in df.columns:
            prices = df.pivot(index='date', columns='symbol', values='adjclose')
        else:
            prices = df.set_index('date')[['adjclose']]
            prices.columns = [tickers[0]] if isinstance(tickers, list) else [tickers]
            
        prices = prices.ffill().dropna()
        return prices, prices.columns.tolist()
    except:
        return None, None

# ==============================================================================
# 🎛️ INTERFACE DE CONFIGURAÇÃO (LAYOUT ORIGINAL)
# ==============================================================================

st.subheader("Configuração da Análise")

with st.expander("❓ Guia de Tickers (Brasil vs Mundo)", expanded=False):
    st.markdown("""
    * **🇧🇷 Brasil:** Use sufixo **.SA** (ex: `PETR4.SA`, `VALE3.SA`).
    * **🇺🇸 Internacional:** Use o Ticker puro (ex: `AAPL`, `MSFT`, `BTC-USD`).
    """)

col_tickers, col_rf = st.columns([2, 1])
with col_tickers:
    tickers_input = st.text_input("Digite os Tickers separados por vírgula:", "PETR4.SA, VALE3.SA, AAPL, MSFT")
with col_rf:
    rf_rate = st.number_input("Taxa Livre de Risco (Decimal):", 0.0, 0.3, 0.1075, step=0.0025)

col_d1, col_d2, col_mc = st.columns(3)
with col_d1:
    start_date = st.date_input("Data Inicial:", date(2020, 1, 1))
with col_d2:
    end_date = st.date_input("Data Final:", date.today())
with col_mc:
    num_sim = st.slider("Simulações Monte Carlo:", 1000, 20000, 10000)

st.subheader("Restrições e Short Selling")
col_short, col_min, col_max = st.columns([1, 1, 1])
with col_short:
    allow_short = st.checkbox("Permitir Venda a Descoberto (Short Selling)", value=False)
with col_min:
    min_w = st.number_input("Peso Mínimo por Ativo:", -2.0 if allow_short else 0.0, 1.0, -0.5 if allow_short else 0.0)
with col_max:
    max_w = st.number_input("Peso Máximo por Ativo:", 0.0, 2.0 if allow_short else 1.0, 1.0)

if st.button("🚀 Rodar Otimização Completa", use_container_width=True):
    ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    with st.spinner("Processando..."):
        prices, valid_tickers = get_data(ticker_list, start_date.isoformat(), end_date.isoformat())
        
        if prices is not None:
            returns = prices.pct_change().dropna()
            exp_ret = returns.mean() * 252
            cov_mat = returns.cov() * 252
            num_assets = len(valid_tickers)

            # --- Monte Carlo ---
            mc_results = np.zeros((3, num_sim))
            for i in range(num_sim):
                # Se short for permitido, os pesos podem ser negativos
                if allow_short:
                    w = np.random.uniform(min_w, max_w, num_assets)
                else:
                    w = np.random.random(num_assets)
                w /= np.sum(w)
                ret, vol, shrp = calculate_portfolio_performance(w, exp_ret, cov_mat, rf_rate)
                mc_results[0,i] = ret
                mc_results[1,i] = vol
                mc_results[2,i] = shrp

            # --- SciPy ---
            bounds = tuple((min_w, max_w) for _ in range(num_assets))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init_guess = num_assets * [1./num_assets]
            
            opt_sharpe = sco.minimize(neg_sharpe, init_guess, args=(exp_ret, cov_mat, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
            opt_vol = sco.minimize(get_vol, init_guess, args=(exp_ret, cov_mat, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)

            # --- Resultados ---
            st.write("---")
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🎯 Máximo Sharpe")
                w_s = opt_sharpe.x
                r_s, v_s, s_s = calculate_portfolio_performance(w_s, exp_ret, cov_mat, rf_rate)
                va_s, cv_s = calculate_var_cvar(returns, w_s)
                st.write(f"Retorno: {r_s:.2%} | Risco: {v_s:.2%}")
                st.write(f"VaR (95%): {va_s:.2%} | CVaR: {cv_s:.2%}")
                st.table(pd.DataFrame({'Ativo': valid_tickers, 'Peso (%)': (w_s*100).round(2)}).set_index('Ativo'))

            with c2:
                st.subheader("🛡️ Mínima Variância")
                w_v = opt_vol.x
                r_v, v_v, s_v = calculate_portfolio_performance(w_v, exp_ret, cov_mat, rf_rate)
                va_v, cv_v = calculate_var_cvar(returns, w_v)
                st.write(f"Retorno: {r_v:.2%} | Risco: {v_v:.2%}")
                st.write(f"VaR (95%): {va_v:.2%} | CVaR: {cv_v:.2%}")
                st.table(pd.DataFrame({'Ativo': valid_tickers, 'Peso (%)': (w_v*100).round(2)}).set_index('Ativo'))

            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mc_results[1,:]*100, y=mc_results[0,:]*100, mode='markers', 
                                     marker=dict(color=mc_results[2,:], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe"))))
            fig.add_trace(go.Scatter(x=[v_s*100], y=[r_s*100], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
            fig.update_layout(xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Correlação (Corrigido: sem dependência de matplotlib)
            st.subheader("Matriz de Correlação")
            st.dataframe(returns.corr().round(2))

        else:
            st.error("Erro ao buscar dados.")

st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Grilli Research")
