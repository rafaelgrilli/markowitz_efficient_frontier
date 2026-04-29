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
# ⚙️ CONFIGURAÇÃO
# ==============================================================================
st.set_page_config(page_title="Portfolio Optimization Pro", layout="wide", initial_sidebar_state="collapsed")

st.title("📊 Simulador de Portfólio Markowitz Pro") 
st.write("⚠️ Ferramenta de nível profissional para análise fundamentalista e quantitativa.")

# ==============================================================================
# 🔢 FUNÇÕES TÉCNICAS
# ==============================================================================

def calculate_stats(weights, rets_anual, cov_mat, rf):
    p_ret = np.sum(rets_anual * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    p_sharpe = (p_ret - rf) / p_vol if p_vol != 0 else 0
    return p_ret, p_vol, p_sharpe

def calculate_var_cvar(rets_series, weights, conf=0.95):
    p_rets = rets_series.dot(weights)
    sorted_rets = np.sort(p_rets)
    idx = int((1 - conf) * len(sorted_rets))
    var_d = sorted_rets[idx]
    cvar_d = sorted_rets[:idx].mean()
    return -var_d * np.sqrt(252), -cvar_d * np.sqrt(252)

# Funções SciPy
def min_func_sharpe(weights, rets_anual, cov_mat, rf):
    return -calculate_stats(weights, rets_anual, cov_mat, rf)[2]

def min_func_vol(weights, rets_anual, cov_mat, rf):
    return calculate_stats(weights, rets_anual, cov_mat, rf)[1]

# ==============================================================================
# 📥 BUSCA DE DADOS (ROBUSTA)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_clean_data(tickers, start, end):
    try:
        t_obj = Ticker(tickers)
        data = t_obj.history(start=start, end=end, interval="1d")
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or isinstance(data, str):
            return None, None
        df = data.reset_index()
        col = 'adjclose' if 'adjclose' in df.columns else 'close'
        prices = df.pivot(index='date', columns='symbol', values=col).ffill().dropna()
        return prices, prices.columns.tolist()
    except:
        return None, None

# ==============================================================================
# 🎛️ UI - CONFIGURAÇÃO (ESTILO ORIGINAL)
# ==============================================================================

st.subheader("Configuração da Estratégia")

with st.expander("❓ Guia de Preenchimento (Brasil vs Exterior)", expanded=False):
    st.markdown("""
    * **Mercado Brasileiro (B3):** Use `.SA` (Ex: `PETR4.SA`, `VALE3.SA`).
    * **Mercado Americano:** Use o Ticker puro (Ex: `AAPL`, `MSFT`, `TSLA`).
    * **Crypto:** Use o par com USD (Ex: `BTC-USD`, `ETH-USD`).
    """)

col_t, col_rf = st.columns([2, 1])
with col_t:
    t_input = st.text_input("Ativos separados por vírgula:", "PETR4.SA, VALE3.SA, AAPL, MSFT, BTC-USD")
with col_rf:
    rf_input = st.number_input("Taxa Livre de Risco (Anual Decimal):", 0.0, 0.3, 0.1075, step=0.0025, format="%.4f")

c_d1, c_d2, c_mc = st.columns(3)
with c_d1:
    s_date = st.date_input("Início:", date(2020, 1, 1))
with c_d2:
    e_date = st.date_input("Fim:", date.today())
with c_mc:
    n_sim = st.slider("Simulações Monte Carlo:", 1000, 50000, 10000)

st.subheader("Restrições de Alocação e Short Selling")
col_sh, col_mi, col_ma = st.columns(3)
with col_sh:
    allow_short = st.checkbox("Permitir Venda a Descoberto (Short Selling)", value=False)
with col_mi:
    min_weight = st.number_input("Peso Mínimo por Ativo:", -1.0 if allow_short else 0.0, 1.0, -0.2 if allow_short else 0.0)
with col_ma:
    max_weight = st.number_input("Peso Máximo por Ativo:", 0.0, 2.0, 1.0)

# ==============================================================================
# 🚀 EXECUÇÃO
# ==============================================================================

if st.button("📈 Rodar Otimização e Gerar Dashboard", use_container_width=True):
    t_list = [t.strip().upper() for t in t_input.split(",") if t.strip()]
    
    with st.spinner("Processando dados históricos e otimizando..."):
        prices, valid_t = get_clean_data(t_list, s_date.isoformat(), e_date.isoformat())
        
        if prices is not None:
            rets = prices.pct_change().dropna()
            rets_a = rets.mean() * 252
            cov_a = rets.cov() * 252
            n_assets = len(valid_t)

            # --- Monte Carlo ---
            mc_res = np.zeros((3, n_sim))
            for i in range(n_sim):
                if allow_short:
                    w = np.random.uniform(min_weight, max_weight, n_assets)
                else:
                    w = np.random.random(n_assets)
                w /= np.sum(w)
                r, v, s = calculate_stats(w, rets_a, cov_a, rf_input)
                mc_res[0,i], mc_res[1,i], mc_res[2,i] = r, v, s

            # --- SciPy ---
            bnds = tuple((min_weight, max_weight) for _ in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init = n_assets * [1./n_assets]
            
            opt_s = sco.minimize(min_func_sharpe, init, args=(rets_a, cov_a, rf_input), method='SLSQP', bounds=bnds, constraints=cons)
            opt_v = sco.minimize(min_func_vol, init, args=(rets_a, cov_a, rf_input), method='SLSQP', bounds=bnds, constraints=cons)

            # --- Dashboard ---
            st.write("---")
            res1, res2 = st.columns(2)
            
            with res1:
                st.subheader("🎯 Portfólio Ótimo (Max Sharpe)")
                ws = opt_s.x
                rs, vs, ss = calculate_stats(ws, rets_a, cov_a, rf_input)
                vars, cvs = calculate_var_cvar(rets, ws)
                st.metric("Retorno Esperado", f"{rs:.2%}")
                st.metric("Volatilidade", f"{vs:.2%}")
                st.write(f"**VaR (95%):** {vars:.2%} | **CVaR (95%):** {cvs:.2%}")
                st.table(pd.DataFrame({'Ativo': valid_t, 'Peso (%)': (ws*100).round(2)}).set_index('Ativo'))

            with res2:
                st.subheader("🛡️ Mínima Variância")
                wv = opt_vol.x
                rv, vv, sv = calculate_stats(wv, rets_a, cov_a, rf_input)
                varv, cvv = calculate_var_cvar(rets, wv)
                st.metric("Retorno Esperado", f"{rv:.2%}")
                st.metric("Volatilidade", f"{vv:.2%}")
                st.write(f"**VaR (95%):** {varv:.2%} | **CVaR (95%):** {cvv:.2%}")
                st.table(pd.DataFrame({'Ativo': valid_t, 'Peso (%)': (wv*100).round(2)}).set_index('Ativo'))

            # Gráfico
            st.subheader("Fronteira Eficiente")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mc_res[1,:]*100, y=mc_res[0,:]*100, mode='markers', 
                                     marker=dict(color=mc_res[2,:], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe"))))
            fig.add_trace(go.Scatter(x=[vs*100], y=[rs*100], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
            fig.update_layout(xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Correlação e Alocação (Pizza)
            st.subheader("Análise de Ativos")
            ca, cb = st.columns(2)
            with ca:
                st.write("**Matriz de Correlação**")
                st.dataframe(rets.corr().round(2))
            with cb:
                st.write("**Distribuição de Pesos (Max Sharpe)**")
                fig_pie = go.Figure(data=[go.Pie(labels=valid_t, values=np.abs(ws).round(4), hole=.3)])
                st.plotly_chart(fig_pie, use_container_width=True)

        else:
            st.error("Erro ao buscar dados. Verifique os tickers e as datas.")

st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Portfolio Optimization - Grilli Research")
