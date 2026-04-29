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
# 🔢 FUNÇÕES DE CÁLCULO (MARKOWITZ & RISCO)
# ==============================================================================

def calculate_portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate):
    p_ret = np.sum(expected_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol != 0 else 0
    return p_ret, p_vol, p_sharpe

def calculate_var_cvar(returns_series, weights, confidence_level=0.95):
    """Calcula VaR e CVaR Histórico Anualizado."""
    portfolio_returns = returns_series.dot(weights)
    sorted_returns = np.sort(portfolio_returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var_diario = sorted_returns[var_index]
    cvar_diario = sorted_returns[:var_index].mean()
    # Anualização pela raiz do tempo (sqrt de 252 dias úteis)
    return -var_diario * np.sqrt(252), -cvar_diario * np.sqrt(252)

def neg_sharpe(weights, returns, cov, rf):
    return -calculate_portfolio_performance(weights, returns, cov, rf)[2]

def get_vol(weights, returns, cov, rf):
    return calculate_portfolio_performance(weights, returns, cov, rf)[1]

# ==============================================================================
# 📥 BUSCA DE DADOS (CORRIGIDA E ROBUSTA)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    try:
        data = Ticker(tickers).history(start=start, end=end, interval="1d")
        if data is None or isinstance(data, str) or data.empty:
            return None, None
        
        df = data.reset_index()
        if 'symbol' in df.columns:
            prices = df.pivot(index='date', columns='symbol', values='adjclose')
        else:
            prices = df.set_index('date')[['adjclose']]
            prices.columns = [tickers[0]]
            
        prices = prices.ffill().dropna()
        return prices, prices.columns.tolist()
    except:
        return None, None

# ==============================================================================
# 🎛️ INTERFACE DE CONFIGURAÇÃO (LAYOUT ORIGINAL)
# ==============================================================================

st.subheader("Configuração da Análise")

# Expansor com instruções claras de preenchimento
with st.expander("❓ Como preencher os ativos (Brasileiros vs Estrangeiros)", expanded=True):
    st.markdown("""
    Para que o simulador encontre os dados corretamente, siga o padrão do Yahoo Finance:
    
    * **🇧🇷 Ativos Brasileiros (B3):** Adicione obrigatoriamente o sufixo **.SA**.
        * Exemplos: `PETR4.SA`, `VALE3.SA`, `ITUB4.SA`, `IVVB11.SA`.
    * **🇺🇸 Ativos Estrangeiros (EUA):** Use apenas o Ticker original da Nasdaq ou NYSE.
        * Exemplos: `AAPL` (Apple), `MSFT` (Microsoft), `TSLA` (Tesla), `VOO` (ETF S&P 500).
    * **💡 Dica de Mistura:** Você pode simular uma carteira global inserindo ambos, separados por vírgula. 
        * Exemplo: `PETR4.SA, VALE3.SA, AAPL, MSFT, BTC-USD`.
    """)

col_tickers, col_rf = st.columns([2, 1])
with col_tickers:
    tickers_input = st.text_input(
        "Digite os Tickers separados por vírgula:", 
        "PETR4.SA, VALE3.SA, AAPL, MSFT",
        help="Lembre-se do .SA para ativos brasileiros."
    )
with col_rf:
    rf_rate = st.number_input("Taxa Livre de Risco (Anual Decimal):", 0.0, 0.3, 0.1075, step=0.0025, help="Ex: 0.1075 para 10,75% (Selic/CDI)")

col_date1, col_date2, col_mc = st.columns(3)
with col_date1:
    start_date = st.date_input("Data Inicial:", date(2020, 1, 1))
with col_date2:
    end_date = st.date_input("Data Final:", date.today())
with col_mc:
    num_sim = st.slider("Simulações Monte Carlo:", 1000, 20000, 10000)

# Botão de execução centralizado
if st.button("🚀 Rodar Otimização e Análise de Risco", use_container_width=True):
    ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    with st.spinner("Buscando dados e processando matrizes de covariância..."):
        prices, valid_tickers = get_data(ticker_list, start_date.isoformat(), end_date.isoformat())
        
        if prices is not None:
            returns = prices.pct_change().dropna()
            exp_ret = returns.mean() * 252
            cov_mat = returns.cov() * 252
            num_assets = len(valid_tickers)

            # --- Monte Carlo ---
            mc_results = np.zeros((3, num_sim))
            for i in range(num_sim):
                w = np.random.random(num_assets)
                w /= np.sum(w)
                ret, vol, shrp = calculate_portfolio_performance(w, exp_ret, cov_mat, rf_rate)
                mc_results[0,i] = ret
                mc_results[1,i] = vol
                mc_results[2,i] = shrp

            # --- Otimização SciPy ---
            bounds = tuple((0, 1) for _ in range(num_assets))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init_guess = num_assets * [1./num_assets]
            
            # Max Sharpe
            opt_sharpe = sco.minimize(neg_sharpe, init_guess, args=(exp_ret, cov_mat, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
            w_s = opt_sharpe.x
            ret_s, vol_s, shrp_s = calculate_portfolio_performance(w_s, exp_ret, cov_mat, rf_rate)
            var_s, cvar_s = calculate_var_cvar(returns, w_s)

            # Min Vol
            opt_vol = sco.minimize(get_vol, init_guess, args=(exp_ret, cov_mat, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
            w_v = opt_vol.x
            ret_v, vol_v, shrp_v = calculate_portfolio_performance(w_v, exp_ret, cov_mat, rf_rate)
            var_v, cvar_v = calculate_var_cvar(returns, w_v)

            # ==============================================================================
            # 📊 DASHBOARD DE RESULTADOS
            # ==============================================================================
            st.write("---")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("🎯 Máximo Sharpe (Tangência)")
                st.write(f"**Retorno:** {ret_s:.2%} | **Risco:** {vol_s:.2%} | **Sharpe:** {shrp_s:.2f}")
                st.write(f"**VaR (95%):** {var_s:.2%} | **CVaR (95%):** {cvar_s:.2%}")
                st.dataframe(pd.DataFrame({'Ativo': valid_tickers, 'Peso (%)': (w_s*100).round(2)}).set_index('Ativo'))

            with col_res2:
                st.subheader("🛡️ Mínima Variância")
                st.write(f"**Retorno:** {ret_v:.2%} | **Risco:** {vol_v:.2%} | **Sharpe:** {shrp_v:.2f}")
                st.write(f"**VaR (95%):** {var_v:.2%} | **CVaR (95%):** {cvar_v:.2%}")
                st.dataframe(pd.DataFrame({'Ativo': valid_tickers, 'Peso (%)': (w_v*100).round(2)}).set_index('Ativo'))

            # Gráfico da Fronteira Eficiente
            st.subheader("Gráfico da Fronteira Eficiente")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mc_results[1,:]*100, y=mc_results[0,:]*100,
                mode='markers', marker=dict(color=mc_results[2,:], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
                name="Simulações Monte Carlo"
            ))
            fig.add_trace(go.Scatter(x=[vol_s*100], y=[ret_s*100], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
            fig.update_layout(xaxis_title="Volatilidade (%)", yaxis_title="Retorno Esperado (%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Correlação
            st.subheader("Matriz de Correlação")
            st.dataframe(returns.corr().style.background_gradient(cmap='coolwarm').format("{:.2f}"))

        else:
            st.error("❌ Erro ao buscar dados. Verifique os sufixos (.SA para Brasil) ou o intervalo de datas.")

st.sidebar.markdown("© 2026 Rafael Grilli Felizardo - Grilli Research")
