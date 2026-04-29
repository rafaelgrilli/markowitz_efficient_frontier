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
# ⚙️ Configuração da Página
# ==============================================================================
st.set_page_config(
    page_title="Otimizador de Portfólio Grilli",      
    layout="wide",                             
    initial_sidebar_state="collapsed",         
)

st.title("📊 Simulador de Portfólio Markowitz Pro") 
st.write("---")

# ==============================================================================
# 🔢 Funções de Cálculo Estatístico e Risco
# ==============================================================================

def calculate_portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate):
    """Retorna o retorno, volatilidade e índice de Sharpe do portfólio."""
    p_ret = np.sum(expected_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol != 0 else 0
    return p_ret, p_vol, p_sharpe

def calculate_var_cvar(returns_series, weights, confidence_level=0.95):
    """Calcula VaR e CVaR histórico anualizado para o portfólio."""
    portfolio_returns = returns_series.dot(weights)
    sorted_returns = np.sort(portfolio_returns)
    
    # VaR Histórico
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var_diario = sorted_returns[var_index]
    
    # CVaR (Expected Shortfall)
    cvar_diario = sorted_returns[:var_index].mean()
    
    # Anualização (Escalonamento pela raiz do tempo)
    var_anual = -var_diario * np.sqrt(252)
    cvar_anual = -cvar_diario * np.sqrt(252)
    
    return var_anual, cvar_anual

# Funções para o Otimizador SciPy
def neg_sharpe(weights, returns, cov, rf):
    return -calculate_portfolio_performance(weights, returns, cov, rf)[2]

def get_vol(weights, returns, cov, rf):
    return calculate_portfolio_performance(weights, returns, cov, rf)[1]

# ==============================================================================
# 📥 Gestão de Dados
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
# 🎛️ Interface e Parâmetros
# ==============================================================================

with st.container():
    col_input, col_info = st.columns([2, 1])
    
    with col_input:
        st.subheader("Entrada de Dados")
        tickers_raw = st.text_input("Tickers (ex: AAPL,MSFT,PETR4.SA,VALE3.SA):", "PETR4.SA, VALE3.SA, ITUB4.SA, WEGE3.SA")
        
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Início da análise:", date(2020, 1, 1))
            rf_rate = st.number_input("Taxa Livre de Risco (Anual Decimal):", 0.0, 0.3, 0.1075, step=0.0025, help="Ex: 0.1075 para Selic a 10,75%")
        with c2:
            end_date = st.date_input("Fim da análise:", date.today())
            num_sim = st.slider("Simulações de Monte Carlo:", 1000, 20000, 5000)

    with col_info:
        st.subheader("Instruções")
        st.info("""
        1. Digite os tickers separados por vírgula.
        2. Ajuste a Taxa Livre de Risco conforme o CDI/Selic atual.
        3. O simulador buscará dados históricos para calcular a **Fronteira Eficiente**.
        """)

# ==============================================================================
# 🚀 Execução da Lógica
# ==============================================================================

if st.button("📊 Rodar Análise Completa"):
    ticker_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    
    with st.spinner("Buscando dados e calculando métricas de risco..."):
        prices, valid_tickers = get_data(ticker_list, start_date.isoformat(), end_date.isoformat())
        
        if prices is not None:
            returns = prices.pct_change().dropna()
            exp_ret = returns.mean() * 252
            cov_mat = returns.cov() * 252
            num_assets = len(valid_tickers)

            # --- Monte Carlo ---
            mc_results = np.zeros((3, num_sim))
            all_weights = []
            for i in range(num_sim):
                w = np.random.random(num_assets)
                w /= np.sum(w)
                ret, vol, shrp = calculate_portfolio_performance(w, exp_ret, cov_mat, rf_rate)
                mc_results[0,i] = ret
                mc_results[1,i] = vol
                mc_results[2,i] = shrp
                all_weights.append(w)

            # --- Otimização via SciPy ---
            bounds = tuple((0, 1) for _ in range(num_assets))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init_guess = num_assets * [1./num_assets]
            
            # Max Sharpe
            opt_sharpe = sco.minimize(neg_sharpe, init_guess, args=(exp_ret, cov_mat, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
            w_sharpe = opt_sharpe.x
            ret_s, vol_s, shrp_s = calculate_portfolio_performance(w_sharpe, exp_ret, cov_mat, rf_rate)
            var_s, cvar_s = calculate_var_cvar(returns, w_sharpe)

            # Min Vol
            opt_vol = sco.minimize(get_vol, init_guess, args=(exp_ret, cov_mat, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
            w_vol = opt_vol.x
            ret_v, vol_v, shrp_v = calculate_portfolio_performance(w_vol, exp_ret, cov_mat, rf_rate)
            var_v, cvar_v = calculate_var_cvar(returns, w_vol)

            # ==============================================================================
            # 📈 Exibição de Resultados
            # ==============================================================================
            
            st.write("---")
            st.subheader("Análise de Portfólios Ótimos")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.success("🎯 **Portfólio de Máximo Sharpe**")
                st.metric("Retorno Esperado", f"{ret_s:.2%}")
                st.metric("Volatilidade (Risco)", f"{vol_s:.2%}")
                st.metric("Índice de Sharpe", f"{shrp_s:.2f}")
                st.write("**Métricas de Cauda:**")
                st.write(f"VaR Anualizado (95%): `{var_s:.2%}`")
                st.write(f"CVaR Anualizado (95%): `{cvar_s:.2%}`")
                
            with col_res2:
                st.info("🛡️ **Portfólio de Mínima Volatilidade**")
                st.metric("Retorno Esperado", f"{ret_v:.2%}")
                st.metric("Volatilidade (Risco)", f"{vol_v:.2%}")
                st.metric("Índice de Sharpe", f"{shrp_v:.2f}")
                st.write("**Métricas de Cauda:**")
                st.write(f"VaR Anualizado (95%): `{var_v:.2%}`")
                st.write(f"CVaR Anualizado (95%): `{cvar_v:.2%}`")

            # Gráfico da Fronteira Eficiente
            st.subheader("Visualização da Fronteira Eficiente (Monte Carlo)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mc_results[1,:]*100, y=mc_results[0,:]*100,
                mode='markers', marker=dict(color=mc_results[2,:], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
                name="Simulações"
            ))
            fig.add_trace(go.Scatter(x=[vol_s*100], y=[ret_s*100], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
            fig.add_trace(go.Scatter(x=[vol_v*100], y=[ret_v*100], mode='markers', marker=dict(color='white', size=12, symbol='circle'), name="Min Vol"))
            fig.update_layout(xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Tabelas de Alocação
            st.subheader("Alocação Sugerida por Ativo (%)")
            df_weights = pd.DataFrame({
                'Ativo': valid_tickers,
                'Max Sharpe (%)': (w_sharpe * 100).round(2),
                'Min Vol (%)': (w_vol * 100).round(2)
            })
            st.table(df_weights.set_index('Ativo'))

        else:
            st.error("Erro ao buscar dados. Verifique a conexão ou os tickers informados.")

st.sidebar.markdown(f"**Usuário:** {st.session_state.get('user_name', 'Rafael Grilli')}")
st.sidebar.markdown("© 2026 Grilli Capital Research")
