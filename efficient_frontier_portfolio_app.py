# streamlit_app.py

# ==============================================================================
# 📦 Importação de bibliotecas
# ==============================================================================
import streamlit as st          
from yahooquery import Ticker   
import pandas as pd             
import numpy as np              
import plotly.graph_objects as go 
from datetime import date       
import io                       
import scipy.optimize as sco    

# ==============================================================================
# ⚙️ Configuração da Página Streamlit
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Optimization",      
    layout="wide",                             
    initial_sidebar_state="collapsed",         
)

st.title("📊 Simulador de Portfólio Markowitz") 
st.write("⚠️ Esta ferramenta tem fins meramente educativos e não deve ser considerada conselho financeiro. Performance passada não é garantia de resultados futuros.")

# ==============================================================================
# 🔢 Funções Utilitárias
# ==============================================================================

def random_weights(n):
    weights = np.random.rand(n)
    return weights / weights.sum()

def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

def portfolio_risk(weights, std_devs, correlation_matrix):
    cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix
    variance = np.dot(weights, np.dot(cov_matrix, weights))
    return np.sqrt(variance)

def sharpe_ratio(portfolio_ret, portfolio_risk, risk_free_rate):
    if portfolio_risk == 0:
        return 0
    return (portfolio_ret - risk_free_rate) / portfolio_risk

def portfolio_return_scipy(weights, expected_returns):
    return np.sum(expected_returns * weights)

def portfolio_risk_scipy(weights, cov_matrix):
    weights = np.array(weights).reshape(-1, 1)
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance[0, 0])

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    p_ret = portfolio_return_scipy(weights, expected_returns)
    p_vol = portfolio_risk_scipy(weights, cov_matrix)
    if p_vol <= 0:
        return np.inf
    return - (p_ret - risk_free_rate) / p_vol

def get_portfolio_volatility(weights, expected_returns, cov_matrix):
    return portfolio_risk_scipy(weights, cov_matrix)

def calculate_historical_var_cvar(portfolio_returns_series, confidence_level=0.95, annualize=False):
    if not isinstance(portfolio_returns_series, (pd.Series, np.ndarray)):
        portfolio_returns_series = np.array(portfolio_returns_series)
    sorted_returns = np.sort(portfolio_returns_series)
    var_index = int(np.floor(len(sorted_returns) * (1 - confidence_level)))
    var = sorted_returns[var_index]
    cvar_returns = sorted_returns[sorted_returns <= var]
    cvar = np.mean(cvar_returns)
    if annualize:
        var *= np.sqrt(252)
        cvar *= np.sqrt(252)
    return -var, -cvar

# ==============================================================================
# 📥 Busca de Dados (Corrigida)
# ==============================================================================

@st.cache_data(ttl=3600)
def validate_and_fetch_tickers(input_tickers_list):
    tickers = []
    invalid_tickers = []
    try:
        # Apenas limpamos a lista, a validação real ocorrerá na busca do histórico
        for t in input_tickers_list:
            if t and len(t) >= 2:
                tickers.append(t)
            else:
                invalid_tickers.append(t)
    except Exception:
        tickers = input_tickers_list
    return tickers, invalid_tickers

@st.cache_data(ttl=3600)
def fetch_historical_data(valid_tickers, start_date_str, end_date_str):
    try:
        ticker_obj = Ticker(valid_tickers)
        df = ticker_obj.history(start=start_date_str, end=end_date_str, interval="1d")
        
        if df is None or (isinstance(df, pd.DataFrame) and df.empty) or isinstance(df, str):
            st.error("❌ O Yahoo Finance não retornou dados para estes ativos no período selecionado.")
            st.stop()

        df = df.reset_index()
        if 'symbol' in df.columns:
            prices = df.pivot(index='date', columns='symbol', values='adjclose')
        else:
            prices = df.set_index('date')[['adjclose']]
            prices.columns = valid_tickers

        prices = prices.ffill().dropna()
        if prices.empty:
            st.error("❌ Dados insuficientes após limpeza (NaNs).")
            st.stop()
            
        return prices, prices.columns.tolist()
    except Exception as e:
        st.error(f"❌ Erro ao buscar dados: {e}")
        st.stop()

@st.cache_data(ttl=3600)
def perform_optimizations(prices, risk_free_rate, num_portfolios_mc, min_weight_constraint, max_weight_constraint):
    returns = prices.pct_change().dropna()
    annualization_factor = 252
    expected_returns = returns.mean() * annualization_factor
    std_devs = returns.std() * np.sqrt(annualization_factor)
    cov_matrix = returns.cov() * annualization_factor
    tickers = prices.columns.tolist()
    num_assets = len(tickers)
    correlation_matrix = returns.corr().values if num_assets > 1 else np.array([[1.0]])

    # Monte Carlo
    portfolio_returns_mc, portfolio_risks_mc, portfolio_weights_mc, sharpe_ratios_mc = [], [], [], []
    progress_bar = st.progress(0)
    for i in range(num_portfolios_mc):
        weights = random_weights(num_assets)
        ret = portfolio_return(weights, expected_returns)
        risk = portfolio_risk(weights, std_devs, correlation_matrix)
        sr = sharpe_ratio(ret, risk, risk_free_rate)
        portfolio_returns_mc.append(ret)
        portfolio_risks_mc.append(risk)
        portfolio_weights_mc.append(weights)
        sharpe_ratios_mc.append(sr)
        if (i + 1) % 1000 == 0: progress_bar.progress((i + 1) / num_portfolios_mc)
    progress_bar.empty()

    # SciPy
    bounds = tuple((min_weight_constraint, max_weight_constraint) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_w = num_assets * [1./num_assets]
    opt_s = sco.minimize(neg_sharpe_ratio, init_w, args=(expected_returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    opt_v = sco.minimize(get_portfolio_volatility, init_w, args=(expected_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

    return {
        "tickers": tickers, "expected_returns": expected_returns, "std_devs": std_devs,
        "correlation_matrix": correlation_matrix, "cov_matrix": cov_matrix,
        "portfolio_returns_mc": portfolio_returns_mc, "portfolio_risks_mc": portfolio_risks_mc,
        "sharpe_ratios_mc": sharpe_ratios_mc, "returns_df": returns,
        "opt_s_weights": opt_s.x, "opt_s_ret": portfolio_return_scipy(opt_s.x, expected_returns),
        "opt_s_vol": portfolio_risk_scipy(opt_s.x, cov_matrix), "opt_s_sr": -opt_s.fun,
        "opt_v_weights": opt_v.x, "opt_v_ret": portfolio_return_scipy(opt_v.x, expected_returns),
        "opt_v_vol": portfolio_risk_scipy(opt_v.x, cov_matrix)
    }

# ==============================================================================
# 🎛️ Interface (Layout Original Restaurado)
# ==============================================================================

st.subheader("Configuração")

with st.expander("❓ Como Usar Esta Ferramenta", expanded=False):
    st.markdown("""
    Este simulador ajuda você a explorar a Fronteira Eficiente usando a Teoria de Portfólio de Markowitz.
    1. **Insira os Tickers**: Códigos separados por vírgula (ex: `AAPL, MSFT, PETR4.SA`).
    2. **Intervalo de Datas**: Período para análise histórica.
    3. **Taxa Livre de Risco**: Usada para o Índice de Sharpe (decimal, ex: 0.1075).
    4. **Restrições**: Defina limites de peso por ativo.
    """)

col1, col2 = st.columns([2, 1])
with col1:
    tickers_input = st.text_input('Insira os Tickers:', 'AAPL, MSFT, GOOG')
with col2:
    risk_free_rate_input = st.number_input('Taxa Livre de Risco (anual decimal):', 0.0, 0.2, 0.04, step=0.001, format="%.3f")

col3, col4 = st.columns(2)
with col3:
    start_date_value = st.date_input('Data Inicial:', date(2018, 1, 1))
with col4:
    end_date_value = st.date_input('Data Final:', date.today())

num_portfolios_value = st.slider('Número de Portfólios (Monte Carlo):', 1000, 50000, 10000, 1000)

st.subheader("Restrições do Portfólio")
allow_short = st.checkbox("Permitir Venda a Descoberto", value=False)
col5, col6 = st.columns(2)
with col5:
    g_min = st.number_input('Peso Mínimo:', -2.0 if allow_short else 0.0, 1.0, 0.0, step=0.01)
with col6:
    g_max = st.number_input('Peso Máximo:', 0.0, 2.0 if allow_short else 1.0, 1.0, step=0.01)

if 'simulation_run' not in st.session_state: st.session_state.simulation_run = False

if st.button("📈 Executar Otimização"):
    st.session_state.simulation_run = True
    list_t = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    with st.spinner("Calculando..."):
        v_t, _ = validate_and_fetch_tickers(list_t)
        prices, f_v = fetch_historical_data(v_t, start_date_value.isoformat(), end_date_value.isoformat())
        st.session_state.results = perform_optimizations(prices, risk_free_rate_input, num_portfolios_value, g_min, g_max)

if st.session_state.simulation_run:
    r = st.session_state.results
    
    st.subheader("Estatísticas e Portfólios Ótimos")
    c_a, c_b = st.columns(2)
    with c_a:
        st.write("**Máximo Sharpe:**")
        st.write(f"Retorno: {r['opt_s_ret']:.2%} | Risco: {r['opt_s_vol']:.2%} | Sharpe: {r['opt_s_sr']:.2f}")
        st.dataframe(pd.DataFrame({'Ativo': r["tickers"], 'Peso (%)': (r["opt_s_weights"]*100).round(2)}))
    with c_b:
        st.write("**Mínima Variância:**")
        st.write(f"Retorno: {r['opt_v_ret']:.2%} | Risco: {r['opt_v_vol']:.2%}")
        st.dataframe(pd.DataFrame({'Ativo': r["tickers"], 'Peso (%)': (r["opt_v_weights"]*100).round(2)}))

    # GRÁFICO CORRIGIDO (O erro estava aqui)
    st.subheader("Gráfico da Fronteira Eficiente")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.array(r["portfolio_risks_mc"])*100, 
        y=np.array(r["portfolio_returns_mc"])*100,
        mode='markers', 
        marker=dict(
            color=r["sharpe_ratios_mc"], 
            colorscale='Viridis', 
            showscale=True, 
            colorbar=dict(title="Sharpe") # Título corrigido aqui
        ),
        name="Monte Carlo"
    ))
    fig.add_trace(go.Scatter(x=[r['opt_s_vol']*100], y=[r['opt_s_ret']*100], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.info("Desenvolvido por Rafael Grilli Felizardo.")
