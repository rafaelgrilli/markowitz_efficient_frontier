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
    page_title="Otimização de Portfólio",      
    layout="wide",                             
    initial_sidebar_state="collapsed",         
)

st.title("📊 Simulador de Portfólio Markowitz") 
st.write("⚠️ Esta ferramenta tem fins meramente educativos e não deve ser considerada conselho financeiro. Performance passada não é garantia de resultados futuros.")

# ==============================================================================
# 🔢 Funções Utilitárias (Cálculos Financeiros)
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
    if p_vol == 0:
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
# 📥 Busca de Dados (VERSÃO CORRIGIDA)
# ==============================================================================

@st.cache_data(ttl=3600)
def validate_and_fetch_tickers(input_tickers_list):
    """
    Versão mais robusta: assume que os tickers são válidos e deixa 
    o erro para a função de histórico, contornando falhas no .price
    """
    tickers = []
    invalid_tickers = []
    try:
        # Tenta uma verificação leve apenas para ver se o Yahoo retorna algo
        batch_ticker_obj = Ticker(input_tickers_list)
        # Se falhar o dicionário de preços, tentamos validar pela existência do objeto
        for ticker_symbol in input_tickers_list:
            # Simplificação: aceitamos o ticker se ele tiver o formato correto
            if ticker_symbol and len(ticker_symbol) >= 2:
                tickers.append(ticker_symbol)
            else:
                invalid_tickers.append(ticker_symbol)
    except Exception:
        # Se houver erro crítico, passamos a lista original e validamos no histórico
        tickers = input_tickers_list
    return tickers, invalid_tickers

@st.cache_data(ttl=3600)
def fetch_historical_data(valid_tickers, start_date_str, end_date_str):
    """Busca histórico de preços de fechamento ajustado."""
    try:
        ticker_obj_for_history = Ticker(valid_tickers)
        df = ticker_obj_for_history.history(start=start_date_str, end=end_date_str, interval="1d")
        
        # O Yahooquery às vezes retorna uma string se o erro for do servidor
        if isinstance(df, str) or df is None or df.empty:
            st.error("❌ O Yahoo Finance não retornou dados. Verifique os Tickers ou tente um intervalo de datas diferente.")
            st.stop()

        df = df.reset_index()
        
        # Pivotagem robusta: lida com multi-index e índices simples
        if 'symbol' in df.columns:
            prices = df.pivot(index='date', columns='symbol', values='adjclose')
        else:
            # Caso de um único ticker onde 'symbol' pode não vir no DataFrame
            prices = df.set_index('date')[['adjclose']]
            prices.columns = [valid_tickers[0]] if isinstance(valid_tickers, list) else [valid_tickers]

        prices = prices.ffill().dropna()
        final_tickers = prices.columns.tolist()
        return prices, final_tickers
    except Exception as e:
        st.error(f"❌ Erro ao processar dados históricos: {e}")
        st.stop()

@st.cache_data(ttl=3600)
def perform_optimizations(prices, risk_free_rate, num_portfolios_mc, min_weight_constraint, max_weight_constraint):
    returns = prices.pct_change().dropna()
    if returns.empty:
        st.error("❌ Dados insuficientes para calcular retornos (precisamos de pelo menos 2 dias de preços).")
        st.stop()

    annualization_factor = 252
    expected_returns = returns.mean() * annualization_factor
    std_devs = returns.std() * np.sqrt(annualization_factor)
    cov_matrix = returns.cov() * annualization_factor
    tickers = prices.columns.tolist()
    num_assets = len(tickers)
    correlation_matrix = returns.corr().values if num_assets > 1 else np.array([[1.0]])

    # --- Monte Carlo ---
    portfolio_returns_mc, portfolio_risks_mc, portfolio_weights_mc, sharpe_ratios_mc = [], [], [], []
    progress_bar_mc = st.progress(0)
    
    for i in range(num_portfolios_mc):
        weights = random_weights(num_assets)
        ret = portfolio_return(weights, expected_returns)
        risk = portfolio_risk(weights, std_devs, correlation_matrix)
        sr = sharpe_ratio(ret, risk, risk_free_rate)
        portfolio_returns_mc.append(ret)
        portfolio_risks_mc.append(risk)
        portfolio_weights_mc.append(weights)
        sharpe_ratios_mc.append(sr)
        if (i + 1) % 1000 == 0 or (i + 1) == num_portfolios_mc:
            progress_bar_mc.progress((i + 1) / num_portfolios_mc)
    progress_bar_mc.empty()

    # Otimização via SciPy
    bounds = tuple((min_weight_constraint, max_weight_constraint) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = num_assets * [1./num_assets]

    opt_sharpe = sco.minimize(neg_sharpe_ratio, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    opt_var = sco.minimize(get_portfolio_volatility, initial_weights, args=(expected_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

    # Pontos Ótimos do Monte Carlo
    optimal_index_mc = sharpe_ratios_mc.index(max(sharpe_ratios_mc))
    min_index_mc = portfolio_risks_mc.index(min(portfolio_risks_mc))

    return {
        "tickers": tickers, "expected_returns": expected_returns, "std_devs": std_devs,
        "correlation_matrix": correlation_matrix, "cov_matrix": cov_matrix, "num_assets": num_assets,
        "portfolio_returns_mc": portfolio_returns_mc, "portfolio_risks_mc": portfolio_risks_mc,
        "portfolio_weights_mc": portfolio_weights_mc, "sharpe_ratios_mc": sharpe_ratios_mc,
        "optimal_sharpe_weights_scipy": opt_sharpe.x, "optimal_sharpe_return_scipy": portfolio_return_scipy(opt_sharpe.x, expected_returns),
        "optimal_sharpe_risk_scipy": portfolio_risk_scipy(opt_sharpe.x, cov_matrix), "max_sharpe_ratio_scipy": -opt_sharpe.fun,
        "min_variance_weights_scipy": opt_var.x, "min_variance_return_scipy": portfolio_return_scipy(opt_var.x, expected_returns),
        "min_variance_risk_scipy": portfolio_risk_scipy(opt_var.x, cov_matrix), "min_variance_sharpe_scipy": sharpe_ratio(portfolio_return_scipy(opt_var.x, expected_returns), portfolio_risk_scipy(opt_var.x, cov_matrix), risk_free_rate),
        "min_weights_mc": portfolio_weights_mc[min_index_mc], "min_return_mc": portfolio_returns_mc[min_index_mc], "min_risk_mc": portfolio_risks_mc[min_index_mc], "min_sharpe_mc": sharpe_ratios_mc[min_index_mc],
        "optimal_weights_mc": portfolio_weights_mc[optimal_index_mc], "optimal_return_mc": portfolio_returns_mc[optimal_index_mc], "optimal_risk_mc": portfolio_risks_mc[optimal_index_mc], "max_sharpe_ratio_mc": sharpe_ratios_mc[optimal_index_mc],
        "returns_df": returns
    }

# ==============================================================================
# 🎛️ Seção de Interface do Usuário (Layout Original)
# ==============================================================================

st.subheader("Configuração")

with st.expander("❓ Como Usar Esta Ferramenta", expanded=False):
    st.markdown("""
    Este simulador ajuda você a explorar a Fronteira Eficiente usando a Teoria de Portfólio de Markowitz.

    1. **Insira os Símbolos (Tickers)**: Forneça os códigos das ações ou ETFs separados por vírgula (ex: `AAPL, MSFT, GOOG` ou ativos brasileiros como `PETR4.SA, VALE3.SA, ITUB4.SA`).
    2. **Selecione o Intervalo de Datas**: Escolha o período histórico para análise dos dados.
    3. **Taxa Livre de Risco**: Insira a taxa anual livre de risco (ex: `0.1075` para 10,75% Selic). Isso é crucial para o cálculo do Índice de Sharpe.
    4. **Número de Portfólios**: Ajuste o controle deslizante para a simulação de Monte Carlo. Mais portfólios geram uma fronteira visualmente mais densa, mas levam mais tempo para calcular.
    5. **Restrições de Portfólio**: Defina pesos mínimos e máximos por ativo e escolha se deseja permitir vendas a descoberto (short sales).
    6. **Executar Simulação**: Clique no botão para buscar os dados e realizar as otimizações via Monte Carlo e SciPy.

    A ferramenta exibirá estatísticas dos ativos, o gráfico da Fronteira Eficiente, alocações ótimas e medidas de risco como VaR e CVaR.
    """)

col1, col2 = st.columns([2, 1])
with col1:
    tickers_input = st.text_input('Insira os Tickers (separados por vírgula):', 'AAPL, MSFT, GOOG')
with col2:
    risk_free_rate_input = st.number_input('Taxa Livre de Risco (anual decimal):', 0.0, 0.2, 0.04, step=0.001, format="%.3f", help="Ex: 0.04 para 4%")

col3, col4 = st.columns(2)
with col3:
    start_date_value = st.date_input('Data Inicial:', date(2018, 1, 1))
with col4:
    end_date_value = st.date_input('Data Final:', date.today())

num_portfolios_value = st.slider('Número de Portfólios para Simulação Monte Carlo:', 1000, 50000, 10000, 1000)

st.subheader("Restrições do Portfólio")
allow_short_sales = st.checkbox("Permitir Venda a Descoberto (Pesos Negativos)", value=False)
min_weight_floor = -2.0 if allow_short_sales else 0.0
max_weight_ceiling = 2.0 if allow_short_sales else 1.0

col5, col6 = st.columns(2)
with col5:
    global_min_weight = st.number_input('Peso Mínimo por Ativo:', min_weight_floor, 1.0, 0.0 if not allow_short_sales else -0.5, step=0.01, format="%.2f")
with col6:
    global_max_weight = st.number_input('Peso Máximo por Ativo:', 0.0, max_weight_ceiling, 1.0, step=0.01, format="%.2f")

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

run_button = st.button("📈 Executar Otimização de Portfólio")

# ==============================================================================
# 🚀 LÓGICA PRINCIPAL DE EXECUÇÃO
# ==============================================================================

if run_button:
    st.session_state.simulation_run = True
    input_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    with st.spinner("Processando simulações e otimizações..."):
        valid, invalid = validate_and_fetch_tickers(input_tickers)
        # Mostramos o aviso apenas se houver algo realmente inválido detectado na pré-validação
        if invalid: st.warning(f"⚠️ Atenção com os tickers: {', '.join(invalid)}")
        
        prices, final_valid = fetch_historical_data(valid, start_date_value.isoformat(), end_date_value.isoformat())
        st.session_state.optimization_results = perform_optimizations(prices, risk_free_rate_input, num_portfolios_value, global_min_weight, global_max_weight)
        
        # Construção do Gráfico Plotly
        res = st.session_state.optimization_results
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.array(res["portfolio_risks_mc"])*100, y=np.array(res["portfolio_returns_mc"])*100,
            mode='markers', marker=dict(color=res["sharpe_ratios_mc"], colorscale='Viridis', showscale=True, title="Sharpe"),
            name="Monte Carlo"
        ))
        fig.add_trace(go.Scatter(
            x=[res["optimal_sharpe_risk_scipy"]*100], y=[res["optimal_sharpe_return_scipy"]*100],
            mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Máximo Sharpe (SciPy)"
        ))
        fig.update_layout(title="Fronteira Eficiente", xaxis_title="Risco (Volatilidade %)", yaxis_title="Retorno Esperado (%)", template="plotly_white")
        st.session_state.plotly_fig = fig

# --- Exibição de Resultados ---
if st.session_state.simulation_run and st.session_state.optimization_results:
    res = st.session_state.optimization_results
    
    st.subheader("Estatísticas dos Ativos")
    st.write("📈 Retornos Anuais Esperados (%):")
    st.dataframe((res["expected_returns"] * 100).round(2).rename("Retorno (%)"))
    st.write("📊 Volatilidade Anual (%):")
    st.dataframe((res["std_devs"] * 100).round(2).rename("Volatilidade (%)"))
    st.write("🔗 Matriz de Correlação:")
    st.dataframe(pd.DataFrame(res["correlation_matrix"], index=res["tickers"], columns=res["tickers"]).round(2))

    st.subheader("Portfólios Ótimos (Otimização SciPy)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("🚀 **Carteira de Máximo Sharpe:**")
        st.dataframe(pd.DataFrame({'Ativo': res["tickers"], 'Peso (%)': (res["optimal_sharpe_weights_scipy"] * 100).round(2)}))
        st.write(f"Retorno: {res['optimal_sharpe_return_scipy']*100:.2f}% | Risco: {res['optimal_sharpe_risk_scipy']*100:.2f}% | Sharpe: {res['max_sharpe_ratio_scipy']:.2f}")
    with col_b:
        st.write("🌟 **Carteira de Mínima Variância:**")
        st.dataframe(pd.DataFrame({'Ativo': res["tickers"], 'Peso (%)': (res["min_variance_weights_scipy"] * 100).round(2)}))
        st.write(f"Retorno: {res['min_variance_return_scipy']*100:.2f}% | Risco: {res['min_variance_risk_scipy']*100:.2f}% | Sharpe: {res['min_variance_sharpe_scipy']:.2f}")

    st.subheader("Gráfico da Fronteira Eficiente")
    st.plotly_chart(st.session_state.plotly_fig, use_container_width=True)

    # Medidas de Risco
    st.subheader("Medidas de Risco do Portfólio (VaR & CVaR)")
    p_rets_opt = res["returns_df"].dot(res["optimal_sharpe_weights_scipy"])
    v95, cv95 = calculate_historical_var_cvar(p_rets_opt, annualize=True)
    st.write(f"**Máximo Sharpe:** VaR Anualizado (95%): {v95:.2%} | CVaR Anualizado: {cv95:.2%}")

    # Download
    csv_output = io.StringIO()
    pd.DataFrame({'Ativo': res["tickers"], 'Peso Max Sharpe': res["optimal_sharpe_weights_scipy"]}).to_csv(csv_output, index=False)
    st.download_button(label="📥 Baixar Alocação (CSV)", data=csv_output.getvalue(), file_name="alocacao_portfolio.csv", mime="text/csv")

# Footer Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Sobre este App")
st.sidebar.info("Desenvolvido por Rafael Grilli Felizardo.")
st.sidebar.markdown("© 2025 Rafael Grilli Felizardo")
