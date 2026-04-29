# streamlit_app.py

# ==============================================================================
# 📦 Import necessary libraries
# ==============================================================================
import streamlit as st          # Streamlit library for building interactive web apps
from yahooquery import Ticker   # For fetching historical stock data from Yahoo Finance
import pandas as pd             # For data manipulation and analysis, especially DataFrames
import numpy as np              # For numerical operations, especially array manipulations
import plotly.graph_objects as go # Plotly for more customized graphs
from datetime import date       # For handling date inputs
import io                       # For handling in-memory binary streams (used for CSV download)
import scipy.optimize as sco    # SciPy's optimization module for finding precise optimal portfolios

# ==============================================================================
# ⚙️ Streamlit Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Optimization",      # Title that appears in the browser tab
    layout="wide",                             # Use wide layout for more horizontal space
    initial_sidebar_state="collapsed",         # Sidebar starts collapsed, can be expanded
)

st.title("📊 Markowitz Portfolio Simulator") # Main title of the application
st.write("⚠️ This tool is for educational purposes only and should not be considered financial advice. Past performance is not indicative of future results.")

# ==============================================================================
# 🔢 Utility Functions
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
# 📥 Data Fetching (FIXED)
# ==============================================================================

@st.cache_data(ttl=3600)
def validate_and_fetch_tickers(input_tickers_list):
    tickers = []
    invalid_tickers = []
    try:
        batch_ticker_obj = Ticker(input_tickers_list)
        # Fix: Using 'price' can be unstable, checking keys is safer
        price_data_check = batch_ticker_obj.price
        for ticker_symbol in input_tickers_list:
            if isinstance(price_data_check, dict) and ticker_symbol in price_data_check and 'regularMarketPrice' in price_data_check[ticker_symbol]:
                tickers.append(ticker_symbol)
            else:
                invalid_tickers.append(ticker_symbol)
    except Exception as e:
        st.error(f"❌ Error during initial ticker validation: {e}")
        st.stop()
    return tickers, invalid_tickers

@st.cache_data(ttl=3600)
def fetch_historical_data(valid_tickers, start_date_str, end_date_str):
    try:
        ticker_obj_for_history = Ticker(valid_tickers)
        df = ticker_obj_for_history.history(start=start_date_str, end=end_date_str, interval="1d")
        
        if isinstance(df, str) or df.empty:
            st.error("❌ No data found on Yahoo Finance. Try adjusting the date range.")
            st.stop()

        df = df.reset_index()
        # Fix: handle single vs multi-ticker dataframes
        if 'symbol' in df.columns:
            prices = df.pivot(index='date', columns='symbol', values='adjclose')
        else:
            prices = df.set_index('date')[['adjclose']]
            prices.columns = valid_tickers

        prices = prices.ffill().dropna()
        final_tickers = prices.columns.tolist()
        return prices, final_tickers
    except Exception as e:
        st.error(f"❌ Error fetching historical data: {e}")
        st.stop()

@st.cache_data(ttl=3600)
def perform_optimizations(prices, risk_free_rate, num_portfolios_mc, min_weight_constraint, max_weight_constraint):
    returns = prices.pct_change().dropna()
    if returns.empty:
        st.error("❌ Not enough data points for returns.")
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
        if (i + 1) % 1000 == 0:
            progress_bar_mc.progress((i + 1) / num_portfolios_mc)
    progress_bar_mc.empty()

    # Optimal MC points
    optimal_index_mc = sharpe_ratios_mc.index(max(sharpe_ratios_mc))
    min_index_mc = portfolio_risks_mc.index(min(portfolio_risks_mc))

    # --- SciPy Optimization ---
    bounds = tuple((min_weight_constraint, max_weight_constraint) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = num_assets * [1./num_assets]

    # Max Sharpe
    opt_sharpe = sco.minimize(neg_sharpe_ratio, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    # Min Variance
    opt_var = sco.minimize(get_portfolio_volatility, initial_weights, args=(expected_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

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
# 🎛️ UI Section (YOUR ORIGINAL LAYOUT)
# ==============================================================================

st.subheader("Configuration")

with st.expander("❓ How to Use This Tool", expanded=False):
    st.markdown("""
    This simulator helps you explore the Efficient Frontier using Markowitz Portfolio Theory.
    1. **Enter Ticker Symbols**: Provide comma-separated symbols (e.g., `AAPL,MSFT,GOOG` ou `PETR4.SA, VALE3.SA`).
    ... (Rest of your instructions)
    """)

col1, col2 = st.columns([2, 1])
with col1:
    tickers_input = st.text_input('Enter Ticker Symbols (comma-separated):', 'AAPL,MSFT,GOOG')
with col2:
    risk_free_rate_input = st.number_input('Risk-Free Rate (annual %):', 0.0, 0.2, 0.04, step=0.001, format="%.3f")

col3, col4 = st.columns(2)
with col3:
    start_date_value = st.date_input('Start Date:', date(2018, 1, 1))
with col4:
    end_date_value = st.date_input('End Date:', date(2024, 12, 31))

num_portfolios_value = st.slider('Number of Portfolios for Monte Carlo Simulation:', 1000, 50000, 10000, 1000)

st.subheader("Portfolio Constraints")
allow_short_sales = st.checkbox("Allow Short Sales (Negative Weights)", value=False)
min_weight_floor = -2.0 if allow_short_sales else 0.0
max_weight_ceiling = 2.0 if allow_short_sales else 1.0

col5, col6 = st.columns(2)
with col5:
    global_min_weight = st.number_input('Minimum Weight per Asset:', min_weight_floor, 1.0, 0.0 if not allow_short_sales else -0.5, step=0.01, format="%.2f")
with col6:
    global_max_weight = st.number_input('Maximum Weight per Asset:', 0.0, max_weight_ceiling, 1.0, step=0.01, format="%.2f")

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

run_button = st.button("📈 Run Portfolio Optimization")

# ==============================================================================
# 🚀 MAIN LOGIC
# ==============================================================================

if run_button:
    st.session_state.simulation_run = True
    input_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    with st.spinner("Processing..."):
        valid, invalid = validate_and_fetch_tickers(input_tickers)
        if invalid: st.warning(f"⚠️ Invalid tickers: {', '.join(invalid)}")
        if not valid: st.error("❌ No valid tickers."); st.stop()
        
        prices, final_valid = fetch_historical_data(valid, start_date_value.isoformat(), end_date_value.isoformat())
        st.session_state.optimization_results = perform_optimizations(prices, risk_free_rate_input, num_portfolios_value, global_min_weight, global_max_weight)

# --- Display Results ---
if st.session_state.simulation_run and st.session_state.optimization_results:
    res = st.session_state.optimization_results
    
    st.subheader("Asset Statistics")
    st.write("📈 Expected Annual Returns (%):")
    st.dataframe((res["expected_returns"] * 100).round(2).rename("Return (%)"))
    st.write("📊 Annual Volatility (%):")
    st.dataframe((res["std_devs"] * 100).round(2).rename("Volatility (%)"))
    st.write("🔗 Correlation Matrix:")
    st.dataframe(pd.DataFrame(res["correlation_matrix"], index=res["tickers"], columns=res["tickers"]).round(2))

    st.subheader("Optimal Portfolios (SciPy Optimization)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("🚀 **Optimal Risky Portfolio (Max Sharpe):**")
        st.dataframe(pd.DataFrame({'Ticker': res["tickers"], 'Weight (%)': (res["optimal_sharpe_weights_scipy"] * 100).round(2)}))
        st.write(f"Return: {res['optimal_sharpe_return_scipy']*100:.2f}% | Vol: {res['optimal_sharpe_risk_scipy']*100:.2f}% | Sharpe: {res['max_sharpe_ratio_scipy']:.2f}")
    with col_b:
        st.write("🌟 **Minimum Variance Portfolio:**")
        st.dataframe(pd.DataFrame({'Ticker': res["tickers"], 'Weight (%)': (res["min_variance_weights_scipy"] * 100).round(2)}))
        st.write(f"Return: {res['min_variance_return_scipy']*100:.2f}% | Vol: {res['min_variance_risk_scipy']*100:.2f}% | Sharpe: {res['min_variance_sharpe_scipy']:.2f}")

    # (Add your Plotly figure code here using res["portfolio_risks_mc"] etc.)
    # (Add your Risk Measures VaR/CVaR code here)

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.header("About This App")
st.sidebar.info("Developed by Rafael Grilli Felizardo...")
