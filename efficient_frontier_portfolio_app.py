#!/usr/bin/env python
# coding: utf-8

# In[1]:


# first improvement: improved user experience and more advanced optimization tools

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
# Plotly's static image export requires kaleido, which is implicitly used by fig.to_image
# import kaleido.scopes.plotly # Not directly imported, but needs to be installed


# ==============================================================================
# ⚙️ Streamlit Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Optimization",      # Title that appears in the browser tab
    layout="wide",                            # Use wide layout for more horizontal space
    initial_sidebar_state="collapsed",        # Sidebar starts collapsed, can be expanded
    # icon="📈" # Optional: Add a favicon emoji for the page tab
)

st.title("📊 Markowitz Portfolio Simulator") # Main title of the application
st.write("⚠️ This tool is for educational purposes only and should not be considered financial advice. Past performance is not indicative of future results.")
# Disclaimer message to inform users about the non-advisory nature of the tool.

# ==============================================================================
# 🔢 Utility Functions
# These functions perform the core calculations for portfolio theory.
# ==============================================================================

def random_weights(n):
    """
    Generates 'n' random weights that sum to 1.
    Used for Monte Carlo simulation to create diverse portfolios.
    """
    weights = np.random.rand(n) # Generate 'n' random numbers between 0 and 1
    return weights / weights.sum() # Normalize them so they sum to 1 (representing proportions)

def portfolio_return(weights, expected_returns):
    """
    Calculates the expected annual return of a portfolio given asset weights
    and individual asset expected returns.
    This version is used for Monte Carlo where expected_returns are already annualized.
    """
    return np.dot(weights, expected_returns) # Dot product sums (weight * return) for each asset

def portfolio_risk(weights, std_devs, correlation_matrix):
    """
    Calculates the expected annual volatility (risk) of a portfolio.
    Uses asset standard deviations and their correlation matrix.
    This version is used for Monte Carlo.
    """
    # Create a covariance matrix from standard deviations and correlation matrix
    # Outer product creates a matrix where element (i,j) is std_i * std_j
    cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix
    # Calculate portfolio variance: w^T * Cov * w
    variance = np.dot(weights, np.dot(cov_matrix, weights))
    return np.sqrt(variance) # Volatility is the square root of variance

def sharpe_ratio(portfolio_ret, portfolio_risk, risk_free_rate):
    """
    Calculates the Sharpe Ratio of a portfolio.
    Measures risk-adjusted return (excess return per unit of risk).
    """
    if portfolio_risk == 0: # Avoid division by zero if risk is somehow 0
        return 0
    return (portfolio_ret - risk_free_rate) / portfolio_risk

# --- SciPy specific functions ---
# These functions are designed to be compatible with scipy.optimize.minimize
# which requires functions to minimize (or maximize by minimizing their negative).

def portfolio_return_scipy(weights, expected_returns):
    """
    Calculates portfolio return for SciPy optimization.
    Assumes expected_returns are already annualized.
    """
    return np.sum(expected_returns * weights) # Sum of (weight * annualized return)

def portfolio_risk_scipy(weights, cov_matrix):
    """
    Calculates portfolio volatility (risk) for SciPy optimization.
    Requires the annualized covariance matrix directly.
    """
    weights = np.array(weights).reshape(-1, 1) # Reshape weights to a column vector for matrix multiplication
    # Variance = w^T * Cov * w
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance[0, 0]) # Extract the single scalar value from the 1x1 matrix

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    Calculates the negative Sharpe Ratio.
    This function is minimized by scipy.optimize to find the maximum Sharpe Ratio.
    """
    p_ret = portfolio_return_scipy(weights, expected_returns)
    p_vol = portfolio_risk_scipy(weights, cov_matrix)
    if p_vol == 0: # Handle cases where risk is zero to avoid division by zero
        return np.inf # Return infinity so this portfolio is not chosen for max Sharpe
    return - (p_ret - risk_free_rate) / p_vol # Return the negative value

def get_portfolio_volatility(weights, expected_returns, cov_matrix):
    """
    Helper function to get portfolio volatility directly, used for minimizing risk.
    """
    return portfolio_risk_scipy(weights, cov_matrix) # Simply returns the annualized volatility

# --- Risk Measures specific functions ---
def calculate_historical_var_cvar(portfolio_returns_series, confidence_level=0.95, annualize=False):
    """
    Calculates historical VaR and CVaR (Expected Shortfall) for a series of portfolio returns.

    Args:
        portfolio_returns_series (pd.Series or np.array): Daily or weekly returns of the portfolio.
        confidence_level (float): Confidence level for VaR/CVaR (e.95 for 95%).
        annualize (bool): Whether to annualize the VaR/CVaR. Set to True if input returns are daily/weekly
                          and you want an annualized measure. Assumes 252 trading days.

    Returns:
        tuple: (VaR, CVaR) at the specified confidence level.
    """
    if not isinstance(portfolio_returns_series, (pd.Series, np.ndarray)):
        portfolio_returns_series = np.array(portfolio_returns_series)

    # Sort returns in ascending order
    sorted_returns = np.sort(portfolio_returns_series)

    # Calculate VaR
    # The index for VaR at (1-confidence_level) percentile
    # E.g., for 95% confidence, we want the 5th percentile (index 0.05 * N)
    var_index = int(np.floor(len(sorted_returns) * (1 - confidence_level)))
    var = sorted_returns[var_index] # This is the return at the VaR threshold

    # Calculate CVaR (average of returns below VaR)
    cvar_returns = sorted_returns[sorted_returns <= var]
    cvar = np.mean(cvar_returns)

    # Annualize if requested (typical for daily returns)
    if annualize:
        # For simplicity, scaling by sqrt(252) is often used for consistency with std dev.
        # This is an approximation and assumes returns are independently and identically distributed.
        var *= np.sqrt(252)
        cvar *= np.sqrt(252)

    # Return as positive values (representing losses)
    return -var, -cvar

# ==============================================================================
# @st.cache_data for Performance Optimization
# These functions will cache their results based on input parameters.
# If inputs don't change, the function won't re-run, speeding up the app.
# `ttl` (time to live) can be set to clear cache after a certain time (e.g., 3600 seconds = 1 hour)
# ==============================================================================

@st.cache_data(ttl=3600) # Cache data for 1 hour
def validate_and_fetch_tickers(input_tickers_list):
    """
    Validates ticker symbols by checking for real-time price data and returns
    lists of valid and invalid tickers. Caches the result.
    """
    print("DEBUG: validate_and_fetch_tickers: Starting ticker validation.") # Debug print
    tickers = []
    invalid_tickers = []
    try:
        batch_ticker_obj = Ticker(input_tickers_list)
        price_data_check = batch_ticker_obj.price
        for ticker_symbol in input_tickers_list:
            if ticker_symbol in price_data_check and 'regularMarketPrice' in price_data_check[ticker_symbol]:
                tickers.append(ticker_symbol)
            else:
                invalid_tickers.append(ticker_symbol)
        print(f"DEBUG: validate_and_fetch_tickers: Found {len(tickers)} valid and {len(invalid_tickers)} invalid tickers.") # Debug print
    except Exception as e:
        print(f"ERROR: validate_and_fetch_tickers: {e}") # Debug print for error
        st.error(f"❌ Error during initial ticker validation: {e}. Please check your internet connection or API limits.")
        st.stop() # Stop execution if there's a fundamental issue fetching prices
    return tickers, invalid_tickers

@st.cache_data(ttl=3600) # Cache historical data for 1 hour
def fetch_historical_data(valid_tickers, start_date_str, end_date_str):
    """
    Fetches historical adjusted close price data for valid tickers. Caches the result.
    Handles pivoting and initial data cleaning.
    """
    print(f"DEBUG: fetch_historical_data: Starting data fetch for {len(valid_tickers)} tickers from {start_date_str} to {end_date_str}.") # Debug print
    try:
        ticker_obj_for_history = Ticker(" ".join(valid_tickers))
        df = ticker_obj_for_history.history(start=start_date_str, end=end_date_str, interval="1d")
        print("DEBUG: fetch_historical_data: Raw data fetched from yahooquery.") # Debug print
    except Exception as e:
        print(f"ERROR: fetch_historical_data: {e}") # Debug print for error
        st.error(f"❌ Error fetching historical data for valid tickers: {e}. Check internet connection or API limits.")
        st.stop()

    if df.empty:
        print("DEBUG: fetch_historical_data: DataFrame is empty after initial fetch.") # Debug print
        st.error("❌ No historical price data found for the valid tickers in the specified date range. Adjust inputs.")
        st.stop()

    # Ensure multi-level index is handled for 'adjclose' extraction
    if 'symbol' in df.index.names:
        prices = df.reset_index().pivot(index='date', columns='symbol', values='adjclose')
        print("DEBUG: fetch_historical_data: Pivoted DataFrame with 'symbol' in index.") # Debug print
    else: # Handle case of single ticker where 'symbol' is not an index level
        prices = df[['adjclose']]
        prices.columns = valid_tickers # Assign the single ticker name
        print("DEBUG: fetch_historical_data: Handled single ticker case.") # Debug print

    # Filter out tickers that might not have data in the specified range
    final_tickers = [t for t in valid_tickers if t in prices.columns]
    if len(final_tickers) != len(valid_tickers):
        missing_after_fetch = set(valid_tickers) - set(final_tickers)
        print(f"DEBUG: fetch_historical_data: Data incomplete for: {missing_after_fetch}. Filtering.") # Debug print
        st.warning(f"⚠️ Data incomplete for: **{', '.join(missing_after_fetch)}** within the specified date range. Proceeding with available data.")

    if not final_tickers:
        print("DEBUG: fetch_historical_data: No tickers with complete data after filtering.") # Debug print
        st.error("❌ No tickers with complete data in the specified range. Exiting simulation.")
        st.stop()

    prices = prices[final_tickers] # Ensure only valid and fetched tickers are kept
    prices = prices.ffill().dropna() # Forward fill and then drop any remaining NaNs
    print(f"DEBUG: fetch_historical_data: Prices DataFrame cleaned. Shape: {prices.shape}") # Debug print

    if prices.empty or len(prices) < 2:
        print("DEBUG: fetch_historical_data: Insufficient valid price data after cleaning.") # Debug print
        st.error("❌ Insufficient valid price data after cleaning (less than 2 data points). Adjust date range or tickers.")
        st.stop()
    return prices, final_tickers

@st.cache_data(ttl=3600) # Cache heavy calculations for 1 hour
def perform_optimizations(prices, risk_free_rate, num_portfolios_mc, min_weight_constraint, max_weight_constraint):
    """
    Performs Monte Carlo simulation and SciPy optimizations. Caches the results.
    Includes global min/max weight constraints for SciPy optimization.
    """
    print("DEBUG: perform_optimizations: Starting calculations.") # Debug print
    returns = prices.pct_change().dropna()

    if returns.empty:
        print("DEBUG: perform_optimizations: Returns DataFrame is empty.") # Debug print
        st.error("❌ Not enough valid data points for returns after calculating daily changes. Check date range.")
        st.stop()

    # Annualize returns, standard deviations, and covariance matrix
    annualization_factor = 252 # For daily data, 252 trading days
    expected_returns = returns.mean() * annualization_factor
    std_devs = returns.std() * np.sqrt(annualization_factor)
    cov_matrix = returns.cov() * annualization_factor
    tickers = prices.columns.tolist() # Get tickers from the cleaned prices DataFrame
    num_assets = len(tickers)

    print(f"DEBUG: perform_optimizations: Calculated annualized stats for {num_assets} assets.") # Debug print

    # Handle single asset case for correlation
    if num_assets == 1:
        correlation_matrix = np.array([[1.0]])
        print("DEBUG: perform_optimizations: Single asset detected, correlation matrix set to 1.0.") # Debug print
    else:
        correlation_matrix = returns.corr().values

    # --- Monte Carlo Simulation ---
    print(f"DEBUG: perform_optimizations: Starting Monte Carlo simulation for {num_portfolios_mc} portfolios.") # Debug print
    portfolio_returns_mc = []
    portfolio_risks_mc = []
    portfolio_weights_mc = []
    sharpe_ratios_mc = []

    # Streamlit progress bar and status text for user feedback
    progress_bar_mc = st.progress(0)
    status_text_mc = st.empty()

    for i in range(num_portfolios_mc):
        weights = random_weights(num_assets)
        # Note: Monte Carlo does not apply min/max weight constraints here,
        # as it's a random sampling. Constraints are applied in SciPy.
        ret = portfolio_return(weights, expected_returns)
        risk = portfolio_risk(weights, std_devs, correlation_matrix)
        sr = sharpe_ratio(ret, risk, risk_free_rate)

        portfolio_returns_mc.append(ret)
        portfolio_risks_mc.append(risk)
        portfolio_weights_mc.append(weights)
        sharpe_ratios_mc.append(sr)

        # Update progress bar and text every 1000 portfolios or more frequently for smaller numbers
        if num_portfolios_mc <= 1000 or (i + 1) % 1000 == 0:
            progress_bar_mc.progress((i + 1) / num_portfolios_mc)
            status_text_mc.text(f"Simulating Monte Carlo portfolios... {i+1}/{num_portfolios_mc} generated.")

    progress_bar_mc.empty()
    status_text_mc.empty()
    print("DEBUG: perform_optimizations: Monte Carlo simulation complete.") # Debug print

    # Find approximate Monte Carlo optimal points
    # Handle empty lists in case num_portfolios_mc was 0
    if portfolio_risks_mc:
        min_index_mc = portfolio_risks_mc.index(min(portfolio_risks_mc))
        min_risk_mc = portfolio_risks_mc[min_index_mc]
        min_return_mc = portfolio_returns_mc[min_index_mc]
        min_weights_mc = portfolio_weights_mc[min_index_mc]
        min_sharpe_mc = sharpe_ratio(min_return_mc, min_risk_mc, risk_free_rate)

        optimal_index_mc = sharpe_ratios_mc.index(max(sharpe_ratios_mc))
        optimal_risk_mc = portfolio_risks_mc[optimal_index_mc]
        optimal_return_mc = portfolio_returns_mc[optimal_index_mc]
        optimal_weights_mc = portfolio_weights_mc[optimal_index_mc]
        max_sharpe_ratio_mc = sharpe_ratios_mc[optimal_index_mc]
        print("DEBUG: perform_optimizations: Identified Monte Carlo optimal portfolios.") # Debug print
    else: # Default values if no MC portfolios were generated
        min_risk_mc, min_return_mc, min_weights_mc, min_sharpe_mc = 0, 0, np.array([]), 0
        optimal_risk_mc, optimal_return_mc, optimal_weights_mc, max_sharpe_ratio_mc = 0, 0, np.array([]), 0


    # --- SciPy Optimization ---
    # Initialize variables for SciPy results, in case num_assets is 0 or 1 or constraints are infeasible
    optimal_sharpe_weights_scipy = np.array([])
    optimal_sharpe_return_scipy = 0.0
    optimal_sharpe_risk_scipy = 0.0
    max_sharpe_ratio_scipy = 0.0
    min_variance_weights_scipy = np.array([])
    min_variance_return_scipy = 0.0
    min_variance_risk_scipy = 0.0
    min_variance_sharpe_scipy = 0.0

    if num_assets > 0: # SciPy optimization requires at least one asset
        # Define bounds for SciPy optimization based on user input
        bounds = tuple((min_weight_constraint, max_weight_constraint) for asset in range(num_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights must sum to 1
        initial_weights = num_assets * [1./num_assets,] # Still good initial guess

        # Check for initial feasibility based on constraints
        # This is a basic check; SciPy's minimizer itself will handle more complex infeasibility
        total_min_weight = num_assets * min_weight_constraint
        total_max_weight = num_assets * max_weight_constraint

        if total_min_weight > 1.0 + 1e-6 or total_max_weight < 1.0 - 1e-6:
            st.warning("⚠️ Constraints might make the optimization problem infeasible. SciPy optimization results may not be reliable or available.")
            # Skip SciPy optimization if basic feasibility check fails
        elif num_assets == 1:
            # Special handling for single asset in SciPy optimization
            if min_weight_constraint <= 1.0 <= max_weight_constraint:
                optimal_sharpe_weights_scipy = np.array([1.0])
                optimal_sharpe_return_scipy = expected_returns.iloc[0]
                optimal_sharpe_risk_scipy = std_devs.iloc[0]
                max_sharpe_ratio_scipy = sharpe_ratio(optimal_sharpe_return_scipy, optimal_sharpe_risk_scipy, risk_free_rate)

                min_variance_weights_scipy = np.array([1.0])
                min_variance_return_scipy = expected_returns.iloc[0]
                min_variance_risk_scipy = std_devs.iloc[0]
                min_variance_sharpe_scipy = sharpe_ratio(min_variance_return_scipy, min_variance_risk_scipy, risk_free_rate)
            else:
                st.warning("⚠️ For a single asset, the weight must be 100%. Your min/max constraints make this impossible. SciPy results will be zero/empty.")
        else: # num_assets > 1, perform full SciPy optimization
            with st.spinner("Calculating Optimal Risky Portfolio (SciPy)..."):
                try:
                    optimal_sharpe_results = sco.minimize(
                        neg_sharpe_ratio, initial_weights,
                        args=(expected_returns, cov_matrix, risk_free_rate),
                        method='SLSQP', bounds=bounds, constraints=constraints
                    )
                    if optimal_sharpe_results.success:
                        optimal_sharpe_weights_scipy = optimal_sharpe_results.x
                        optimal_sharpe_return_scipy = portfolio_return_scipy(optimal_sharpe_weights_scipy, expected_returns)
                        optimal_sharpe_risk_scipy = portfolio_risk_scipy(optimal_sharpe_weights_scipy, cov_matrix)
                        max_sharpe_ratio_scipy = sharpe_ratio(optimal_sharpe_return_scipy, optimal_sharpe_risk_scipy, risk_free_rate)
                        print("DEBUG: perform_optimizations: SciPy Max Sharpe optimization complete and successful.")
                    else:
                        st.warning(f"⚠️ SciPy Max Sharpe optimization failed: {optimal_sharpe_results.message}")
                        print(f"DEBUG: SciPy Max Sharpe optimization failed: {optimal_sharpe_results.message}")
                except Exception as e:
                    st.error(f"❌ Error during SciPy Max Sharpe optimization: {e}")
                    print(f"ERROR: SciPy Max Sharpe optimization: {e}")


            with st.spinner("Calculating Minimum Variance Portfolio (SciPy)..."):
                try:
                    min_variance_results = sco.minimize(
                        get_portfolio_volatility, initial_weights,
                        args=(expected_returns, cov_matrix),
                        method='SLSQP', bounds=bounds, constraints=constraints
                    )
                    if min_variance_results.success:
                        min_variance_weights_scipy = min_variance_results.x
                        min_variance_return_scipy = portfolio_return_scipy(min_variance_weights_scipy, expected_returns)
                        min_variance_risk_scipy = portfolio_risk_scipy(min_variance_weights_scipy, cov_matrix)
                        min_variance_sharpe_scipy = sharpe_ratio(min_variance_return_scipy, min_variance_risk_scipy, risk_free_rate)
                        print("DEBUG: perform_optimizations: SciPy Min Variance optimization complete and successful.")
                    else:
                        st.warning(f"⚠️ SciPy Min Variance optimization failed: {min_variance_results.message}")
                        print(f"DEBUG: SciPy Min Variance optimization failed: {min_variance_results.message}")
                except Exception as e:
                    st.error(f"❌ Error during SciPy Min Variance optimization: {e}")
                    print(f"ERROR: SciPy Min Variance optimization: {e}")
    else: # num_assets == 0 case
        print("DEBUG: perform_optimizations: No assets for SciPy optimization.") # Debug print


    print("DEBUG: perform_optimizations: All optimizations complete, returning results.") # Debug print
    return {
        "tickers": tickers,
        "expected_returns": expected_returns,
        "std_devs": std_devs,
        "correlation_matrix": correlation_matrix,
        "cov_matrix": cov_matrix, # Return covariance matrix for manual calculation later
        "num_assets": num_assets,
        "portfolio_returns_mc": portfolio_returns_mc,
        "portfolio_risks_mc": portfolio_risks_mc,
        "portfolio_weights_mc": portfolio_weights_mc,
        "sharpe_ratios_mc": sharpe_ratios_mc,
        "min_risk_mc": min_risk_mc,
        "min_return_mc": min_return_mc,
        "min_weights_mc": min_weights_mc,
        "min_sharpe_mc": min_sharpe_mc,
        "optimal_risk_mc": optimal_risk_mc,
        "optimal_return_mc": optimal_return_mc,
        "optimal_weights_mc": optimal_weights_mc,
        "max_sharpe_ratio_mc": max_sharpe_ratio_mc,
        "optimal_sharpe_weights_scipy": optimal_sharpe_weights_scipy,
        "optimal_sharpe_return_scipy": optimal_sharpe_return_scipy,
        "optimal_sharpe_risk_scipy": optimal_sharpe_risk_scipy,
        "max_sharpe_ratio_scipy": max_sharpe_ratio_scipy,
        "min_variance_weights_scipy": min_variance_weights_scipy,
        "min_variance_return_scipy": min_variance_return_scipy,
        "min_variance_risk_scipy": min_variance_risk_scipy,
        "min_variance_sharpe_scipy": min_variance_sharpe_scipy,
        "returns_df": returns # Add this line to return the daily returns DataFrame
    }


# ==============================================================================
# 🎛️ Streamlit Input Widgets (User Interface Section)
# This section defines the input fields and controls for the user.
# ==============================================================================

st.subheader("Configuration") # Subheader for the input section

# Expander for "How to Use" instructions, collapsed by default
with st.expander("❓ How to Use This Tool", expanded=False):
    st.markdown("""
    This simulator helps you explore the Efficient Frontier using Markowitz Portfolio Theory.

    1.  **Enter Ticker Symbols**: Provide comma-separated stock or ETF symbols (e.g., `AAPL,MSFT,GOOG`).
    2.  **Select Date Range**: Choose the historical period for data analysis. Data will be fetched for this period.
    3.  **Risk-Free Rate**: Input the annual risk-free rate (e.g., `0.04` for 4%). This is crucial for Sharpe Ratio calculation.
    4.  **Number of Portfolios**: Adjust the slider for the Monte Carlo simulation. More portfolios generate a denser and visually smoother efficient frontier, but take longer to compute.
    5.  **Portfolio Constraints**: Define minimum/maximum weights for assets and choose whether to allow short sales.
    6.  **Run Simulation**: Click the button to initiate data fetching, perform Monte Carlo simulations, and run precise SciPy optimizations, then visualize the results.

    The tool will calculate and display:
    * **Asset Statistics**: Annualized expected returns, volatility (risk), and the correlation matrix for your chosen assets.
    * **Efficient Frontier Plot**: A graphical representation of portfolios based on their risk and return, showing the trade-off. Hover over points for details! This plot includes:
        * The **Risk-Free Asset point**.
        * **Individual Asset points** for context (hover for details).
        * **Download Plot:** A button to download the plot as a PNG image.
    * **Optimal Portfolios (SciPy Optimization)**:
        * **Optimal Risky Portfolio**: The portfolio with the highest Sharpe Ratio (best risk-adjusted return), precisely found using numerical optimization.
        * **Minimum Variance Portfolio**: The portfolio with the lowest risk for a given return, precisely found using numerical optimization.
    * **Portfolio Risk Measures (VaR & CVaR)**: Downside risk metrics for the optimal portfolios.
    * **Monte Carlo Portfolio Details**: Provides the approximate optimal portfolios found via random sampling for comparison.

    *Disclaimer: This tool is for educational purposes only and not financial advice. Past performance is not indicative of future results and investing involves risk.*
    """)

# Input fields organized using columns for better layout and space utilization
col1, col2 = st.columns([2, 1]) # Create two columns, col1 takes 2 parts, col2 takes 1 part
with col1:
    tickers_input = st.text_input(
        'Enter Ticker Symbols (comma-separated):', # Label for the text input
        'AAPL,MSFT,GOOG',                          # Default value
        help="Example: AAPL,MSFT,GOOG for Apple, Microsoft, Google" # Help text on hover
    )
with col2:
    risk_free_rate_input = st.number_input(
        'Risk-Free Rate (annual %):', # Label for the number input
        min_value=0.0,                # Minimum allowed value
        max_value=0.2,                # Maximum allowed value (20%)
        value=0.04,                   # Default value (4%)
        step=0.001,                   # Step size for increment/decrement buttons
        format="%.3f",                # Display format for the number (3 decimal places)
        help="Enter as a decimal, e.g., 0.04 for 4%" # Help text
    )

col3, col4 = st.columns(2) # Create two more columns for date inputs
with col3:
    start_date_value = st.date_input(
        'Start Date:',                   # Label
        date(2018, 1, 1),                # Default start date
        help="Select the start date for historical data" # Help text
    )
with col4:
    end_date_value = st.date_input(
        'End Date:',                     # Label
        date(2024, 12, 31),              # Default end date
        help="Select the end date for historical data" # Help text
    )

# Slider for controlling the number of Monte Carlo portfolios
num_portfolios_value = st.slider(
    'Number of Portfolios for Monte Carlo Simulation:', # Label
    min_value=1000,                                   # Minimum number of portfolios
    max_value=50000,                                  # Increased max for better visual density
    value=10000,                                      # Default value
    step=1000,                                        # Step size for slider
    help="More portfolios mean a more comprehensive visual representation of the efficient frontier, but take longer." # Help text
)

# New section for weight constraints
st.subheader("Portfolio Constraints")
st.write("Define allocation percentages for **each individual asset** in the SciPy optimized portfolios.")

# New checkbox for short sales
allow_short_sales = st.checkbox(
    "Allow Short Sales (Negative Weights)",
    value=False, # Default: Short sales disallowed
    help="If checked, asset weights can be negative, allowing for short positions. This implies higher risk and potential for leverage."
)

# Adjust min_value for global_min_weight based on allow_short_sales
min_weight_floor = -2.0 if allow_short_sales else 0.0 # Allow up to -200% shorting
default_min_weight_value = -0.5 if allow_short_sales else 0.0 # Sensible default if shorting is on

# Adjust max_value for global_max_weight based on allow_short_sales (to allow leverage)
max_weight_ceiling = 2.0 if allow_short_sales else 1.0 # Allow up to 200% single asset allocation if shorting is on
default_max_weight_value = 1.0

col5, col6 = st.columns(2)
with col5:
    global_min_weight = st.number_input(
        'Minimum Weight per Asset:',
        min_value=min_weight_floor, # Adjusted based on short sales toggle
        max_value=1.0, # Max can still be 1.0 (100%)
        value=default_min_weight_value, # Adjusted default
        step=0.01,
        format="%.2f",
        help="Set a minimum allocation percentage for each asset. Negative values allow short positions."
    )
with col6:
    global_max_weight = st.number_input(
        'Maximum Weight per Asset:',
        min_value=0.0,
        max_value=max_weight_ceiling, # Adjusted if short sales allowed to accommodate leverage
        value=default_max_weight_value,
        step=0.01,
        format="%.2f",
        help="Set a maximum allocation percentage for each asset."
    )

# Initialize session state variables at the top of the script
# These variables persist across reruns and are crucial for managing app state
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'plotly_fig' not in st.session_state:
    st.session_state.plotly_fig = None
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False


# Run Simulation Button - The primary trigger for all calculations and displays
run_button = st.button("📈 Run Portfolio Optimization", help="Click to start the Monte Carlo simulation and SciPy optimization")

# ==============================================================================
# 🚀 Main Simulation Logic (Executes when the "Run" button is clicked)
# This block performs calculations and updates session state.
# ==============================================================================
if run_button:
    st.session_state.simulation_run = True # Set flag that simulation has run
    print("\nDEBUG: Step 1: Run button clicked. Starting main execution block (storing results).") # Debug print
    results_container = st.empty() # Placeholder for dynamic messages (info, error, success)

    with st.spinner("Starting simulation and fetching data..."):
        print("DEBUG: Step 2: Spinner activated.") # Debug print

        # 1. Input Processing and Initial Validation
        input_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        print(f"DEBUG: Step 3: Input tickers parsed: {input_tickers}") # Debug print

        if not input_tickers:
            print("DEBUG: Step 3a: No tickers entered. Showing error.") # Debug print
            results_container.error("❌ Please enter valid ticker symbols.")
            st.stop()

        # Basic validation for constraints
        if global_min_weight > global_max_weight:
            results_container.error("❌ Minimum weight per asset cannot be greater than maximum weight per asset.")
            st.stop()

        start_date_str = start_date_value.isoformat()
        end_date_str = end_date_value.isoformat()
        print(f"DEBUG: Step 3b: Date range selected: {start_date_str} to {end_date_str}") # Debug print

        if start_date_value >= end_date_value:
            print("DEBUG: Step 3c: Invalid date range. Showing error.") # Debug print
            results_container.error("❌ Start Date must be before End Date.")
            st.stop()

        # Call cached ticker validation function
        results_container.info("🔎 Validating ticker symbols...")
        print("DEBUG: Step 4: Calling validate_and_fetch_tickers().") # Debug print
        valid_tickers, invalid_tickers = validate_and_fetch_tickers(input_tickers)
        print(f"DEBUG: Step 4a: Ticker validation complete. Valid: {valid_tickers}, Invalid: {invalid_tickers}") # Debug print

        if invalid_tickers:
            st.warning(f"⚠️ The following tickers were not found or could not be fetched: **{', '.join(invalid_tickers)}**. Proceeding with valid tickers only.")

        if not valid_tickers:
            print("DEBUG: Step 4b: No valid ticker symbols found after validation. Simulation cannot proceed.") # Debug print
            results_container.error("❌ No valid ticker symbols found after validation. Simulation cannot proceed.")
            st.stop()

        # Check for constraint feasibility with actual number of assets (more robust check with negative weights possible)
        num_assets_after_validation = len(valid_tickers)
        if num_assets_after_validation > 0:
            # Check if it's possible for weights to sum to 1 given min/max constraints
            # Sum of minimums should not exceed 1 (with a small tolerance)
            if num_assets_after_validation * global_min_weight > 1.0 + 1e-6:
                results_container.error(f"❌ Infeasible constraints: With {num_assets_after_validation} assets, a minimum weight of {global_min_weight*100:.1f}% per asset sums to over 100%. Reduce min weight or remove assets.")
                st.stop()
            # Sum of maximums should not be less than 1 (with a small tolerance)
            if num_assets_after_validation * global_max_weight < 1.0 - 1e-6:
                results_container.error(f"❌ Infeasible constraints: With {num_assets_after_validation} assets, a maximum weight of {global_max_weight*100:.1f}% per asset sums to less than 100%. Increase max weight or remove assets.")
                st.stop()


        # Call cached historical data fetch function
        results_container.info("⏳ Fetching historical price data for valid tickers...")
        print("DEBUG: Step 5: Calling fetch_historical_data().") # Debug print
        prices, final_valid_tickers = fetch_historical_data(valid_tickers, start_date_str, end_date_str)
        print(f"DEBUG: Step 5a: Historical data fetch complete. Final valid tickers: {final_valid_tickers}") # Debug print

        # Call cached optimization function, passing constraints
        results_container.info("⚙️ Performing Monte Carlo simulation and SciPy optimization...")
        print("DEBUG: Step 6: Calling perform_optimizations().") # Debug print

        # Perform optimizations and store results directly in session state
        st.session_state.optimization_results = perform_optimizations(
            prices, risk_free_rate_input, num_portfolios_value, global_min_weight, global_max_weight
        )
        print("DEBUG: Step 6a: Optimizations complete, results stored in session_state.")

        # Rebuild the Plotly figure from scratch when run_button is clicked (fresh simulation)
        print("DEBUG: Rebuilding Plotly figure after new optimization.")
        # Unpack necessary results for fig creation from session state
        tickers = st.session_state.optimization_results["tickers"]
        expected_returns = st.session_state.optimization_results["expected_returns"]
        std_devs = st.session_state.optimization_results["std_devs"]
        portfolio_returns_mc = st.session_state.optimization_results["portfolio_returns_mc"]
        portfolio_risks_mc = st.session_state.optimization_results["portfolio_risks_mc"]
        sharpe_ratios_mc = st.session_state.optimization_results["sharpe_ratios_mc"]
        portfolio_weights_mc = st.session_state.optimization_results["portfolio_weights_mc"]
        min_variance_risk_scipy = st.session_state.optimization_results["min_variance_risk_scipy"]
        min_variance_return_scipy = st.session_state.optimization_results["min_variance_return_scipy"]
        min_variance_sharpe_scipy = st.session_state.optimization_results["min_variance_sharpe_scipy"]
        min_variance_weights_scipy = st.session_state.optimization_results["min_variance_weights_scipy"]
        optimal_sharpe_risk_scipy = st.session_state.optimization_results["optimal_sharpe_risk_scipy"]
        optimal_sharpe_return_scipy = st.session_state.optimization_results["optimal_sharpe_return_scipy"]
        max_sharpe_ratio_scipy = st.session_state.optimization_results["max_sharpe_ratio_scipy"]
        optimal_sharpe_weights_scipy = st.session_state.optimization_results["optimal_sharpe_weights_scipy"]
        num_assets = st.session_state.optimization_results["num_assets"]


        # Create a DataFrame for Monte Carlo portfolios for Plotly
        mc_portfolios_df = pd.DataFrame({
            'Volatility (%)': np.array(portfolio_risks_mc) * 100,
            'Expected Return (%)': np.array(portfolio_returns_mc) * 100,
            'Sharpe Ratio': np.array(sharpe_ratios_mc)
        })

        # Add weights to hover text for MC portfolios
        hover_text_mc = []
        for i, weights in enumerate(portfolio_weights_mc):
            weights_str = '<br>'.join([f'{ticker}: {w*100:.2f}%' for ticker, w in zip(tickers, weights)])
            hover_text_mc.append(f'Return: {mc_portfolios_df.iloc[i]["Expected Return (%)"]:.2f}%<br>'
                                 f'Volatility: {mc_portfolios_df.iloc[i]["Volatility (%)"]:.2f}%<br>'
                                 f'Sharpe Ratio: {mc_portfolios_df.iloc[i]["Sharpe Ratio"]:.2f}<br>'
                                 f'--- Weights ---<br>{weights_str}')

        mc_portfolios_df['Hover Text'] = hover_text_mc

        # Build the Plotly figure
        fig = go.Figure()

        # Add Monte Carlo portfolios scatter plot
        fig.add_trace(go.Scatter(
            x=mc_portfolios_df['Volatility (%)'],
            y=mc_portfolios_df['Expected Return (%)'],
            mode='markers',
            marker=dict(
                size=5,
                color=mc_portfolios_df['Sharpe Ratio'], # Color by Sharpe Ratio
                colorbar=dict(title='Sharpe Ratio'),    # This defines the color bar
                colorscale='Viridis', # Color scale for Sharpe Ratio
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name='Monte Carlo Portfolios (Color by Sharpe Ratio)', # More descriptive legend name
            hoverinfo='text',
            hovertext=mc_portfolios_df['Hover Text'],
            showlegend=True, # Explicitly ensure it appears in legend
            legendgroup='mc_portfolios' # Assign a legend group for clearer interaction
        ))

        # Prepare data for axis ranges
        all_returns = []
        all_risks = []

        if num_assets > 0:
            # Add SciPy Minimum Variance Portfolio
            if min_variance_weights_scipy.size > 0:
                fig.add_trace(go.Scatter(
                    x=[min_variance_risk_scipy * 100],
                    y=[min_variance_return_scipy * 100],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star', line=dict(width=1, color='Black')),
                    name='SciPy Minimum Variance Portfolio',
                    hoverinfo='text',
                    hovertext=[
                        f'Return: {min_variance_return_scipy*100:.2f}%<br>'
                        f'Volatility: {min_variance_risk_scipy*100:.2f}%<br>'
                        f'Sharpe Ratio: {min_variance_sharpe_scipy:.2f}<br>'
                        f'--- Weights ---<br>' + '<br>'.join([f'{t}: {w*100:.2f}%' for t, w in zip(tickers, min_variance_weights_scipy)])
                    ],
                    showlegend=True,
                    legendgroup='scipy_min_var'
                ))
                all_returns.append(min_variance_return_scipy * 100)
                all_risks.append(min_variance_risk_scipy * 100)

            # Add SciPy Optimal Risky Portfolio (Max Sharpe)
            if optimal_sharpe_weights_scipy.size > 0:
                fig.add_trace(go.Scatter(
                    x=[optimal_sharpe_risk_scipy * 100],
                    y=[optimal_sharpe_return_scipy * 100],
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='star', line=dict(width=1, color='Black')),
                    name='SciPy Optimal Risky Portfolio',
                    hoverinfo='text',
                    hovertext=[
                        f'Return: {optimal_sharpe_return_scipy*100:.2f}%<br>'
                        f'Volatility: {optimal_sharpe_risk_scipy*100:.2f}%<br>'
                        f'Sharpe Ratio: {max_sharpe_ratio_scipy:.2f}<br>'
                        f'--- Weights ---<br>' + '<br>'.join([f'{t}: {w*100:.2f}%' for t, w in zip(tickers, optimal_sharpe_weights_scipy)])
                    ],
                    showlegend=True,
                    legendgroup='scipy_max_sharpe'
                ))
                all_returns.append(optimal_sharpe_return_scipy * 100)
                all_risks.append(optimal_sharpe_risk_scipy * 100)

            # Individual Assets to the plot for context (no text labels, rely on hover)
            if tickers and len(tickers) > 0 and not expected_returns.empty and not std_devs.empty:
                individual_asset_hover_text = [
                    f'Asset: {t}<br>Return: {expected_returns[t]*100:.2f}%<br>Volatility: {std_devs[t]*100:.2f}%'
                    for t in tickers
                ]
                fig.add_trace(go.Scatter(
                    x=std_devs * 100,
                    y=expected_returns * 100,
                    mode='markers',
                    marker=dict(size=8, color='gray', symbol='circle', line=dict(width=1, color='DarkSlateGrey')),
                    name='Individual Assets',
                    hoverinfo='text',
                    hovertext=individual_asset_hover_text,
                    showlegend=True,
                    legendgroup='individual_assets'
                ))
                all_returns.extend((expected_returns * 100).tolist())
                all_risks.extend((std_devs * 100).tolist())

        # Collect all Monte Carlo returns and risks for axis scaling
        all_returns.extend(mc_portfolios_df['Expected Return (%)'].tolist())
        all_risks.extend(mc_portfolios_df['Volatility (%)'].tolist())

        # Calculate dynamic axis ranges with a buffer
        if all_returns and all_risks: # Ensure lists are not empty
            min_return_val = min(all_returns)
            max_return_val = max(all_returns)
            min_risk_val = min(all_risks)
            max_risk_val = max(all_risks)

            # Add a 5% buffer to the ranges
            y_range_buffer = (max_return_val - min_return_val) * 0.05
            x_range_buffer = (max_risk_val - min_risk_val) * 0.05

            y_axis_range = [min_return_val - y_range_buffer, max_return_val + y_range_buffer]
            x_axis_range = [min_risk_val - x_range_buffer, max_risk_val + x_range_buffer]
            # Ensure X-axis starts at or near zero for risk
            if x_axis_range[0] > -0.5: # Allow slight negative for very small range or if shorting allows
                x_axis_range[0] = -0.5 # Clamp at -0.5 (or slightly below if needed for specific cases)

            # Ensure Y-axis minimum is not too high if risk-free rate is very low or negative
            # This ensures the risk-free rate (if shown) is always in view
            if y_axis_range[0] > risk_free_rate_input * 100:
                y_axis_range[0] = risk_free_rate_input * 100 - y_range_buffer
        else: # Fallback to default ranges if no data points
            y_axis_range = [-10, 50]
            x_axis_range = [0, 50]


        # Update layout for Plotly figure
        fig.update_layout(
            title={
                'text': 'Efficient Frontier with Monte Carlo & SciPy Optimization',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(t=80, b=150, l=0, r=0), # Increased bottom margin for legend
            xaxis_title='Volatility (%)',
            yaxis_title='Expected Return (%)',
            hovermode='closest',
            height=600,
            template="plotly_white",
            legend=dict( # Position the legend BELOW the plot area
                orientation="h", # Horizontal legend for better fit below
                yanchor="top", # Anchor from the top of the legend box
                y=-0.2, # Position below the plot area (0 is bottom of plot, negative moves it further down)
                xanchor="center", # Center horizontally
                x=0.5,
                bgcolor='rgba(0,0,0,0)', # Transparent background
                bordercolor="rgba(0,0,0,0)", # Transparent border
                borderwidth=0
            ),
            xaxis=dict(range=x_axis_range), # Apply dynamic x-axis range
            yaxis=dict(range=y_axis_range)  # Apply dynamic y-axis range
        )
        st.session_state.plotly_fig = fig # Store the freshly built figure in session state

    results_container.empty() # Clear the info/spinner message
    st.success("✅ Simulation and Optimization completed! Scroll down to see results.")


# ==============================================================================
# 🚀 Main Display Logic (Executes if simulation has run at least once)
# This block displays results by retrieving data from session state.
# ==============================================================================
if st.session_state.simulation_run and st.session_state.optimization_results:
    print("DEBUG: Main display block active.") # Debug print
    # Unpack results from session state for display
    optimization_results = st.session_state.optimization_results
    tickers = optimization_results["tickers"]
    expected_returns = optimization_results["expected_returns"]
    std_devs = optimization_results["std_devs"]
    correlation_matrix = optimization_results["correlation_matrix"]
    cov_matrix = optimization_results["cov_matrix"]
    num_assets = optimization_results["num_assets"]
    portfolio_returns_mc = optimization_results["portfolio_returns_mc"]
    portfolio_risks_mc = optimization_results["portfolio_risks_mc"]
    portfolio_weights_mc = optimization_results["portfolio_weights_mc"]
    sharpe_ratios_mc = optimization_results["sharpe_ratios_mc"]
    min_risk_mc = optimization_results["min_risk_mc"]
    min_return_mc = optimization_results["min_return_mc"]
    min_weights_mc = optimization_results["min_weights_mc"]
    min_sharpe_mc = optimization_results["min_sharpe_mc"]
    optimal_risk_mc = optimization_results["optimal_risk_mc"]
    optimal_return_mc = optimization_results["optimal_return_mc"]
    optimal_weights_mc = optimization_results["optimal_weights_mc"]
    max_sharpe_ratio_mc = optimization_results["max_sharpe_ratio_mc"]
    optimal_sharpe_weights_scipy = optimization_results["optimal_sharpe_weights_scipy"]
    optimal_sharpe_return_scipy = optimization_results["optimal_sharpe_return_scipy"]
    optimal_sharpe_risk_scipy = optimization_results["optimal_sharpe_risk_scipy"]
    max_sharpe_ratio_scipy = optimization_results["max_sharpe_ratio_scipy"]
    min_variance_weights_scipy = optimization_results["min_variance_weights_scipy"]
    min_variance_return_scipy = optimization_results["min_variance_return_scipy"]
    min_variance_risk_scipy = optimization_results["min_variance_risk_scipy"]
    min_variance_sharpe_scipy = optimization_results["min_variance_sharpe_scipy"]
    returns_df = optimization_results["returns_df"]

    # =========================================================================
    # 2. Display Asset Statistics
    # =========================================================================
    st.subheader("Asset Statistics")
    st.write("📈 Expected Annual Returns (%):")
    st.dataframe((expected_returns * 100).round(2).rename("Return (%)"))
    st.write("📊 Annual Volatility (%):")
    st.dataframe((std_devs * 100).round(2).rename("Volatility (%)"))
    print("DEBUG: Step 8: Displayed Asset Statistics.") # Debug print

    if num_assets > 1:
        st.write("🔗 Correlation Matrix:")
        st.dataframe(pd.DataFrame(correlation_matrix, index=tickers, columns=tickers).round(2))
    elif num_assets == 1:
        st.info("Since you only provided one asset, the correlation matrix is simply 1.0 (an asset is perfectly correlated with itself).")

    st.write(f"Risk-Free Rate ($R_f$): {risk_free_rate_input*100:.2f}%")
    print("DEBUG: Step 9: Displayed Correlation Matrix and Risk-Free Rate.") # Debug print


    # =========================================================================
    # 3. SciPy Optimization Results
    # =========================================================================
    st.subheader("Optimal Portfolios (SciPy Optimization)")
    print("DEBUG: Displaying SciPy Optimization Results.") # Debug print

    if num_assets == 0:
        st.error("No assets available for optimization.")
    elif (num_assets == 1 and not (global_min_weight <= 1.0 <= global_max_weight)):
        st.warning("Skipping SciPy results for single asset due to infeasible constraints (100% not allowed by min/max).")
    else:
        if optimal_sharpe_weights_scipy.size > 0:
            st.write("🚀 **Optimal Risky Portfolio (Max Sharpe Ratio - SciPy):**")
            optimal_sharpe_df_scipy = pd.DataFrame({'Ticker': tickers, 'Weight (%)': (optimal_sharpe_weights_scipy * 100).round(2)})
            st.dataframe(optimal_sharpe_df_scipy)
            st.write(f"Expected Return: {optimal_sharpe_return_scipy*100:.2f}%")
            st.write(f"Volatility (Risk): {optimal_sharpe_risk_scipy*100:.2f}%")
            st.write(f"Sharpe Ratio: {max_sharpe_ratio_scipy:.2f}")
        else:
            st.warning("Optimal Risky Portfolio (SciPy) could not be found with current constraints or data.")

        if min_variance_weights_scipy.size > 0:
            st.write("\n🌟 **Minimum Variance Portfolio (SciPy):**")
            min_var_df_scipy = pd.DataFrame({'Ticker': tickers, 'Weight (%)': (min_variance_weights_scipy * 100).round(2)})
            st.dataframe(min_var_df_scipy)
            st.write(f"Expected Return: {min_variance_return_scipy*100:.2f}%")
            st.write(f"Volatility (Risk): {min_variance_risk_scipy*100:.2f}%")
            st.write(f"Sharpe Ratio: {min_variance_sharpe_scipy:.2f}")
        else:
            st.warning("Minimum Variance Portfolio (SciPy) could not be found with current constraints or data.")
    print("DEBUG: SciPy Optimization Results displayed.") # Debug print

    # =========================================================================
    # 4. Display Portfolio Risk Measures (VaR & CVaR)
    # =========================================================================
    st.subheader("Portfolio Risk Measures (VaR & CVaR)")
    print("DEBUG: Calculating and displaying VaR/CVaR.") # Debug print

    if num_assets == 0:
        st.error("No assets to calculate VaR/CVaR.")
    elif (num_assets == 1 and not (global_min_weight <= 1.0 <= global_max_weight)):
        st.warning("Skipping VaR/CVaR for single asset due to infeasible constraints.")
    else:
        # Calculate daily returns for the SciPy optimal portfolios
        # Ensure weights are aligned with returns_df columns
        # This handles cases where some tickers might have been dropped due to missing data
        if optimal_sharpe_weights_scipy.size > 0 and len(optimal_sharpe_weights_scipy) == len(returns_df.columns):
            portfolio_returns_sharpe = returns_df.dot(optimal_sharpe_weights_scipy)
            var_sharpe_95, cvar_sharpe_95 = calculate_historical_var_cvar(portfolio_returns_sharpe, confidence_level=0.95, annualize=True)
            var_sharpe_99, cvar_sharpe_99 = calculate_historical_var_cvar(portfolio_returns_sharpe, confidence_level=0.99, annualize=True)

            st.write("📈 **Optimal Risky Portfolio (Max Sharpe Ratio - SciPy):**")
            st.write(f"  Annualized 95% VaR: {var_sharpe_95*100:.2f}%")
            st.write(f"  Annualized 95% CVaR: {cvar_sharpe_95*100:.2f}%")
            st.write(f"  Annualized 99% VaR: {var_sharpe_99*100:.2f}%")
            st.write(f"  Annualized 99% CVaR: {cvar_sharpe_99*100:.2f}%")
        else:
            st.warning("Could not calculate VaR/CVaR for Max Sharpe Portfolio due to missing weights or data misalignment/infeasibility.")


        if min_variance_weights_scipy.size > 0 and len(min_variance_weights_scipy) == len(returns_df.columns):
            portfolio_returns_min_var = returns_df.dot(min_variance_weights_scipy)
            var_min_var_95, cvar_min_var_95 = calculate_historical_var_cvar(portfolio_returns_min_var, confidence_level=0.95, annualize=True)
            var_min_var_99, cvar_min_var_99 = calculate_historical_var_cvar(portfolio_returns_min_var, confidence_level=0.99, annualize=True)

            st.write("\n📊 **Minimum Variance Portfolio (SciPy):**")
            st.write(f"  Annualized 95% VaR: {var_min_var_95*100:.2f}%")
            st.write(f"  Annualized 95% CVaR: {cvar_min_var_95*100:.2f}%")
            st.write(f"  Annualized 99% VaR: {var_min_var_99*100:.2f}%")
            st.write(f"  Annualized 99% CVaR: {cvar_min_var_99*100:.2f}%")
        else:
             st.warning("Could not calculate VaR/CVaR for Min Variance Portfolio due to missing weights or data misalignment/infeasibility.")

    print("DEBUG: VaR/CVaR calculations and display complete.") # Debug print

    # =========================================================================
    # 5. Plot Efficient Frontier (Static Plotly)
    # =========================================================================
    st.subheader("Efficient Frontier Plot")
    print("DEBUG: Displaying Plotly chart from session_state.")

    fig_to_display = st.session_state.plotly_fig

    if fig_to_display:
        st.plotly_chart(fig_to_display, use_container_width=True) # Display the Plotly chart statically

        st.info("💡 **Tip for the chart:** Hover over points for detailed information. Click on a legend item to hide/show that series. Double-click an item to isolate it (hide all others).")

        st.download_button(
            label="Download Plot as PNG",
            data=fig_to_display.to_image(format="png"),
            file_name="efficient_frontier_plot.png",
            mime="image/png",
            help="Requires 'kaleido' library (pip install kaleido)"
        )

        st.markdown("<br>", unsafe_allow_html=True) # Add a line break for spacing
        st.write("---") # A horizontal line for clear separation

        print("DEBUG: Plotly chart displayed.")

    # =========================================================================
    # 6. Display Monte Carlo Portfolio Details (for comparison)
    # =========================================================================
    st.subheader("Monte Carlo Portfolio Details (Approximate)")
    st.write("🌟 **Monte Carlo Minimum Variance Portfolio Allocation:**")
    min_var_df_mc = pd.DataFrame({'Ticker': tickers, 'Weight (%)': (min_weights_mc * 100).round(2)})
    st.dataframe(min_var_df_mc)
    st.write(f"Expected Return: {min_return_mc*100:.2f}%")
    st.write(f"Volatility (Risk): {min_risk_mc*100:.2f}%")
    st.write(f"Sharpe Ratio: {min_sharpe_mc:.2f}")

    st.write("\n🚀 **Monte Carlo Optimal Risky Portfolio (Max Sharpe Ratio) Allocation:**")
    optimal_df_mc = pd.DataFrame({'Ticker': tickers, 'Weight (%)': (optimal_weights_mc * 100).round(2)})
    st.dataframe(optimal_df_mc)
    st.write(f"Expected Return: {optimal_return_mc*100:.2f}%")
    st.write(f"Volatility (Risk): {optimal_risk_mc*100:.2f}%")
    st.write(f"Sharpe Ratio: {max_sharpe_ratio_mc:.2f}")
    print("DEBUG: Monte Carlo Portfolio details displayed.") # Debug print

    # =========================================================================
    # 7. Download All Results Button
    # Provides combined results from both Monte Carlo and SciPy optimizations.
    # =========================================================================
    st.markdown("---") # Horizontal separator for visual grouping
    st.subheader("Download All Tabular Results")
    print("DEBUG: Preparing download button.") # Debug print

    csv_output = io.StringIO()
    csv_output.write("Optimal Risky Portfolio (SciPy) Allocation:\n")
    if num_assets > 0 and optimal_sharpe_weights_scipy.size > 0:
        optimal_sharpe_df_scipy.to_csv(csv_output, index=False, header=True)
    csv_output.write("\n\nMinimum Variance Portfolio (SciPy) Allocation:\n")
    if num_assets > 0 and min_variance_weights_scipy.size > 0:
        min_var_df_scipy.to_csv(csv_output, index=False, header=True)
    csv_output.write("\n\nMonte Carlo Optimal Risky Portfolio Allocation (Approximate):\n")
    optimal_df_mc.to_csv(csv_output, index=False, header=True)
    csv_output.write("\n\nMonte Carlo Minimum Variance Portfolio Allocation (Approximate):\n")
    min_var_df_mc.to_csv(csv_output, index=False, header=True)

    st.download_button(
        label="Download All Portfolio Data (CSV)",
        data=csv_output.getvalue(),
        file_name="optimized_portfolios_details.csv",
        mime="text/csv",
        help="Download the optimal and minimum variance portfolio allocations from both Monte Carlo and SciPy methods."
    )
    print("DEBUG: Download All Results button displayed. End of run_button block.") # Debug print

# ==============================================================================
# 8. Sidebar Information / Footer
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("About This App")
st.sidebar.info("Developed by Rafael Grilli Felizardo with AI tools. This app demonstrates Markowitz Portfolio Optimization, a fundamental concept in modern portfolio theory. For feedback, inquiries, or collaboration, please feel free to reach out!")
st.sidebar.markdown("© 2025 Rafael Grilli Felizardo - Portfolio Simulator. All rights reserved.")

