# streamlit_app.py

# ==============================================================================
# 📦 Import necessary libraries
# ==============================================================================
import streamlit as st        # Streamlit library for building interactive web apps
from yahooquery import Ticker # For fetching historical stock data from Yahoo Finance
import pandas as pd           # For data manipulation and analysis, especially DataFrames
import numpy as np            # For numerical operations, especially array manipulations
import plotly.graph_objects as go # Plotly for more customized graphs
from datetime import date     # For handling date inputs
import io                     # For handling in-memory binary streams (used for CSV download)
import scipy.optimize as sco  # SciPy's optimization module for finding precise optimal portfolios
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

        # --- NEW DEBUG PRINT: Raw price_data_check ---
        print(f"DEBUG: validate_and_fetch_tickers: Raw price_data_check received: {price_data_check}")
        # --- END NEW DEBUG PRINT ---

        for ticker_symbol in input_tickers_list:
            # --- NEW DEBUG PRINT: Checking individual ticker ---
            print(f"DEBUG: Checking ticker: {ticker_symbol}")
            # --- END NEW DEBUG PRINT ---

            if ticker_symbol in price_data_check and 'regularMarketPrice' in price_data_check[ticker_symbol]:
                tickers.append(ticker_symbol)
                # --- NEW DEBUG PRINT: Valid ticker ---
                print(f"DEBUG: {ticker_symbol} is VALID (found regularMarketPrice).")
                # --- END NEW DEBUG PRINT ---
            else:
                invalid_tickers.append(ticker_symbol)
                # --- NEW DEBUG PRINT: Invalid ticker details ---
                # More detailed invalid reason
                if ticker_symbol not in price_data_check:
                    print(f"DEBUG: {ticker_symbol} is INVALID (not found in price_data_check keys).")
                elif 'regularMarketPrice' not in price_data_check[ticker_symbol]:
                    print(f"DEBUG: {ticker_symbol} is INVALID (missing 'regularMarketPrice' for found ticker). Data for {ticker_symbol}: {price_data_check.get(ticker_symbol)}")
                else:
                    print(f"DEBUG: {ticker_symbol} is INVALID (unknown reason).")
                # --- END NEW DEBUG PRINT ---

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
        help="Example: AAPL,MSFT,GOOG for Apple, Microsoft, Google. For Brazilian stocks such as bbas3, insert .sa in the end" # Help text on hover
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
        'Start Date:',                # Label
        date(2018, 1, 1),             # Default start date
        help="Select the start date for historical data" # Help text
    )
with col4:
    end_date_value = st.date_input(
        'End Date:',                  # Label
        date(2024, 12, 31),           # Default end date
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

        if not input_tickers:
            st.error("❌ No ticker symbols entered. Please enter at least one ticker.")
            st.session_state.simulation_run = False
            results_container.empty()
            print("DEBUG: No ticker symbols entered. Stopping execution flow.")
            st.stop() # Stop further execution in this run if no tickers.

        valid_tickers, invalid_tickers = validate_and_fetch_tickers(input_tickers)

        if invalid_tickers:
            st.warning(f"⚠️ The following tickers could not be validated and will be ignored: **{', '.join(invalid_tickers)}**")

        if not valid_tickers:
            st.error("❌ No valid ticker symbols found after validation. Simulation cannot proceed.")
            st.session_state.simulation_run = False
            results_container.empty()
            print("DEBUG: No valid ticker symbols found after validation. Stopping execution flow.")
            st.stop() # Stop further execution in this run if no valid tickers.

        # 2. Fetch Historical Data
        start_date_str = start_date_value.strftime('%Y-%m-%d')
        end_date_str = end_date_value.strftime('%Y-%m-%d')

        # Use st.spinner for a better user experience during data fetching
        with st.spinner("Fetching historical data... This may take a moment."):
            prices_df, final_tickers = fetch_historical_data(valid_tickers, start_date_str, end_date_str)

        # 3. Perform Optimizations
        if prices_df is not None and not prices_df.empty and len(final_tickers) > 0:
            optimization_results = perform_optimizations(
                prices_df,
                risk_free_rate_input,
                num_portfolios_value,
                global_min_weight,
                global_max_weight
            )
            st.session_state.optimization_results = optimization_results
            st.success("✅ Portfolio optimization completed successfully!")
            print("DEBUG: Step 3: Optimization results stored in session state.") # Debug print
        else:
            st.error("❌ Could not proceed with optimization due to missing or insufficient data.")
            st.session_state.simulation_run = False
            print("DEBUG: Optimization skipped due to insufficient data.") # Debug print

    # Ensure results_container is cleared or updated after spinner
    results_container.empty() # Clear the "Starting simulation..." message if successful.


# ==============================================================================
# 📈 Display Results (Only if simulation has been run and results exist)
# This block displays the analysis and visualizations.
# ==============================================================================
if st.session_state.simulation_run and st.session_state.optimization_results:
    results = st.session_state.optimization_results
    tickers = results["tickers"]
    num_assets = results["num_assets"]

    st.subheader("Asset Statistics")
    if num_assets > 0:
        asset_stats_df = pd.DataFrame({
            'Expected Annual Return': results["expected_returns"],
            'Annual Volatility (Risk)': results["std_devs"]
        })
        st.dataframe(asset_stats_df.style.format({
            'Expected Annual Return': "{:.2%}",
            'Annual Volatility (Risk)': "{:.2%}"
        }))

        st.subheader("Correlation Matrix")
        # Ensure correlation_matrix is a DataFrame for display
        if num_assets > 0:
            correlation_df = pd.DataFrame(results["correlation_matrix"],
                                          index=tickers,
                                          columns=tickers)
            st.dataframe(correlation_df.style.format("{:.2f}"))
    else:
        st.info("No asset statistics or correlation matrix to display as no valid tickers were found.")


    st.subheader("Efficient Frontier Plot")

    # Only attempt to plot if there are Monte Carlo portfolios
    if results["portfolio_risks_mc"]:
        fig = go.Figure()

        # Scatter plot for Monte Carlo portfolios
        fig.add_trace(go.Scatter(
            x=results["portfolio_risks_mc"],
            y=results["portfolio_returns_mc"],
            mode='markers',
            marker=dict(
                size=5,
                color=results["sharpe_ratios_mc"], # Color points by Sharpe Ratio
                colorbar=dict(title='Sharpe Ratio'),
                colorscale='Viridis',
                line=dict(width=0.5, color='white')
            ),
            name='Monte Carlo Portfolios',
            text=[
                f"Return: {ret:.2%}<br>Risk: {risk:.2%}<br>Sharpe: {sr:.2f}<br>Weights: {', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, weights)])}"
                for ret, risk, sr, weights in zip(results["portfolio_returns_mc"], results["portfolio_risks_mc"], results["sharpe_ratios_mc"], results["portfolio_weights_mc"])
            ],
            hoverinfo='text'
        ))

        # Add individual assets
        if num_assets > 0: # Check again to avoid error if tickers somehow empty
            fig.add_trace(go.Scatter(
                x=results["std_devs"],
                y=results["expected_returns"],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='circle'),
                name='Individual Assets',
                text=[f'{ticker}' for ticker in tickers],
                textposition="top center",
                textfont=dict(size=10),
                texttemplate='%{text}', # Display ticker symbol
                hoverinfo='text',
                hovertext=[
                    f"Asset: {t}<br>Return: {r:.2%}<br>Risk: {s:.2%}"
                    for t, r, s in zip(tickers, results["expected_returns"], results["std_devs"])
                ]
            ))

        # Add Minimum Variance Portfolio (SciPy)
        if results["min_variance_risk_scipy"] > 0 and results["min_variance_return_scipy"] > 0: # Ensure non-zero values
            fig.add_trace(go.Scatter(
                x=[results["min_variance_risk_scipy"]],
                y=[results["min_variance_return_scipy"]],
                mode='markers',
                marker=dict(size=12, color='darkorange', symbol='star'),
                name='Min Variance Portfolio (SciPy)',
                text=f"Return: {results['min_variance_return_scipy']:.2%}<br>Risk: {results['min_variance_risk_scipy']:.2%}<br>Sharpe: {results['min_variance_sharpe_scipy']:.2f}<br>Weights: {', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, results['min_variance_weights_scipy'])])}",
                hoverinfo='text'
            ))

        # Add Optimal Risky Portfolio (Max Sharpe - SciPy)
        if results["optimal_sharpe_risk_scipy"] > 0 and results["optimal_sharpe_return_scipy"] > 0: # Ensure non-zero values
            fig.add_trace(go.Scatter(
                x=[results["optimal_sharpe_risk_scipy"]],
                y=[results["optimal_sharpe_return_scipy"]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Optimal Risky Portfolio (SciPy - Max Sharpe)',
                text=f"Return: {results['optimal_sharpe_return_scipy']:.2%}<br>Risk: {results['optimal_sharpe_risk_scipy']:.2%}<br>Sharpe: {results['max_sharpe_ratio_scipy']:.2f}<br>Weights: {', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, results['optimal_sharpe_weights_scipy'])])}",
                hoverinfo='text'
            ))

        # Add Capital Market Line (CML)
        # Assuming risk-free rate is annual
        cml_x = np.array([0, results["optimal_sharpe_risk_scipy"] + 0.05]) # Extend line slightly
        cml_y = risk_free_rate_input + (results["optimal_sharpe_return_scipy"] - risk_free_rate_input) / results["optimal_sharpe_risk_scipy"] * cml_x
        fig.add_trace(go.Scatter(
            x=cml_x, y=cml_y,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Capital Market Line'
        ))


        fig.update_layout(
            title='Efficient Frontier with Monte Carlo Simulation and Optimization',
            xaxis_title='Annual Volatility (Risk)',
            yaxis_title='Annual Return',
            hovermode='closest',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
            width=800, # Set a default width, Streamlit will adjust to column width
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.plotly_fig = fig # Store the figure for download

        # Download button for the plot
        if st.session_state.plotly_fig:
            buffer = io.BytesIO()
            st.session_state.plotly_fig.write_image(buffer, format="png")
            st.download_button(
                label="Download Plot as PNG",
                data=buffer,
                file_name="efficient_frontier.png",
                mime="image/png",
                key="download_plot_png"
            )
    else:
        st.warning("⚠️ No Monte Carlo portfolios were generated. Check your inputs (e.g., number of portfolios, valid tickers).")


    st.subheader("Optimal Portfolios (SciPy Optimization)")
    col_max_sharpe, col_min_var = st.columns(2)

    if results["optimal_sharpe_weights_scipy"].size > 0:
        with col_max_sharpe:
            st.write("**Optimal Risky Portfolio (Highest Sharpe Ratio)**")
            st.metric(label="Expected Annual Return", value=f"{results['optimal_sharpe_return_scipy']:.2%}")
            st.metric(label="Annual Volatility (Risk)", value=f"{results['optimal_sharpe_risk_scipy']:.2%}")
            st.metric(label="Sharpe Ratio", value=f"{results['max_sharpe_ratio_scipy']:.2f}")

            st.write("Portfolio Weights:")
            # Create a DataFrame for weights for better display and download
            weights_df_sharpe = pd.DataFrame({
                'Ticker': tickers,
                'Weight': results['optimal_sharpe_weights_scipy']
            }).set_index('Ticker')
            st.dataframe(weights_df_sharpe.style.format("{:.2%}"))

            csv_sharpe = weights_df_sharpe.to_csv().encode('utf-8')
            st.download_button(
                label="Download Optimal Sharpe Weights (CSV)",
                data=csv_sharpe,
                file_name="optimal_sharpe_weights.csv",
                mime="text/csv",
                key="download_sharpe_weights"
            )
    else:
        with col_max_sharpe:
            st.info("Optimal Risky Portfolio not found (likely due to invalid inputs or constraints).")


    if results["min_variance_weights_scipy"].size > 0:
        with col_min_var:
            st.write("**Minimum Variance Portfolio**")
            st.metric(label="Expected Annual Return", value=f"{results['min_variance_return_scipy']:.2%}")
            st.metric(label="Annual Volatility (Risk)", value=f"{results['min_variance_risk_scipy']:.2%}")
            st.metric(label="Sharpe Ratio", value=f"{results['min_variance_sharpe_scipy']:.2f}")

            st.write("Portfolio Weights:")
            weights_df_min_var = pd.DataFrame({
                'Ticker': tickers,
                'Weight': results['min_variance_weights_scipy']
            }).set_index('Ticker')
            st.dataframe(weights_df_min_var.style.format("{:.2%}"))

            csv_min_var = weights_df_min_var.to_csv().encode('utf-8')
            st.download_button(
                label="Download Minimum Variance Weights (CSV)",
                data=csv_min_var,
                file_name="min_variance_weights.csv",
                mime="text/csv",
                key="download_min_var_weights"
            )
    else:
        with col_min_var:
            st.info("Minimum Variance Portfolio not found (likely due to invalid inputs or constraints).")

    st.subheader("Portfolio Risk Measures (VaR & CVaR)")
    if results["returns_df"] is not None and not results["returns_df"].empty and results["optimal_sharpe_weights_scipy"].size > 0:
        # Calculate daily returns for the optimal Sharpe portfolio
        optimal_sharpe_portfolio_returns = results["returns_df"].dot(results["optimal_sharpe_weights_scipy"])
        var_sharpe, cvar_sharpe = calculate_historical_var_cvar(optimal_sharpe_portfolio_returns, confidence_level=0.95, annualize=True)

        st.markdown(f"**Optimal Risky Portfolio (95% Confidence)**")
        st.metric("Annualized Value-at-Risk (VaR)", f"{var_sharpe:.2%}")
        st.metric("Annualized Conditional VaR (CVaR)", f"{cvar_sharpe:.2%}")

    else:
        st.info("VaR and CVaR for Optimal Risky Portfolio not available (no valid portfolio or data).")


    st.subheader("Monte Carlo Portfolio Approximation (For Comparison)")
    if results["portfolio_risks_mc"]: # Check if MC results exist
        col_mc_sharpe, col_mc_min_risk = st.columns(2)

        with col_mc_sharpe:
            st.write("**Monte Carlo Highest Sharpe Ratio Portfolio**")
            st.metric(label="Expected Annual Return", value=f"{results['optimal_return_mc']:.2%}")
            st.metric(label="Annual Volatility (Risk)", value=f"{results['optimal_risk_mc']:.2%}")
            st.metric(label="Sharpe Ratio", value=f"{results['max_sharpe_ratio_mc']:.2f}")
            st.write("Weights:")
            mc_sharpe_weights_df = pd.DataFrame({
                'Ticker': tickers,
                'Weight': results['optimal_weights_mc']
            }).set_index('Ticker')
            st.dataframe(mc_sharpe_weights_df.style.format("{:.2%}"))

        with col_mc_min_risk:
            st.write("**Monte Carlo Minimum Risk Portfolio**")
            st.metric(label="Expected Annual Return", value=f"{results['min_return_mc']:.2%}")
            st.metric(label="Annual Volatility (Risk)", value=f"{results['min_risk_mc']:.2%}")
            st.metric(label="Sharpe Ratio", value=f"{results['min_sharpe_mc']:.2f}")
            st.write("Weights:")
            mc_min_risk_weights_df = pd.DataFrame({
                'Ticker': tickers,
                'Weight': results['min_weights_mc']
            }).set_index('Ticker')
            st.dataframe(mc_min_risk_weights_df.style.format("{:.2%}"))
    else:
        st.info("No Monte Carlo portfolio approximations to display.")

# ==============================================================================
# 8. Sidebar Information / Footer
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("About This App")
st.sidebar.info("Developed by Rafael Grilli Felizardo with AI tools. This app demonstrates Markowitz Portfolio Optimization, a fundamental concept in modern portfolio theory. For feedback, inquiries, or collaboration, please feel free to reach out!")
st.sidebar.markdown("© 2025 Rafael Grilli Felizardo - Portfolio Simulator. All rights reserved.")

