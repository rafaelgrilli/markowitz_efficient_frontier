#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit_app.py

# ========================
# 📦 Import necessary libraries
# ========================
import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import io
import base64 # Needed for download_button in some older Streamlit versions, though st.download_button is now preferred

# ========================
# ⚙️ Streamlit Page Configuration
# ========================
st.set_page_config(
    page_title="Portfolio Optimization",
    layout="wide", # Use wide layout for better visual space
    initial_sidebar_state="collapsed"
)

st.title("📊 Markowitz Portfolio Simulator")
st.write("⚠️ This tool is for educational purposes only and should not be considered financial advice.")

# ========================
# 🔢 Utility Functions
# ========================

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

# ========================
# 🎛️ Streamlit Input Widgets
# ========================

# Input fields organized for better layout
col1, col2 = st.columns([2, 1]) # Adjust column ratios if needed
with col1:
    tickers_input = st.text_input(
        'Enter Ticker Symbols (comma-separated):',
        'AAPL,MSFT,GOOG',
        help="Example: AAPL,MSFT,GOOG for Apple, Microsoft, Google"
    )
with col2:
    risk_free_rate_input = st.number_input(
        'Risk-Free Rate (annual %):',
        min_value=0.0,
        max_value=0.2,
        value=0.04,
        step=0.001,
        format="%.3f",
        help="Enter as a decimal, e.g., 0.04 for 4%"
    )

col3, col4, col5 = st.columns(3)
with col3:
    start_date_value = st.date_input(
        'Start Date:',
        date(2018, 1, 1),
        help="Select the start date for historical data"
    )
with col4:
    end_date_value = st.date_input(
        'End Date:',
        date(2024, 12, 31),
        help="Select the end date for historical data"
    )
with col5:
    num_portfolios_value = st.slider(
        'Number of Portfolios for Simulation:',
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000,
        help="More portfolios mean more precise results but take longer"
    )

# Run Simulation Button
if st.button("Run Portfolio Optimization", help="Click to start the Monte Carlo simulation"):
    # This block now contains the core logic of your run_simulation function
    # It will execute when the button is clicked.
    
    # Use st.empty() as a placeholder for dynamic messages/results
    results_container = st.empty()
    
    with st.spinner("Starting simulation..."): # Streamlit's way of showing progress
        
        input_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        if not input_tickers:
            results_container.error("❌ Please enter valid ticker symbols.")
            st.stop() # Stop execution
            
        # Convert date objects to string for yahooquery
        start_date_str = start_date_value.isoformat()
        end_date_str = end_date_value.isoformat()

        if start_date_value >= end_date_value:
            results_container.error("❌ Start Date must be before End Date.")
            st.stop()

        # =========================================================================
        # Ticker Validation
        # =========================================================================
        results_container.info("🔎 Validating ticker symbols...")
        
        tickers = []
        invalid_tickers = []

        batch_ticker_obj = Ticker(input_tickers)

        for ticker_symbol in input_tickers:
            try:
                price_data = batch_ticker_obj.price.get(ticker_symbol)

                if price_data and 'error' not in price_data and 'regularMarketPrice' in price_data:
                    tickers.append(ticker_symbol)
                else:
                    invalid_tickers.append(ticker_symbol)
            except Exception as e:
                invalid_tickers.append(ticker_symbol)

        if invalid_tickers:
            st.warning(f"⚠️ The following tickers were not found or could not be fetched: {', '.join(invalid_tickers)}. Proceeding with valid tickers only.")

        if not tickers:
            results_container.error("❌ No valid ticker symbols found. Simulation cannot proceed.")
            st.stop()
        # =========================================================================

        # ✅ Fetch historical price data using Yahoo Finance
        results_container.info("⏳ Fetching historical price data for valid tickers...")
        try:
            ticker_obj_for_history = Ticker(" ".join(tickers))
            df = ticker_obj_for_history.history(start=start_date_str, end=end_date_str)
        except Exception as e:
            results_container.error(f"❌ Error fetching historical data for valid tickers: {e}. Check internet connection or API limits.")
            st.stop()

        if df.empty:
            results_container.error("❌ No historical price data found for the valid tickers in the specified date range. Adjust inputs.")
            st.stop()

        prices = df.reset_index().pivot(index='date', columns='symbol', values='adjclose')

        valid_tickers_after_fetch = [t for t in tickers if t in prices.columns]
        if len(valid_tickers_after_fetch) != len(tickers):
            missing_after_fetch = set(tickers) - set(valid_tickers_after_fetch)
            st.warning(f"⚠️ Data incomplete for: {', '.join(missing_after_fetch)} within the specified date range. Proceeding with available data.")
            tickers = valid_tickers_after_fetch
            if not tickers:
                results_container.error("❌ No tickers with complete data in the specified range. Exiting simulation.")
                st.stop()
            prices = prices[tickers]

        prices = prices.ffill().dropna()

        if prices.empty or len(prices) < 2:
            results_container.error("❌ Insufficient valid price data after cleaning. Adjust date range or tickers.")
            st.stop()

        returns = prices.pct_change().dropna()

        if returns.empty:
            results_container.error("❌ Not enough valid data points for returns after calculating daily changes. Check date range.")
            st.stop()

        expected_returns = returns.mean() * 252
        std_devs = returns.std() * np.sqrt(252)

        num_assets = len(tickers)

        if num_assets == 1:
            st.info("\n⚠️ Only one asset provided. Correlation matrix is not applicable for a single asset.")
            correlation_matrix = np.array([[1.0]])
        else:
            correlation_matrix = returns.corr().values


        st.subheader("Asset Statistics")
        st.write("📈 Expected Annual Returns (%):")
        st.dataframe((expected_returns * 100).round(2).rename("Return (%)")) # Display as DataFrame for better formatting
        st.write("📊 Annual Volatility (%):")
        st.dataframe((std_devs * 100).round(2).rename("Volatility (%)"))
        
        if num_assets > 1:
            st.write("🔗 Correlation Matrix:")
            st.dataframe(pd.DataFrame(correlation_matrix, index=tickers, columns=tickers).round(2))
        
        st.write(f"R_f (Risk-Free Rate): {risk_free_rate_input*100:.2f}%")


        # ========================
        # 🔁 Monte Carlo Simulation
        # ========================
        st.info(f"Running Monte Carlo simulation with {num_portfolios_value} portfolios...")
        portfolio_returns = []
        portfolio_risks = []
        portfolio_weights = []
        sharpe_ratios = []

        for _ in range(num_portfolios_value): # Use the value from the slider
            weights = random_weights(num_assets)
            ret = portfolio_return(weights, expected_returns)
            risk = portfolio_risk(weights, std_devs, correlation_matrix)
            sr = sharpe_ratio(ret, risk, risk_free_rate_input) # Use the value from the number_input

            portfolio_returns.append(ret)
            portfolio_risks.append(risk)
            portfolio_weights.append(weights)
            sharpe_ratios.append(sr)

        # ========================
        # 🎯 Find Minimum Variance Portfolio
        # ========================
        min_index = portfolio_risks.index(min(portfolio_risks))
        min_risk = portfolio_risks[min_index]
        min_return = portfolio_returns[min_index]
        min_weights = portfolio_weights[min_index]

        # ========================
        # 🌟 Find Optimal Risky Portfolio (Max Sharpe Ratio)
        # ========================
        optimal_index = sharpe_ratios.index(max(sharpe_ratios))
        optimal_risk = portfolio_risks[optimal_index]
        optimal_return = portfolio_returns[optimal_index]
        optimal_weights = portfolio_weights[optimal_index]
        max_sharpe_ratio = sharpe_ratios[optimal_index]


        # ========================
        # 📈 Plot Efficient Frontier
        # ========================
        st.subheader("Efficient Frontier")
        fig, ax = plt.subplots(figsize=(12, 7)) # Create a figure and axes object
        ax.scatter(np.array(portfolio_risks) * 100, np.array(portfolio_returns) * 100,
                   c=np.array(sharpe_ratios), cmap='viridis', label='Portfolios', s=10)
        cbar = fig.colorbar(ax.scatter(0, 0, c=[0], cmap='viridis'), ax=ax, label='Sharpe Ratio') # Dummy scatter for colorbar
        
        ax.scatter(min_risk * 100, min_return * 100, c='red', marker='*', s=200,
                   label='Minimum Variance')
        ax.scatter(optimal_risk * 100, optimal_return * 100, c='green', marker='*', s=200,
                   label='Optimal Risky Portfolio (Max Sharpe)')

        ax.set_title('Efficient Frontier with Monte Carlo Simulation')
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Expected Return (%)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig) # Display the matplotlib figure in Streamlit

        # ========================
        # 📜 Display Portfolio Details
        # ========================
        st.subheader("Portfolio Details")

        st.write("🌟 **Minimum Variance Portfolio Allocation:**")
        min_var_df = pd.DataFrame({'Ticker': tickers, 'Weight (%)': (min_weights * 100).round(2)})
        st.dataframe(min_var_df)
        st.write(f"Expected Return: {min_return*100:.2f}%")
        st.write(f"Volatility (Risk): {min_risk*100:.2f}%")
        st.write(f"Sharpe Ratio: {sharpe_ratio(min_return, min_risk, risk_free_rate_input):.2f}")


        st.write("\n🚀 **Optimal Risky Portfolio (Max Sharpe Ratio) Allocation:**")
        optimal_df = pd.DataFrame({'Ticker': tickers, 'Weight (%)': (optimal_weights * 100).round(2)})
        st.dataframe(optimal_df)
        st.write(f"Expected Return: {optimal_return*100:.2f}%")
        st.write(f"Volatility (Risk): {optimal_risk*100:.2f}%")
        st.write(f"Sharpe Ratio: {max_sharpe_ratio:.2f}")

        st.success("✅ Simulation completed!")

        # =========================================================================
        # Streamlit Download Button
        # Streamlit provides a direct download_button, making it much simpler
        # =========================================================================
        st.markdown("---") # Separator
        st.subheader("Download Results")
        
        # Combine both dataframes into a single CSV string for download
        csv_output = io.StringIO()
        csv_output.write("Optimal Risky Portfolio Allocation:\n")
        optimal_df.to_csv(csv_output, index=False, header=True)
        csv_output.write("\n\nMinimum Variance Portfolio Allocation:\n")
        min_var_df.to_csv(csv_output, index=False, header=True)

        st.download_button(
            label="Download Portfolio Data (CSV)",
            data=csv_output.getvalue(),
            file_name="simulated_portfolios.csv",
            mime="text/csv",
            help="Download the optimal and minimum variance portfolio allocations."
        )

# No separate "Simulate Another Portfolio" button needed as Streamlit reruns on input changes,
# and users can simply adjust inputs and click "Run" again.
# The `st.button` above implicitly acts as a "Run Simulation" and "Simulate Another" trigger.

