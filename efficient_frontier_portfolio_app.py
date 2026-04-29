# efficient_frontier_portfolio_app.py

import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import io
import scipy.optimize as sco

# ==============================================================================
# ⚙️ CONFIGURAÇÃO E ESTILO
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Analytics Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS para notas explicativas e cards de métricas
st.markdown("""
    <style>
    .nota-explicativa { font-size: 0.88rem; color: #555; font-style: italic; margin-bottom: 20px; line-height: 1.4; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; }
    h1, h2, h3 { color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Estação de Trabalho: Análise e Otimização de Portfólio")
st.write("Versão Consolidada 2026 - Rigor Analítico e Educativo")
st.write("---")

# ==============================================================================
# 🔢 FUNÇÕES ANALÍTICAS E MATEMÁTICAS
# ==============================================================================

def calculate_stats(weights, rets_anual, cov_mat, rf):
    """Calcula estatísticas básicas do portfólio."""
    p_ret = np.sum(rets_anual * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    p_sharpe = (p_ret - rf) / p_vol if p_vol != 0 else 0
    return p_ret, p_vol, p_sharpe

def calculate_advanced_metrics(rets_series, weights, rf_diario):
    """Calcula Sortino, Drawdown, VaR e CVaR."""
    p_rets = rets_series.dot(weights)
    
    # Índice de Sortino (Risco de queda)
    downside_rets = p_rets[p_rets < 0]
    down_std = np.std(downside_rets) * np.sqrt(252)
    p_ret_anual = p_rets.mean() * 252
    sortino = (p_ret_anual - (rf_diario * 252)) / down_std if down_std != 0 else 0
    
    # Drawdown Máximo
    cum_rets = (1 + p_rets).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
    
    # VaR e CVaR Histórico (95% confiança)
    sorted_rets = np.sort(p_rets)
    idx = int(0.05 * len(sorted_rets))
    var_95 = -sorted_rets[idx] * np.sqrt(252)
    cvar_95 = -sorted_rets[:idx].mean() * np.sqrt(252)
    
    return sortino, max_dd, var_95, cvar_95

def risk_parity_objective(weights, cov_mat):
    """Função objetivo para equalizar a contribuição de risco de cada ativo."""
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    marginal_risk = np.dot(cov_mat, weights) / p_vol
    risk_contribution = weights * marginal_risk
    target_risk = p_vol / len(weights)
    return np.sum(np.square(risk_contribution - target_risk))

# ==============================================================================
# 📥 GESTÃO DE DADOS (YAHOO FINANCE)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_consolidated_data(tickers, start, end):
    try:
        # Lógica híbrida de benchmark
        bench = "^BVSP" if any(".SA" in t.upper() for t in tickers) else "^GSPC"
        full_list = list(set(tickers + [bench]))
        
        t_obj = Ticker(full_list)
        df = t_obj.history(start=start, end=end)
        
        if df is None or (isinstance(df, pd.DataFrame) and df.empty) or isinstance(df, str):
            return None, None, None
        
        df = df.reset_index()
        col = 'adjclose' if 'adjclose' in df.columns else 'close'
        prices = df.pivot(index='date', columns='symbol', values=col).ffill().dropna()
        
        if bench not in prices.columns:
            return prices, None, None
            
        bench_data = prices[bench]
        assets_data = prices.drop(columns=[bench])
        return assets_data, bench_data, bench
    except:
        return None, None, None

# ==============================================================================
# 🎛️ INTERFACE DO USUÁRIO (WIDGETS)
# ==============================================================================

with st.expander("❓ GUIA ANALÍTICO: Como interpretar os indicadores?", expanded=False):
    st.markdown("""
    Este simulador utiliza modelos de **Finanças Quantitativas** para otimizar sua alocação.
    
    * **Índice de Sharpe:** Mede o retorno excedente por unidade de volatilidade total. Útil para comparar portfólios de perfis similares.
    * **Índice de Sortino:** Foca no 'risco ruim'. Penaliza apenas as variações negativas, sendo ideal para investidores fundamentalistas.
    * **VaR (95%):** 'Value at Risk'. Indica que há 95% de chance de sua perda anual não exceder esse valor.
    * **Max Drawdown:** Mostra a maior queda histórica da carteira. Ajuda a entender a resiliência psicológica necessária.
    * **Paridade de Risco:** Diferente de Markowitz, aqui o objetivo não é o lucro máximo, mas sim garantir que nenhum ativo domine o risco da carteira sozinho.
    """)

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("Configuração do Portfólio")
    tickers_in = st.text_input(
        "Ativos (Use .SA para B3. Ex: VALE3.SA, PETR4.SA, AAPL, BTC-USD):", 
        "VALE3.SA, ITUB4.SA, AAPL, MSFT"
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        s_date = st.date_input("Início da análise:", date(2021, 1, 1))
    with c2:
        e_date = st.date_input("Fim da análise:", date.today())
    with c3:
        num_sim = st.slider("Simulações Monte Carlo:", 1000, 20000, 5000)

with col_side:
    st.subheader("Parâmetros")
    rf_rate = st.number_input("Taxa Livre de Risco (Anual %):", 0.0, 0.20, 0.1075, step=0.0025, format="%.4f")
    allow_short = st.checkbox("Permitir Short Selling (Venda a Descoberto)", value=False)

# Restrições granulares (conforme código inicial)
st.subheader("Restrições de Pesos")
c_min, c_max = st.columns(2)
with c_min:
    min_w = st.number_input("Peso Mínimo por Ativo:", -1.0 if allow_short else 0.0, 1.0, 0.0)
with c_max:
    max_w = st.number_input("Peso Máximo por Ativo:", 0.0, 2.0, 1.0)

# ==============================================================================
# 🚀 EXECUÇÃO E DASHBOARD
# ==============================================================================

if st.button("📈 GERAR RELATÓRIO CONSOLIDADO", use_container_width=True):
    t_list = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    
    with st.spinner("Sincronizando com Yahoo Finance e rodando otimizadores..."):
        prices, bench_prices, bench_ticker = get_market_data = get_consolidated_data(t_list, s_date.isoformat(), e_date.isoformat())
        
        if prices is not None:
            rets = prices.pct_change().dropna()
            rets_a = rets.mean() * 252
            cov_a = rets.cov() * 252
            n_assets = len(prices.columns)
            rf_d = rf_rate / 252

            # Otimização Markowitz (Max Sharpe)
            bnds = tuple((min_w, max_w) for _ in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            init = n_assets * [1./n_assets]
            
            opt_s = sco.minimize(lambda w: -calculate_stats(w, rets_a, cov_a, rf_rate)[2], init, method='SLSQP', bounds=bnds, constraints=cons)
            
            # Otimização Paridade de Risco
            opt_rp = sco.minimize(risk_parity_objective, init, args=(cov_a,), method='SLSQP', bounds=bnds, constraints=cons)

            # --- RESULTADOS ---
            st.header("1. Comparativo de Alocação e Risco")
            
            tab_s, tab_r = st.tabs(["🎯 Máximo Sharpe (Tangência)", "⚖️ Paridade de Risco"])
            
            with tab_s:
                ws = opt_s.x
                rs, vs, ss = calculate_stats(ws, rets_a, cov_a, rf_rate)
                sort_s, dd_s, var_s, cvar_s = calculate_advanced_metrics(rets, ws, rf_d)
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Retorno Esperado", f"{rs:.2%}")
                col_m2.metric("Volatilidade", f"{vs:.2%}")
                col_m3.metric("Índice de Sortino", f"{sort_s:.2f}")
                col_m4.metric("Max Drawdown", f"{dd_s:.2%}")
                
                st.write(f"**VaR (95%):** {var_s:.2%} | **CVaR (95%):** {cvar_s:.2%}")
                st.table(pd.DataFrame({'Peso': ws}, index=prices.columns).style.format("{:.2%}"))

            with tab_r:
                wr = opt_rp.x
                rr, vr, sr = calculate_stats(wr, rets_a, cov_a, rf_rate)
                sort_r, dd_r, var_r, cvar_r = calculate_advanced_metrics(rets, wr, rf_d)
                
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                col_r1.metric("Retorno Esperado", f"{rr:.2%}")
                col_r2.metric("Volatilidade", f"{vr:.2%}")
                col_r3.metric("Índice de Sortino", f"{sort_r:.2f}")
                col_r4.metric("Max Drawdown", f"{dd_r:.2%}")
                
                st.write(f"**VaR (95%):** {var_r:.2%} | **CVaR (95%):** {cvar_r:.2%}")
                st.table(pd.DataFrame({'Peso': wr}, index=prices.columns).style.format("{:.2%}"))

            # --- GRÁFICOS ---
            st.header("2. Fronteira Eficiente e Backtesting")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                # Simulação Monte Carlo para o gráfico
                mc_r, mc_v = [], []
                for _ in range(num_sim):
                    w = np.random.random(n_assets); w /= np.sum(w)
                    mc_r.append(np.sum(rets_a * w)); mc_v.append(np.sqrt(np.dot(w.T, np.dot(cov_a, w))))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.array(mc_v)*100, y=np.array(mc_r)*100, 
                    mode='markers', 
                    marker=dict(color=(np.array(mc_r)-rf_rate)/np.array(mc_v), colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
                    name="Portfólios Simulados"
                ))
                fig.add_trace(go.Scatter(x=[vs*100], y=[rs*100], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
                fig.update_layout(
                    title="Fronteira Eficiente", xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
                    margin=dict(r=100)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_g2:
                # Backtesting vs Benchmark
                if bench_prices is not None:
                    bench_rets = bench_prices.pct_change().dropna()
                    cum_port = (1 + rets.dot(ws)).cumprod() * 10000
                    cum_bench = (1 + bench_rets).cumprod() * 10000
                    
                    fig_back = go.Figure()
                    fig_back.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Meu Portfólio (Max Sharpe)", line=dict(color='#10b981', width=3)))
                    fig_back.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=f"Mercado ({bench_ticker})", line=dict(color='#94a3b8', dash='dash')))
                    fig_back.update_layout(title="Crescimento de $10.000", xaxis_title="Data", yaxis_title="Valor Acumulado ($)", legend=dict(orientation="h", y=-0.4))
                    st.plotly_chart(fig_back, use_container_width=True)

            # Matriz de Correlação
            st.subheader("3. Matriz de Correlação entre Ativos")
            st.dataframe(rets.corr().round(2))

            # Download
            csv = io.StringIO()
            pd.DataFrame({'Ativo': prices.columns, 'Peso Max Sharpe': ws, 'Peso Risk Parity': wr}).to_csv(csv, index=False)
            st.download_button("📥 Baixar Relatório de Pesos (CSV)", csv.getvalue(), "portfolio_weights.csv", "text/csv")

        else:
            st.error("Erro ao buscar dados. Tente ajustar os tickers ou as datas.")

st.sidebar.markdown(f"© 2026 Rafael Grilli Felizardo")
st.sidebar.info("App Analytics Pro: Estabilidade e Rigor Financeiro.")
