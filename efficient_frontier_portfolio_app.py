import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import io
import scipy.optimize as sco

# ==============================================================================
# ⚙️ CONFIGURAÇÃO E ESTÉTICA
# ==============================================================================
st.set_page_config(page_title="Markowitz Portfolio Pro", layout="wide", initial_sidebar_state="expanded")

# CSS para melhorar a aparência
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Simulador Avançado de Carteiras: Markowitz & Risco")
st.write("---")

# ==============================================================================
# 🔢 MOTOR DE CÁLCULO E RISCO (ESPÍRITO ORIGINAL)
# ==============================================================================

def calcular_metricas_risco(retornos_portfolio, confianca=0.95):
    """Cálculo detalhado de VaR e CVaR Histórico Anualizado"""
    rets = np.sort(retornos_portfolio)
    idx = int((1 - confianca) * len(rets))
    var_diario = rets[idx]
    cvar_diario = rets[:idx].mean()
    # Escalonamento para o ano (252 dias úteis)
    return abs(var_diario * np.sqrt(252)), abs(cvar_diario * np.sqrt(252))

def calcular_stats_carteira(pesos, rets_anuais, matriz_cov, rf):
    """Retorna Retorno, Volatilidade e Sharpe de uma composição"""
    p_ret = np.dot(pesos, rets_anuais)
    p_vol = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))
    p_sharpe = (p_ret - rf) / p_vol if p_vol != 0 else 0
    return p_ret, p_vol, p_sharpe

# Funções de Otimização (SciPy)
def min_func_sharpe(pesos, rets_anuais, matriz_cov, rf):
    return -calcular_stats_carteira(pesos, rets_anuais, matriz_cov, rf)[2]

def min_func_vol(pesos, rets_anuais, matriz_cov, rf):
    return calcular_stats_carteira(pesos, rets_anuais, matriz_cov, rf)[1]

# ==============================================================================
# 📥 GERENCIAMENTO DE DADOS (VERSÃO ROBUSTA)
# ==============================================================================

@st.cache_data(ttl=3600)
def fetch_data(tickers, start, end):
    try:
        t_obj = Ticker(tickers)
        df = t_obj.history(start=start, end=end)
        if isinstance(df, str) or df.empty: return None
        df = df.reset_index()
        col_preco = 'adjclose' if 'adjclose' in df.columns else 'close'
        prices = df.pivot(index='date', columns='symbol', values=col_preco).ffill().dropna()
        return prices
    except: return None

# ==============================================================================
# 🎛️ INTERFACE LATERAL (FUNCIONALIDADES COMPLETAS)
# ==============================================================================
with st.sidebar:
    st.header("🎯 Configurações da Estratégia")
    tickers_in = st.text_input("Ativos (Ticker + .SA se B3):", "PETR4.SA, VALE3.SA, ITUB4.SA, WEGE3.SA, AAPL, MSFT")
    
    col_rf, col_sim = st.columns(2)
    with col_rf:
        rf_rate = st.number_input("Tx. Livre Risco:", value=0.1075, step=0.005, format="%.4f")
    with col_sim:
        n_portfolios = st.number_input("Simulações MC:", 1000, 50000, 10000, 1000)
    
    d_ini = st.date_input("Início da Análise:", date(2020, 1, 1))
    d_fim = st.date_input("Fim da Análise:", date.today())
    
    st.markdown("---")
    st.subheader("⚖️ Restrições de Peso")
    min_w = st.slider("Peso Mínimo por Ativo:", 0.0, 0.2, 0.0)
    max_w = st.slider("Peso Máximo por Ativo:", 0.1, 1.0, 1.0)
    
    run = st.button("🚀 EXECUTAR OTIMIZAÇÃO", use_container_width=True)

# ==============================================================================
# 🚀 EXECUÇÃO E DASHBOARD
# ==============================================================================
if run:
    lista_t = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    
    with st.spinner("Puxando dados e processando matrizes..."):
        precos = fetch_data(lista_t, d_ini.isoformat(), d_fim.isoformat())
        
        if precos is None or precos.empty:
            st.error("❌ Erro ao obter dados. Verifique a internet e os tickers.")
            st.stop()

        ativos = precos.columns.tolist()
        n = len(ativos)
        
        # Estatísticas de Mercado
        rets_diarios = precos.pct_change().dropna()
        rets_anual = rets_diarios.mean() * 252
        matriz_cov = rets_diarios.cov() * 252
        correlacao = rets_diarios.corr()

        # 1. Otimização Numérica (SciPy) - Max Sharpe e Min Vol
        bounds = tuple((min_w, max_w) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        init_guess = n * [1./n]
        
        opt_sharpe = sco.minimize(min_func_sharpe, init_guess, args=(rets_anual, matriz_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
        opt_vol = sco.minimize(min_func_vol, init_guess, args=(rets_anual, matriz_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)

        # 2. Monte Carlo (Espírito Original)
        mc_rets, mc_vols, mc_sharpes = [], [], []
        for _ in range(n_portfolios):
            w = np.random.random(n)
            w /= np.sum(w)
            r, v, s = calcular_stats_carteira(w, rets_anual, matriz_cov, rf_rate)
            mc_rets.append(r)
            mc_vols.append(v)
            mc_sharpes.append(s)

        # ----------------------------------------------------------------------
        # 📈 DASHBOARD DE RESULTADOS
        # ----------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["💎 Carteiras Ótimas", "📈 Fronteira Eficiente", "📊 Análise de Ativos"])
        
        with tab1:
            st.subheader("Comparativo de Alocação")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            # Carteira de Tangência (Max Sharpe)
            r_s, v_s, s_s = calcular_stats_carteira(opt_sharpe.x, rets_anual, matriz_cov, rf_rate)
            # Risco de Cauda
            var_s, cvar_s = calcular_metricas_risco(rets_diarios.dot(opt_sharpe.x))
            
            with col_m1:
                st.metric("Retorno Esperado (Max Sharpe)", f"{r_s:.2%}")
                st.metric("Sharpe Ratio", f"{s_s:.2f}")
            with col_m2:
                st.metric("Volatilidade (Risco)", f"{v_s:.2%}")
                st.metric("VaR Anualizado (95%)", f"{var_s:.2%}")
            with col_m3:
                st.metric("CVaR (Expected Shortfall)", f"{cvar_s:.2%}")
            
            # Tabela de Pesos
            st.write("#### Composição Detalhada")
            df_pesos = pd.DataFrame({
                "Ativo": ativos,
                "Max Sharpe (%)": (opt_sharpe.x * 100).round(2),
                "Min Volatilidade (%)": (opt_vol.x * 100).round(2)
            })
            st.dataframe(df_pesos, use_container_width=True)

        with tab2:
            st.subheader("Visualização da Fronteira Eficiente")
            fig = go.Figure()
            
            # Nuvens Monte Carlo
            fig.add_trace(go.Scatter(x=mc_vols, y=mc_rets, mode='markers', 
                                     marker=dict(color=mc_sharpes, colorscale='Viridis', size=5, colorbar=dict(title="Sharpe")),
                                     name="Simulações MC"))
            
            # Ponto Ótimo Sharpe
            fig.add_trace(go.Scatter(x=[v_s], y=[r_s], mode='markers', 
                                     marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='white')),
                                     name="Carteira de Tangência (Max Sharpe)"))
            
            # Ponto Min Vol
            r_v, v_v, _ = calcular_stats_carteira(opt_vol.x, rets_anual, matriz_cov, rf_rate)
            fig.add_trace(go.Scatter(x=[v_v], y=[r_v], mode='markers', 
                                     marker=dict(color='white', size=12, symbol='circle', line=dict(width=2, color='blue')),
                                     name="Mínima Variância"))

            fig.update_layout(xaxis_title="Risco (Volatilidade Anualizada)", yaxis_title="Retorno Esperado Anualizado",
                              height=600, template="plotly_white", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            c_a, c_b = st.columns(2)
            with c_a:
                st.subheader("Correlação entre Ativos")
                st.dataframe(correlacao.style.background_gradient(cmap='coolwarm').format("{:.2f}"))
            with c_b:
                st.subheader("Retornos Acumulados Individuais")
                st.line_chart(precos / precos.iloc[0])

        # Exportação
        csv = io.StringIO()
        df_pesos.to_csv(csv, index=False)
        st.download_button("📥 Baixar Relatório de Alocação (CSV)", csv.getvalue(), "otimizacao_carteira.csv", "text/csv")

else:
    st.info("💡 Configure os ativos na barra lateral e clique em **Executar** para ver a mágica financeira.")
    # Exemplo visual de como o app se organiza
    #
