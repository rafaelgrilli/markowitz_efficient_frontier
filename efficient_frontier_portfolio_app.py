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
st.set_page_config(page_title="Markowitz Pro", layout="wide")

st.title("📊 Simulador de Fronteira Eficiente")
st.write("Versão Otimizada 2026 - Foco em Estabilidade de Dados")

# ==============================================================================
# 🔢 Lógica Financeira e Funções de Risco
# ==============================================================================

def calcular_var_cvar(retornos_portfolio, confianca=0.95):
    """Calcula VaR e CVaR Histórico Anualizado"""
    retornos_ordenados = np.sort(retornos_portfolio)
    indice = int((1 - confianca) * len(retornos_ordenados))
    var = retornos_ordenados[indice]
    cvar = retornos_ordenados[:indice].mean()
    # Annualizando (aproximação por raiz de tempo)
    return -var * np.sqrt(252), -cvar * np.sqrt(252)

def neg_sharpe(pesos, ret_anuais, cov_matrix, rf):
    p_ret = np.dot(pesos, ret_anuais)
    p_vol = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
    return -(p_ret - rf) / p_vol

def vol_portfolio(pesos, cov_matrix):
    return np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))

# ==============================================================================
# 📥 Gestão de Dados (Corrigido para instabilidades do Yahoo)
# ==============================================================================

@st.cache_data(ttl=3600)
def obter_dados(lista_tickers, inicio, fim):
    try:
        obj = Ticker(lista_tickers)
        df = obj.history(start=inicio, end=fim)
        if isinstance(df, str) or df.empty:
            return None, None
        
        df = df.reset_index()
        col = 'adjclose' if 'adjclose' in df.columns else 'close'
        precos = df.pivot(index='date', columns='symbol', values=col).ffill().dropna()
        return precos, precos.columns.tolist()
    except:
        return None, None

# ==============================================================================
# 🎛️ Interface e Inputs
# ==============================================================================
with st.sidebar:
    st.header("Configurações")
    tickers_txt = st.text_input("Ativos:", "PETR4.SA, VALE3.SA, ITUB4.SA, ABEV3.SA, AAPL")
    rf_rate = st.number_input("Taxa Livre de Risco (Selic/T-Bill):", value=0.1075, format="%.4f")
    sd = st.date_input("Início:", date(2020, 1, 1))
    ed = st.date_input("Fim:", date.today())
    n_sim = st.slider("Simulações Monte Carlo:", 1000, 20000, 5000)

if st.button("🚀 Executar Otimização"):
    lista = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
    
    with st.spinner("Sincronizando com Yahoo Finance..."):
        precos, ativos = obter_dados(lista, sd.isoformat(), ed.isoformat())
        
        if precos is None:
            st.error("Erro ao puxar dados. Verifique os tickers (ex: PETR4.SA) e a conexão.")
            st.stop()

        # Cálculos Base
        rets_diarios = precos.pct_change().dropna()
        rets_anual = rets_diarios.mean() * 252
        cov_anual = rets_diarios.cov() * 252
        
        # --- Otimização SciPy ---
        n = len(ativos)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(n))
        init = n * [1./n]
        
        opt_s = sco.minimize(neg_sharpe, init, args=(rets_anual, cov_anual, rf_rate), method='SLSQP', bounds=bnds, constraints=cons)
        opt_v = sco.minimize(vol_portfolio, init, args=(cov_anual), method='SLSQP', bounds=bnds, constraints=cons)

        # --- Monte Carlo ---
        m_ret, m_vol, m_sh = [], [], []
        for _ in range(n_sim):
            w = np.random.rand(n)
            w /= np.sum(w)
            m_ret.append(np.dot(w, rets_anual))
            m_vol.append(np.sqrt(np.dot(w.T, np.dot(cov_anual, w))))
            m_sh.append((m_ret[-1] - rf_rate) / m_vol[-1])

        # ==============================================================================
        # 📈 Visualização e Resultados
        # ==============================================================================
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Carteira de Tangência (Max Sharpe)")
            res_s = pd.DataFrame({'Ativo': ativos, 'Peso': opt_s.x})
            st.dataframe(res_s.style.format({'Peso': '{:.2%}'}))
            
            # Métricas de Risco (Funcionalidade recuperada do seu código original)
            p_rets_opt = rets_diarios.dot(opt_s.x)
            v95, cv95 = calcular_var_cvar(p_rets_opt)
            st.metric("VaR Anualizado (95%)", f"{v95:.2%}")
            st.metric("CVaR (Expected Shortfall)", f"{cv95:.2%}")

        with col_b:
            st.subheader("Fronteira de Eficiência")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m_vol, y=m_ret, mode='markers', marker=dict(color=m_sh, colorscale='Viridis', size=5), name="Monte Carlo"))
            fig.add_trace(go.Scatter(x=[vol_portfolio(opt_s.x, cov_anual)], y=[np.dot(opt_s.x, rets_anual)], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="Max Sharpe"))
            fig.update_layout(xaxis_title="Volatilidade (Risco)", yaxis_title="Retorno Esperado", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Exportação CSV
        csv = io.StringIO()
        res_s.to_csv(csv, index=False)
        st.download_button("📥 Baixar Alocação (CSV)", csv.getvalue(), "portfolio.csv", "text/csv")

st.info("💡 Dica: Para ativos brasileiros, use sempre o sufixo .SA (ex: VALE3.SA).")
