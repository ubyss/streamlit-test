import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Classificador de Câncer de Mama",
    page_icon="🔬",
    layout="wide"
)

# =========================
# ESTILOS (CSS in-line)
# =========================
st.markdown("""
<style>
:root {
  --card-bg: #ffffff0d;
  --card-border: #ffffff22;
}
html, body, [class*="css"]  { font-family: Inter, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"; }
.main .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
.app-card {
  border: 1px solid var(--card-border);
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(6px);
  border-radius: 16px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 6px 24px rgba(0,0,0,0.08);
}
.app-badge {
  display: inline-flex; align-items: center; gap: .5rem;
  padding: .25rem .6rem; border-radius: 999px;
  font-size: .80rem; font-weight: 600; letter-spacing: .2px;
  border: 1px solid var(--card-border); background: rgba(255,255,255,0.06);
}
.app-badge.ok { color: #22c55e; }
.app-badge.warn { color: #f59e0b; }
.app-badge.err { color: #ef4444; }
.result-pill {
  display:inline-block; padding:.35rem .75rem; border-radius:999px;
  font-weight:700; letter-spacing:.2px;
}
.result-pill.m { background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }
.result-pill.b { background:#dcfce7; color:#065f46; border:1px solid #bbf7d0; }
.kpi { font-size: 2rem; font-weight: 800; margin: 0; }
.kpi-sub { color: #8b949e; margin-top: .1rem; }
hr.soft { border: none; height: 1px; background: linear-gradient(90deg, transparent, #ffffff22, transparent); margin: .5rem 0 1rem; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTES
# =========================
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

GROUPS = {
    "Médias (mean)": [f for f in FEATURES if f.endswith("_mean")],
    "Erro padrão (se)": [f for f in FEATURES if f.endswith("_se")],
    "Pior caso (worst)": [f for f in FEATURES if f.endswith("_worst")],
}

# =========================
# CACHES
# =========================
@st.cache_resource
def load_keras_model():
    try:
        return load_model('data/breast_cancer_model.keras')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo 'breast_cancer_model.keras': {e}")
        return None

@st.cache_data
def load_data_and_scaler():
    try:
        df = pd.read_csv('data/Breast_cancer_dataset.csv')
        stats = {
            'M': df[df['diagnosis'] == 'M'][FEATURES].describe().loc[['mean', 'std']],
            'B': df[df['diagnosis'] == 'B'][FEATURES].describe().loc[['mean', 'std']],
            'A': df[FEATURES].describe().loc[['mean', 'std']]
        }
        scaler = StandardScaler().fit(df[FEATURES])
        return df, scaler, stats
    except FileNotFoundError:
        st.error("Erro: O arquivo 'Breast_cancer_dataset.csv' não foi encontrado na pasta 'data'.")
        return None, None, None

# =========================
# FUNÇÕES
# =========================
def generate_synthetic_data(choice, stats):
    target_stats = stats[choice]
    for feature in FEATURES:
        mean = target_stats.loc['mean', feature]
        std = target_stats.loc['std', feature]
        st.session_state[feature] = float(np.random.normal(loc=mean, scale=max(std, 1e-6)))

def status_badge(ok: bool, label_ok: str, label_err: str):
    if ok:
        st.markdown(f'<span class="app-badge ok">🟢 {label_ok}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="app-badge err">🔴 {label_err}</span>', unsafe_allow_html=True)

# =========================
# HEADER
# =========================
col_title, col_status = st.columns([0.7, 0.3])
with col_title:
    st.title("🔬 Classificador Interativo de Câncer de Mama")
    st.markdown("Ajuste os **features** na lateral, gere amostras sintéticas e classifique. Visual limpo, feedback direto.")
with col_status:
    with st.container():
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        model = load_keras_model()
        df, scaler, stats = load_data_and_scaler()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="soft" />', unsafe_allow_html=True)

# =========================
# SIDEBAR (PAINEL)
# =========================
with st.sidebar:
    st.header("Painel de Controle")
    st.caption("Gere dados e execute a predição.")

    st.subheader("1) Geração de Dados")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if st.button("🧬 Amostra Benigna (B)", use_container_width=True, disabled=stats is None):
            generate_synthetic_data('B', stats)
            st.toast("Amostra benigna gerada.", icon="✅")
    with col_g2:
        if st.button("☣️ Amostra Maligna (M)", use_container_width=True, disabled=stats is None):
            generate_synthetic_data('M', stats)
            st.toast("Amostra maligna gerada.", icon="☣️")

    if st.button("🎲 Amostra Aleatória", use_container_width=True, disabled=stats is None):
        generate_synthetic_data('A', stats)
        st.toast("Amostra aleatória gerada.", icon="🎲")

    st.divider()

    st.subheader("2) Classificação")
    if st.button("🔎 Classificar Amostra", type="primary", use_container_width=True, disabled=not (model and scaler)):
        if model and scaler:
            current_data = [st.session_state.get(f, 0.0) for f in FEATURES]
            sample_df = pd.DataFrame([current_data], columns=FEATURES)
            scaled_sample = scaler.transform(sample_df)
            with st.spinner("Executando predição..."):
                time.sleep(0.3)
                prediction_proba = float(model.predict(scaled_sample, verbose=0)[0][0])
            st.session_state['last_prediction'] = prediction_proba
            st.toast("Classificação concluída.", icon="🔬")

# =========================
# CONTEÚDO PRINCIPAL
# =========================
if model and scaler and stats:
    # Inicializa primeira amostra
    if FEATURES[0] not in st.session_state:
        generate_synthetic_data('A', stats)
        st.session_state['last_prediction'] = None

    # TOP RESULT CARD
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    result_col1, result_col2, result_col3 = st.columns([0.35, 0.35, 0.30])

    with result_col1:
        st.subheader("Resultado")
        if 'last_prediction' in st.session_state and st.session_state['last_prediction'] is not None:
            p = st.session_state['last_prediction']
            malignant = p > 0.5
            if malignant:
                st.markdown('<span class="result-pill m">Diagnóstico: Maligno (M)</span>', unsafe_allow_html=True)
                st.markdown(f'<p class="kpi">{p*100:.2f}%</p><div class="kpi-sub">confiança em maligno</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="result-pill b">Diagnóstico: Benigno (B)</span>', unsafe_allow_html=True)
                st.markdown(f'<p class="kpi">{(1-p)*100:.2f}%</p><div class="kpi-sub">confiança em benigno</div>', unsafe_allow_html=True)
        else:
            st.info("Ainda sem predição. Gere/ajuste a amostra e clique em **Classificar Amostra**.")

    with result_col2:
        st.subheader("Distribuição da Confiança")
        if 'last_prediction' in st.session_state and st.session_state['last_prediction'] is not None:
            p = st.session_state['last_prediction']
            st.progress(min(max(int(p*100), 0), 100), text=f"Prob. Maligno: {p*100:.2f}%")
            st.caption(f"Prob. Benigno: {(1-p)*100:.2f}%")
        else:
            st.empty()

    with result_col3:
        st.subheader("Ações Rápidas")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("🔁 Reset Aleatório", use_container_width=True):
                generate_synthetic_data('A', stats)
        with col_q2:
            if st.button("💾 Salvar Amostra", use_container_width=True):
                current = {f: st.session_state.get(f, 0.0) for f in FEATURES}
                df_save = pd.DataFrame([current])
                ts = int(time.time())
                path = f"sample_{ts}.csv"
                df_save.to_csv(path, index=False)
                st.success(f"Amostra salva em `{path}`.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="soft" />', unsafe_allow_html=True)

    # TABS: Entradas | Estatísticas | Sobre
    tab_inputs, tab_stats, tab_info = st.tabs(["🧩 Entradas", "📊 Estatísticas", "ℹ️ Sobre o modelo"])

    # --- Entradas
    with tab_inputs:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("Ajustar Valores da Amostra")

        # Inputs agrupados por categoria, 3 colunas por grupo
        for group_name, group_features in GROUPS.items():
            st.markdown(f"**{group_name}**")
            cols = st.columns(3)
            for i, feature in enumerate(group_features):
                with cols[i % 3]:
                    # valor padrão seguro
                    default_val = float(st.session_state.get(feature, 0.0))
                    st.session_state[feature] = st.number_input(
                        label=feature.replace('_', ' ').title(),
                        value=default_val,
                        key=f"input_{feature}",
                        format="%.4f",
                        help="Ajuste fino do feature."
                    )
            st.markdown('<hr class="soft" />', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Estatísticas
    with tab_stats:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("Resumo Estatístico do Dataset")
        st.caption("Médias e desvios por classe (B/M) e agregado (A).")
        subtab_b, subtab_m, subtab_a = st.tabs(["Benigno (B)", "Maligno (M)", "Agregado (A)"])
        with subtab_b:
            st.dataframe(stats['B'].round(4))
        with subtab_m:
            st.dataframe(stats['M'].round(4))
        with subtab_a:
            st.dataframe(stats['A'].round(4))
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Sobre
    with tab_info:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("Sobre o Modelo")
        st.markdown("""
- **Entrada**: 30 features do conjunto *Breast Cancer Wisconsin*.
- **Pré-processamento**: `StandardScaler` ajustado ao conjunto completo.
- **Saída**: Probabilidade de malignidade (limiar 0.5).
- **Observação**: Este app é para fins educacionais; não substitui diagnóstico médico.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("A aplicação não pode ser iniciada. Verifique se os arquivos 'breast_cancer_model.keras' e 'data/Breast_cancer_dataset.csv' existem.")

