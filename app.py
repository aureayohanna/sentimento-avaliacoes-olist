"""
App do classificador de sentimento. Rodar com:
    streamlit run app.py
(precisa ter rodado o train.py antes para o modelo existir)
"""

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.explicar import classificar, explicar

RAIZ = Path(__file__).resolve().parent
MODELO = RAIZ / "models" / "modelo_sentimento.joblib"
REPORTS = RAIZ / "reports"

st.set_page_config(page_title="Sentimento de avaliações - Olist", layout="wide")


@st.cache_resource
def carregar_modelo():
    return joblib.load(MODELO)


if not MODELO.exists():
    st.error("Modelo não encontrado. Rode `python train.py` primeiro.")
    st.stop()

modelo = carregar_modelo()

st.title("Quem está insatisfeito? Classificador de avaliações")
st.caption("Modelo treinado com ~100 mil avaliações reais de e-commerce brasileiro (dataset público da Olist)")

aba1, aba2, aba3 = st.tabs(["Testar um comentário", "Fila do atendimento", "Como o modelo se saiu"])

# --- 1. um comentário por vez, com a explicação da decisão
with aba1:
    texto = st.text_area(
        "Escreva uma avaliação de cliente",
        "Comprei faz 20 dias e até agora não chegou, ninguém responde no chat",
        height=100,
    )
    if st.button("Classificar") and texto.strip():
        r = classificar(modelo, [texto]).iloc[0]
        col1, col2 = st.columns([1, 2])
        with col1:
            if r["sentimento"] == "negativo":
                st.error(f"NEGATIVO  \nprobabilidade: {r['prob_negativo']:.0%}")
            else:
                st.success(f"POSITIVO  \nprobabilidade: {1 - r['prob_negativo']:.0%}")
        with col2:
            exp = explicar(modelo, texto)
            if not exp.empty:
                st.write("Palavras que mais pesaram na decisão:")
                exp["lado"] = exp["peso"].map(lambda p: "positivo" if p > 0 else "negativo")
                grafico = (
                    alt.Chart(exp)
                    .mark_bar()
                    .encode(
                        x=alt.X("peso:Q", title="peso na decisão"),
                        y=alt.Y("termo:N", sort="x", title=None),
                        color=alt.Color(
                            "lado:N",
                            scale=alt.Scale(domain=["negativo", "positivo"], range=["#C44E52", "#55A868"]),
                            legend=alt.Legend(title=None),
                        ),
                    )
                )
                st.altair_chart(grafico, use_container_width=True)

# --- 2. vários comentários: ordena do mais insatisfeito pro menos
with aba2:
    st.write(
        "Suba um CSV com uma coluna de comentários. O modelo devolve a lista ordenada "
        "do cliente mais insatisfeito para o menos, para o atendimento começar por quem precisa."
    )
    arquivo = st.file_uploader("CSV com comentários", type="csv")
    if arquivo is not None:
        df = pd.read_csv(arquivo)
        coluna = st.selectbox("Coluna com o texto", df.columns)
        textos = df[coluna].fillna("").astype(str).tolist()
        resultado = classificar(modelo, textos).sort_values("prob_negativo", ascending=False)

        n_neg = (resultado["sentimento"] == "negativo").sum()
        st.metric("Avaliações negativas encontradas", f"{n_neg} de {len(resultado)}")
        st.dataframe(resultado, hide_index=True, use_container_width=True)
        st.download_button(
            "Baixar fila priorizada (CSV)",
            resultado.to_csv(index=False).encode("utf-8-sig"),
            "fila_atendimento.csv",
            "text/csv",
        )

# --- 3. métricas e gráficos gerados no treino
with aba3:
    arq_metricas = REPORTS / "metricas.json"
    if arq_metricas.exists():
        m = json.loads(arq_metricas.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelo escolhido", m["modelo"])
        c2.metric("F1 da classe negativa (teste)", f"{m['teste_f1_negativo']:.3f}")
        c3.metric("F1 macro (teste)", f"{m['teste_f1_macro']:.3f}")

        st.subheader("Comparação dos modelos (validação cruzada, 5 folds)")
        comp = pd.DataFrame(m["cv_f1_negativo"].items(), columns=["modelo", "F1 negativo"])
        st.dataframe(comp.sort_values("F1 negativo", ascending=False), hide_index=True)

    for nome, titulo in [
        ("04_matriz_confusao.png", "Matriz de confusão"),
        ("05_termos_importantes.png", "Termos que mais pesam para cada lado"),
        ("01_distribuicao_notas.png", "Distribuição das notas"),
        ("02_comentario_por_nota.png", "Quem escreve comentário, por nota"),
    ]:
        if (REPORTS / nome).exists():
            st.subheader(titulo)
            st.image(str(REPORTS / nome))
