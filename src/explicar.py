"""
Funções de apoio ao app: classificar textos e explicar a decisão do modelo.

Em modelo linear (Regressão Logística / SVM), a "nota" de um texto é a soma de
(peso da palavra no modelo) x (valor TF-IDF da palavra no texto). Então dá pra
mostrar exatamente quais palavras puxaram a decisão para cada lado.
"""

import numpy as np
import pandas as pd

from src.texto import limpar_texto


def prob_negativo(modelo, textos_limpos) -> np.ndarray:
    """Probabilidade (ou score entre 0 e 1) de o texto ser NEGATIVO."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(textos_limpos)[:, 0]
    # LinearSVC não dá probabilidade: uso a distância até a fronteira
    # passada por uma sigmoide, só pra ter uma escala de 0 a 1 para ordenar
    score = modelo.decision_function(textos_limpos)
    return 1 / (1 + np.exp(score))


def classificar(modelo, textos) -> pd.DataFrame:
    limpos = [limpar_texto(t) for t in textos]
    p_neg = prob_negativo(modelo, limpos)
    return pd.DataFrame({
        "texto": textos,
        "sentimento": np.where(p_neg >= 0.5, "negativo", "positivo"),
        "prob_negativo": p_neg.round(3),
    })


def explicar(modelo, texto: str, n: int = 8) -> pd.DataFrame:
    """Termos do texto que mais pesaram na decisão (positivo = puxou para positivo)."""
    clf = modelo.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["termo", "peso"])

    tfidf = modelo.named_steps["tfidf"]
    vetor = tfidf.transform([limpar_texto(texto)])
    idx = vetor.nonzero()[1]
    if len(idx) == 0:
        return pd.DataFrame(columns=["termo", "peso"])

    termos = tfidf.get_feature_names_out()[idx]
    pesos = vetor.toarray()[0, idx] * clf.coef_[0, idx]
    df = pd.DataFrame({"termo": termos, "peso": pesos})
    df = df.reindex(df["peso"].abs().sort_values(ascending=False).index).head(n)
    return df.reset_index(drop=True)
