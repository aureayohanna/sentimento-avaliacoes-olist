"""
Treina e avalia o classificador de sentimento.

Compara 3 modelos com validação cruzada, escolhe o melhor pelo F1 da classe
negativa (que é a que interessa pro negócio: achar cliente insatisfeito) e
avalia no conjunto de teste, que fica separado desde o começo.

    python train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score, make_scorer)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.texto import montar_base

RAIZ = Path(__file__).resolve().parent
ARQ = RAIZ / "data" / "olist_order_reviews_dataset.csv"
REPORTS = RAIZ / "reports"
MODELS = RAIZ / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SEED = 42

base = montar_base(pd.read_csv(ARQ))
X, y = base["texto_limpo"], base["sentimento"]
print(f"Base: {len(base)} avaliações | positivas: {y.mean():.1%}")

X_treino, X_teste, y_treino, y_teste, idx_treino, idx_teste = train_test_split(
    X, y, base.index, test_size=0.2, stratify=y, random_state=SEED
)


def tfidf():
    # unigramas + bigramas pra pegar coisas como "nao chegou", "nao recomendo"
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)


candidatos = {
    "baseline (classe mais comum)": Pipeline([("tfidf", tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))]),
    "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
    "Regressão Logística": Pipeline([("tfidf", tfidf()),
                                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0))]),
    "SVM linear": Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(class_weight="balanced"))]),
}

f1_negativo = make_scorer(f1_score, pos_label=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("\nValidação cruzada (5 folds) - F1 da classe negativa:")
resultados = {}
for nome, modelo in candidatos.items():
    notas = cross_val_score(modelo, X_treino, y_treino, cv=cv, scoring=f1_negativo, n_jobs=-1)
    resultados[nome] = (notas.mean(), notas.std())
    print(f"  {nome:<30} {notas.mean():.3f} (+/- {notas.std():.3f})")

# escolho entre os modelos "de verdade" (o baseline é só referência)
melhor_nome = max((n for n in resultados if not n.startswith("baseline")), key=lambda n: resultados[n][0])
print(f"\nMelhor na validação: {melhor_nome}")

# Regressão Logística costuma empatar com o SVM; se a diferença for pequena
# eu prefiro ela porque dá probabilidade e é mais fácil de explicar.
rl = resultados["Regressão Logística"][0]
if melhor_nome != "Regressão Logística" and resultados[melhor_nome][0] - rl < 0.01:
    print("Diferença < 0.01 para a Regressão Logística -> fico com ela (dá probabilidade e é interpretável)")
    melhor_nome = "Regressão Logística"

modelo = candidatos[melhor_nome]
modelo.fit(X_treino, y_treino)
pred = modelo.predict(X_teste)

print("\n=== Resultado no TESTE (dados que o modelo nunca viu) ===")
relatorio = classification_report(y_teste, pred, target_names=["negativo", "positivo"], digits=3)
print(relatorio)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_teste, pred, display_labels=["negativo", "positivo"], cmap="Blues", ax=ax, colorbar=False
)
ax.set_title(f"Matriz de confusão - {melhor_nome}")
fig.tight_layout()
fig.savefig(REPORTS / "04_matriz_confusao.png", dpi=120)

# palavras que mais puxam para cada lado (só faz sentido em modelo linear)
clf = modelo.named_steps["clf"]
if hasattr(clf, "coef_"):
    vocab = np.array(modelo.named_steps["tfidf"].get_feature_names_out())
    coefs = clf.coef_[0]
    ordem = np.argsort(coefs)
    neg = pd.Series(coefs[ordem[:15]], index=vocab[ordem[:15]])
    pos = pd.Series(coefs[ordem[-15:]], index=vocab[ordem[-15:]])
    print("Termos mais negativos:", ", ".join(neg.index))
    print("Termos mais positivos:", ", ".join(pos.index[::-1]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    neg.sort_values(ascending=False).plot.barh(ax=axes[0], color="#C44E52")
    axes[0].set_title("Termos que indicam avaliação negativa")
    pos.plot.barh(ax=axes[1], color="#55A868")
    axes[1].set_title("Termos que indicam avaliação positiva")
    fig.tight_layout()
    fig.savefig(REPORTS / "05_termos_importantes.png", dpi=120)

# análise de erros: salvo os casos que o modelo errou pra ler com calma
teste = base.loc[idx_teste].assign(previsto=pred)
erros = teste[teste["sentimento"] != teste["previsto"]]
erros[["review_score", "texto", "sentimento", "previsto"]].to_csv(
    REPORTS / "erros_modelo.csv", index=False, encoding="utf-8-sig"
)
print(f"\n{len(erros)} erros salvos em reports/erros_modelo.csv para análise")

# salvo o modelo e as métricas
joblib.dump(modelo, MODELS / "modelo_sentimento.joblib")
metricas = {
    "modelo": melhor_nome,
    "cv_f1_negativo": {k: round(v[0], 4) for k, v in resultados.items()},
    "teste_f1_negativo": round(f1_score(y_teste, pred, pos_label=0), 4),
    "teste_f1_macro": round(f1_score(y_teste, pred, average="macro"), 4),
    "n_treino": int(len(X_treino)),
    "n_teste": int(len(X_teste)),
}
(REPORTS / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
print("Modelo salvo em models/modelo_sentimento.joblib")
