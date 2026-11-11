"""
Análise exploratória das avaliações da Olist.
Gera os gráficos em reports/ e imprime os números principais.

    python eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.texto import montar_base

RAIZ = Path(__file__).resolve().parent
ARQ = RAIZ / "data" / "olist_order_reviews_dataset.csv"
SAIDA = RAIZ / "reports"
SAIDA.mkdir(exist_ok=True)

df = pd.read_csv(ARQ)
print(f"Total de avaliações: {len(df):,}")

tem_comentario = df["review_comment_message"].notna()
print(f"Com comentário escrito: {tem_comentario.mean():.1%}")

# 1) distribuição das notas
dist = df["review_score"].value_counts().sort_index()
print("\nNotas:\n", (dist / dist.sum()).round(3).to_string())

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(dist.index.astype(str), dist.values, color="#4C72B0")
ax.set_title("Distribuição das notas (1 a 5)")
ax.set_xlabel("nota")
ax.set_ylabel("quantidade de avaliações")
fig.tight_layout()
fig.savefig(SAIDA / "01_distribuicao_notas.png", dpi=120)

# 2) quem dá nota baixa escreve mais?
pct_coment = df.assign(comentou=tem_comentario).groupby("review_score")["comentou"].mean()
print("\n% que escreveu comentário, por nota:\n", pct_coment.round(3).to_string())

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(pct_coment.index.astype(str), pct_coment.values * 100, color="#DD8452")
ax.set_title("% de clientes que escreveram comentário, por nota")
ax.set_xlabel("nota")
ax.set_ylabel("%")
fig.tight_layout()
fig.savefig(SAIDA / "02_comentario_por_nota.png", dpi=120)

# 3) tamanho do texto por sentimento (base já tratada)
base = montar_base(df)
base["n_palavras"] = base["texto"].str.split().str.len()
print("\nBase para o modelo:", len(base), "avaliações")
print("Proporção positivo/negativo:\n", base["sentimento"].value_counts(normalize=True).round(3).to_string())
print("\nMediana de palavras por sentimento (0=neg, 1=pos):\n",
      base.groupby("sentimento")["n_palavras"].median().to_string())

fig, ax = plt.subplots(figsize=(6, 4))
for valor, nome, cor in [(1, "positivo", "#55A868"), (0, "negativo", "#C44E52")]:
    ax.hist(base.loc[base["sentimento"] == valor, "n_palavras"].clip(upper=60),
            bins=30, alpha=0.6, label=nome, color=cor)
ax.set_title("Tamanho do comentário (palavras)")
ax.set_xlabel("palavras (cortado em 60)")
ax.legend()
fig.tight_layout()
fig.savefig(SAIDA / "03_tamanho_texto.png", dpi=120)

print(f"\nGráficos salvos em {SAIDA}")
