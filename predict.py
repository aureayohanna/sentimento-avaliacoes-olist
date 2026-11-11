"""
Usa o modelo treinado para classificar textos novos.

    python predict.py "produto chegou quebrado e ninguem responde"
    python predict.py            # roda alguns exemplos
"""

import sys
from pathlib import Path

import joblib

from src.texto import limpar_texto

MODELO = Path(__file__).resolve().parent / "models" / "modelo_sentimento.joblib"

modelo = joblib.load(MODELO)

textos = sys.argv[1:] or [
    "Chegou antes do prazo, produto excelente!",
    "Não recebi o produto até agora e ninguém responde",
    "veio faltando uma peça, péssimo",
    "não é ruim, mas esperava mais",
]

for texto in textos:
    limpo = limpar_texto(texto)
    rotulo = modelo.predict([limpo])[0]
    if hasattr(modelo, "predict_proba"):
        prob = modelo.predict_proba([limpo])[0][1]
        extra = f" (prob. positivo: {prob:.0%})"
    else:
        extra = ""
    print(f"[{'POSITIVO' if rotulo == 1 else 'NEGATIVO'}]{extra}  {texto}")
