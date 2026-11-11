"""
Funções de limpeza de texto para as avaliações em português.

Decisão importante: NÃO removo palavras de negação ("não", "nunca", "nem"...).
Em análise de sentimento, "não chegou" e "chegou" têm sentidos opostos, e
listas de stopwords prontas costumam apagar o "não".
"""

import re
import unicodedata

import pandas as pd

# stopwords básicas, montada à mão a partir das palavras mais frequentes do dataset
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "para", "pra", "pro",
    "com", "e", "que", "se", "ao", "aos", "eu", "me", "meu", "minha", "ele", "ela",
    "isso", "esse", "essa", "este", "esta", "foi", "ser", "sao", "ja", "ate",
    "mais", "muito", "q", "vc", "estou", "esta", "tem", "ter",
}

NEGACOES = {"nao", "nunca", "nem", "nada", "ninguem", "jamais"}
STOPWORDS -= NEGACOES

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_NAO_LETRA = re.compile(r"[^a-z\s]")
RE_ESPACOS = re.compile(r"\s+")


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpar_texto(texto) -> str:
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = RE_URL.sub(" ", t)
    t = remover_acentos(t)
    t = RE_NAO_LETRA.sub(" ", t)   # tira números, pontuação e emojis
    palavras = [p for p in t.split() if p not in STOPWORDS and len(p) > 1]
    return RE_ESPACOS.sub(" ", " ".join(palavras)).strip()


def montar_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir do CSV de reviews da Olist, monta a base de treino:
    - junta título + comentário (o título às vezes resume bem o sentimento)
    - remove quem não escreveu nada
    - nota 1-2 = negativo (0), nota 4-5 = positivo (1), nota 3 sai (neutra/ambígua)
    """
    base = df.copy()
    titulo = base["review_comment_title"].fillna("")
    msg = base["review_comment_message"].fillna("")
    base["texto"] = (titulo + " " + msg).str.strip()
    base = base[base["texto"].str.len() > 0]

    base = base[base["review_score"] != 3]
    base["sentimento"] = (base["review_score"] >= 4).astype(int)

    base["texto_limpo"] = base["texto"].map(limpar_texto)
    base = base[base["texto_limpo"].str.len() > 0]
    base = base.drop_duplicates(subset=["texto_limpo", "sentimento"])
    return base[["review_id", "review_score", "texto", "texto_limpo", "sentimento"]].reset_index(drop=True)
