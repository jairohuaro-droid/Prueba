"""Fase 1: señales heurísticas de texto generado por IA (sin dependencias).

Uso:
    python -m detector.heuristicas archivo.txt
    echo "texto" | python -m detector.heuristicas
"""

import json
import re
import statistics
import sys
import unicodedata

MARCADORES_IA = [
    "es importante destacar", "es importante señalar", "cabe destacar",
    "cabe señalar", "cabe mencionar", "en conclusión", "en resumen",
    "en el panorama actual", "en el mundo actual", "desempeña un papel crucial",
    "juega un papel crucial", "papel fundamental", "sumergirse en",
    "adentrarse en", "un tapiz de", "no solo", "sino también",
    "en última instancia", "a medida que", "es fundamental",
    "es esencial", "sin lugar a dudas", "en definitiva",
    "aquí tienes", "espero que esto te ayude", "como modelo de lenguaje",
    "delve", "tapestry", "testament", "multifaceted",
]

INVISIBLES = {"​", "‌", "‍", "⁠", "﻿", "­"}

_PALABRA = re.compile(r"\w+", re.UNICODE)
_ORACION = re.compile(r"(?<=[.!?…])\s+")


def oraciones(texto):
    return [o for o in _ORACION.split(texto.strip()) if _PALABRA.search(o)]


def palabras(texto):
    return [p.lower() for p in _PALABRA.findall(texto)]


def homoglifos(texto):
    """Letras no latinas mezcladas en palabras latinas (p. ej. 'а' cirílica)."""
    cuenta = 0
    for palabra in _PALABRA.findall(texto):
        scripts = {unicodedata.name(c, "").split(" ")[0] for c in palabra if c.isalpha()}
        if "LATIN" in scripts and len(scripts) > 1:
            cuenta += 1
    return cuenta


def analizar(texto):
    ors = oraciones(texto)
    pals = palabras(texto)
    n = len(pals)
    largos = [len(palabras(o)) for o in ors]
    bajo = texto.lower()

    media = statistics.mean(largos) if largos else 0.0
    desv = statistics.pstdev(largos) if len(largos) > 1 else 0.0
    marcadores = {m: bajo.count(m) for m in MARCADORES_IA if m in bajo}

    senales = {
        "palabras": n,
        "oraciones": len(ors),
        "largo_medio_oracion": round(media, 2),
        # Coeficiente de variación: proxy barato de "burstiness".
        "variacion_largo_oracion": round(desv / media, 3) if media else 0.0,
        "type_token_ratio": round(len(set(pals)) / n, 3) if n else 0.0,
        "marcadores_por_100_palabras": round(100 * sum(marcadores.values()) / n, 3) if n else 0.0,
        "rayas_por_100_palabras": round(100 * texto.count("—") / n, 3) if n else 0.0,
        "markdown": len(re.findall(r"\*\*|^#{1,6} |^\s*[-*] ", texto, re.MULTILINE)),
        "caracteres_invisibles": sum(texto.count(c) for c in INVISIBLES),
        "homoglifos": homoglifos(texto),
    }
    return {
        "senales": senales,
        "marcadores_encontrados": marcadores,
        "puntuacion_ia": puntuar(senales),
        "confiable": n >= 150,
    }


def puntuar(s):
    """Puntuación 0–1 provisional con pesos a mano; se reemplazará por un
    combinador entrenado en la fase 3."""
    puntos = 0.0
    if s["oraciones"] >= 3 and s["variacion_largo_oracion"] < 0.35:
        puntos += 0.25
    puntos += min(s["marcadores_por_100_palabras"] / 2, 1) * 0.35
    puntos += min(s["rayas_por_100_palabras"], 1) * 0.1
    puntos += min(s["markdown"] / 5, 1) * 0.1
    if s["caracteres_invisibles"] or s["homoglifos"]:
        puntos += 0.2
    return round(min(puntos, 1.0), 3)


def main(argv):
    texto = open(argv[1], encoding="utf-8").read() if len(argv) > 1 else sys.stdin.read()
    print(json.dumps(analizar(texto), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)
