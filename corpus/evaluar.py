"""Fase 3: evalúa las señales del detector sobre el corpus.

1. Calcula las señales de las fases 1 y 2 para cada texto de
   datos/corpus.jsonl y las guarda en resultados/senales.csv (sin textos).
2. Mide qué tan bien separa cada señal los textos humanos de los de IA
   (AUROC) en cada grupo.
3. Entrena un combinador (regresión logística) con la mitad de los datos,
   fija un umbral que acuse a lo sumo al 5 % de los humanos de esa mitad y
   lo prueba en la otra mitad y en los clásicos.

Uso:
    python -m corpus.construir todo
    python -m corpus.evaluar            # reutiliza senales.csv si existe
    python -m corpus.evaluar --recalcular
"""

import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

from corpus.construir import DATOS, RAIZ, SEMILLA, leer_jsonl
from detector import heuristicas

RESULTADOS = RAIZ / "resultados"
FPR_OBJETIVO = 0.05

# Señal -> signo: +1 si un valor alto apunta a IA, -1 si un valor bajo.
SENALES = {
    "perplejidad": -1,
    "binoculars": -1,
    "log_rank_medio": -1,
    "fraccion_top10": +1,
    "entropia_media": -1,
    "burstiness": -1,
    "variacion_largo_oracion": -1,
    "marcadores_por_100_palabras": +1,
    "largo_medio_oracion": +1,
    "type_token_ratio": -1,
    "puntuacion_heuristica": +1,
}
COMBINADOR = ["perplejidad", "binoculars", "log_rank_medio", "fraccion_top10",
              "entropia_media", "burstiness", "variacion_largo_oracion",
              "marcadores_por_100_palabras", "largo_medio_oracion", "type_token_ratio"]
CAMPOS_ID = ["id", "grupo", "etiqueta", "area", "modelo", "gemelo_de", "palabras"]


# ------------------------------------------------------------------ señales

def calcular_senales(corpus, ruta):
    from detector.perplejidad import Evaluador
    evaluador = Evaluador()
    filas = []
    for i, doc in enumerate(corpus, 1):
        h = heuristicas.analizar(doc["texto"])
        fila = {c: doc.get(c, "") for c in CAMPOS_ID}
        fila["palabras"] = h["senales"]["palabras"]
        fila.update({k: h["senales"][k] for k in ("variacion_largo_oracion",
                     "marcadores_por_100_palabras", "largo_medio_oracion", "type_token_ratio")})
        fila["puntuacion_heuristica"] = h["puntuacion_ia"]
        p = evaluador.analizar(doc["texto"])
        fila.update({k: p[k] for k in ("perplejidad", "binoculars", "log_rank_medio",
                                       "fraccion_top10", "entropia_media", "burstiness")})
        filas.append(fila)
        print(f"\r{i}/{len(corpus)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    RESULTADOS.mkdir(exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return leer_csv(ruta)


def leer_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    for fila in filas:
        fila["palabras"] = int(fila["palabras"])
        for k in SENALES:
            fila[k] = float(fila[k]) if fila[k] not in ("", "None") else None
    return filas


# ------------------------------------------------------------------ métricas

def auroc(pos, neg):
    """Probabilidad de que un texto de IA puntúe más alto que uno humano."""
    if not pos or not neg:
        return None
    total = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return total / (len(pos) * len(neg))


def puntajes(filas, senal):
    signo = SENALES[senal]
    ia = [signo * f[senal] for f in filas if f["etiqueta"] == "ia" and f[senal] is not None]
    hum = [signo * f[senal] for f in filas if f["etiqueta"] == "humano" and f[senal] is not None]
    return ia, hum


# ---------------------------------------------------------------- combinador

def vector(fila, medias, desvs):
    x = []
    for k in COMBINADOR:
        v = fila[k] if fila[k] is not None else medias[k]
        if k == "perplejidad":
            v = math.log(v)
        x.append((v - medias[k]) / desvs[k])
    return x


def entrenar(filas, epocas=3000, tasa=0.1, l2=0.01):
    medias, desvs = {}, {}
    for k in COMBINADOR:
        vals = [math.log(f[k]) if k == "perplejidad" else f[k] for f in filas if f[k] is not None]
        medias[k] = statistics.mean(vals)
        desvs[k] = statistics.pstdev(vals) or 1.0
    # La media de perplejidad se guarda en escala logarítmica; se corrige
    # para rellenar faltantes antes de aplicar el logaritmo.
    X = [vector(f, {**medias, "perplejidad": math.exp(medias["perplejidad"])}, desvs) for f in filas]
    y = [1.0 if f["etiqueta"] == "ia" else 0.0 for f in filas]
    w, b = [0.0] * len(COMBINADOR), 0.0
    for _ in range(epocas):
        gw, gb = [l2 * wi for wi in w], 0.0
        for xi, yi in zip(X, y):
            p = 1 / (1 + math.exp(-(sum(a * c for a, c in zip(w, xi)) + b)))
            for j, xj in enumerate(xi):
                gw[j] += (p - yi) * xj / len(X)
            gb += (p - yi) / len(X)
        w = [wi - tasa * g for wi, g in zip(w, gw)]
        b -= tasa * gb
    modelo = {"pesos": dict(zip(COMBINADOR, w)), "sesgo": b, "medias": medias, "desvs": desvs}
    return modelo


def probabilidad(modelo, fila):
    medias = {**modelo["medias"], "perplejidad": math.exp(modelo["medias"]["perplejidad"])}
    x = vector(fila, medias, modelo["desvs"])
    z = sum(modelo["pesos"][k] * v for k, v in zip(COMBINADOR, x)) + modelo["sesgo"]
    return 1 / (1 + math.exp(-z))


def umbral_para_fpr(humanos, fpr):
    """Menor umbral que deja como máximo `fpr` de humanos por encima."""
    orden = sorted(humanos, reverse=True)
    k = int(math.floor(fpr * len(orden)))
    return orden[k] + 1e-9 if k < len(orden) else min(orden)


def dividir(filas):
    """Mitad de calibración y mitad de prueba; los pares tesis/gemelo van
    juntos para que el combinador no vea el tema del texto de prueba."""
    rnd = random.Random(SEMILLA)
    llaves = sorted({f["gemelo_de"] or f["id"] for f in filas if f["grupo"] != "clasicos"})
    rnd.shuffle(llaves)
    calib = set(llaves[: len(llaves) // 2])
    a = [f for f in filas if f["grupo"] != "clasicos" and (f["gemelo_de"] or f["id"]) in calib]
    b = [f for f in filas if f["grupo"] != "clasicos" and (f["gemelo_de"] or f["id"]) not in calib]
    return a, b


# ---------------------------------------------------------------------- main

def tasa(xs, umbral):
    return sum(x >= umbral for x in xs) / len(xs) if xs else None


def main(argv):
    ruta = RESULTADOS / "senales.csv"
    if "--recalcular" in argv or not ruta.exists():
        filas = calcular_senales(leer_jsonl(DATOS / "corpus.jsonl"), ruta)
    else:
        filas = leer_csv(ruta)

    grupos = {
        "tesis_vs_gemelos": [f for f in filas if f["grupo"] in ("tesis", "gemelos")],
        "iberautex": [f for f in filas if f["grupo"] == "iberautex"],
        "todo": [f for f in filas if f["grupo"] != "clasicos"],
    }
    metricas = {"n": {g: len(v) for g, v in grupos.items()}, "auroc": {}}
    metricas["n"]["clasicos"] = sum(f["grupo"] == "clasicos" for f in filas)
    for senal in SENALES:
        metricas["auroc"][senal] = {g: round(auroc(*puntajes(v, senal)), 3) for g, v in grupos.items()}
    for dom in sorted({f["area"] for f in grupos["iberautex"]}):
        sub = [f for f in grupos["iberautex"] if f["area"] == dom]
        metricas["auroc"].setdefault("binoculars_por_dominio", {})[dom] = round(auroc(*puntajes(sub, "binoculars")), 3)

    # Promedios por grupo y etiqueta, para explicar los resultados.
    metricas["medianas"] = {}
    for g in ("tesis", "gemelos", "iberautex", "clasicos"):
        for et in ("humano", "ia"):
            sub = [f for f in filas if f["grupo"] == g and f["etiqueta"] == et]
            if sub:
                metricas["medianas"][f"{g}/{et}"] = {
                    k: round(statistics.median(f[k] for f in sub if f[k] is not None), 3)
                    for k in ("perplejidad", "binoculars", "fraccion_top10", "palabras")}

    calib, prueba = dividir(filas)
    modelo = entrenar(calib)
    p_hum_cal = [probabilidad(modelo, f) for f in calib if f["etiqueta"] == "humano"]
    umbral = umbral_para_fpr(p_hum_cal, FPR_OBJETIVO)
    res = {"umbral": round(umbral, 4), "n_calibracion": len(calib), "n_prueba": len(prueba)}
    for nombre, sub in [("prueba_todo", prueba),
                        ("prueba_tesis", [f for f in prueba if f["grupo"] in ("tesis", "gemelos")]),
                        ("prueba_iberautex", [f for f in prueba if f["grupo"] == "iberautex"]),
                        ("clasicos", [f for f in filas if f["grupo"] == "clasicos"])]:
        ia = [probabilidad(modelo, f) for f in sub if f["etiqueta"] == "ia"]
        hum = [probabilidad(modelo, f) for f in sub if f["etiqueta"] == "humano"]
        res[nombre] = {"auroc": round(auroc(ia, hum), 3) if ia and hum else None,
                       "ia_detectada": round(tasa(ia, umbral), 3) if ia else None,
                       "humanos_acusados": round(tasa(hum, umbral), 3) if hum else None,
                       "n_ia": len(ia), "n_humanos": len(hum)}
    # Umbral de Binoculars sola, con el mismo criterio.
    ub = umbral_para_fpr([-f["binoculars"] for f in calib if f["etiqueta"] == "humano"], FPR_OBJETIVO)
    res["binoculars_sola"] = {
        "umbral": round(-ub, 4),
        "ia_detectada": round(tasa([-f["binoculars"] for f in prueba if f["etiqueta"] == "ia"], ub), 3),
        "humanos_acusados": round(tasa([-f["binoculars"] for f in prueba if f["etiqueta"] == "humano"], ub), 3),
        "clasicos_acusados": round(tasa([-f["binoculars"] for f in filas if f["grupo"] == "clasicos"], ub), 3),
    }
    metricas["combinador"] = res
    metricas["combinador_pesos"] = {k: round(v, 3) for k, v in modelo["pesos"].items()}

    with open(RESULTADOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    with open(RESULTADOS / "combinador.json", "w", encoding="utf-8") as f:
        json.dump({**modelo, "umbral": umbral, "fpr_objetivo": FPR_OBJETIVO}, f, indent=2)
    print(json.dumps(metricas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)
