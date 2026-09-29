"""Calibra los umbrales de detector/revisar.py con texto académico.

Humanos: resúmenes de tesis (datos/tesis.jsonl y datos/tesis_extra.jsonl).
IA: corpus/gemelos_ia.jsonl (Claude) y corpus/salamandra_ia.jsonl.
Todos se recortan a ~200 palabras, el tamaño de los tramos que se revisan.

Los humanos se dividen en dos mitades: con una se fijan los umbrales (percentil
1 y 5 de Binoculars) y con la otra se mide cuántos humanos quedarían marcados.
Los textos de IA no intervienen en los umbrales; solo miden la sensibilidad.

Uso:
    python -m corpus.calibrar 0.5B            # calcula y muestra
    python -m corpus.calibrar 1.5B --guardar  # además escribe resultados/umbrales.json
"""

import csv
import json
import random
import sys

from corpus.construir import CORPUS, DATOS, SEMILLA, leer_jsonl, recortar
from corpus.evaluar import RESULTADOS, auroc
from detector.revisar import PALABRAS_TRAMO

MODELOS = {
    "0.5B": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "1.5B": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
}


def textos():
    filas = []
    for nombre in ("tesis", "tesis_extra"):
        filas += leer_jsonl(DATOS / f"{nombre}.jsonl")
    for nombre in ("gemelos_ia", "salamandra_ia"):
        ruta = CORPUS / f"{nombre}.jsonl"
        if ruta.exists():
            filas += [g for g in leer_jsonl(ruta) if len(g["texto"].split()) >= 100]
    for f in filas:
        f["texto"] = recortar(f["texto"], PALABRAS_TRAMO + 20)
    return filas


def senales(etiqueta_modelo):
    ruta = RESULTADOS / f"calibracion_{etiqueta_modelo}.csv"
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        for f in filas:
            f["binoculars"] = float(f["binoculars"])
            f["perplejidad"] = float(f["perplejidad"])
        return filas
    from detector.perplejidad import Evaluador
    evaluador = Evaluador(*MODELOS[etiqueta_modelo])
    filas = []
    corpus = textos()
    for i, t in enumerate(corpus, 1):
        s = evaluador.analizar(t["texto"])
        filas.append({"id": t["id"], "grupo": t["grupo"], "etiqueta": t["etiqueta"],
                      "palabras": len(t["texto"].split()),
                      "binoculars": s["binoculars"], "perplejidad": s["perplejidad"]})
        print(f"\r{i}/{len(corpus)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    RESULTADOS.mkdir(exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return filas


def percentil(valores, p):
    orden = sorted(valores)
    return orden[max(0, int(p * len(orden)) - 1)] if p * len(orden) >= 1 else orden[0] - 1e-9


def calibrar(filas):
    humanos = [f for f in filas if f["etiqueta"] == "humano"]
    rnd = random.Random(SEMILLA)
    rnd.shuffle(humanos)
    mitad = len(humanos) // 2
    cal, prueba = humanos[:mitad], humanos[mitad:]
    b_cal = [f["binoculars"] for f in cal]
    alto, medio = percentil(b_cal, 0.01), percentil(b_cal, 0.05)
    ia = [f for f in filas if f["etiqueta"] == "ia"]
    tasa = lambda xs, u: sum(x["binoculars"] <= u for x in xs) / len(xs)
    por_grupo = {g: {"n": sum(f["grupo"] == g for f in ia),
                     "alto": round(tasa([f for f in ia if f["grupo"] == g], alto), 3),
                     "medio": round(tasa([f for f in ia if f["grupo"] == g], medio), 3)}
                 for g in sorted({f["grupo"] for f in ia})}
    return {
        "alto": round(alto, 4), "medio": round(medio, 4),
        "n_humanos_calibracion": len(cal), "n_humanos_prueba": len(prueba), "n_ia": len(ia),
        "fpr_alto_prueba": round(tasa(prueba, alto), 3),
        "fpr_medio_prueba": round(tasa(prueba, medio), 3),
        "sensibilidad_alto": round(tasa(ia, alto), 3),
        "sensibilidad_medio": round(tasa(ia, medio), 3),
        "sensibilidad_por_grupo": por_grupo,
        "auroc": round(auroc([-f["binoculars"] for f in ia], [-f["binoculars"] for f in humanos]), 3),
    }


def main(argv):
    etiqueta = argv[1] if len(argv) > 1 else "0.5B"
    res = calibrar(senales(etiqueta))
    res["observador"], res["ejecutor"] = MODELOS[etiqueta]
    res["palabras_tramo"] = PALABRAS_TRAMO
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if "--guardar" in argv:
        (RESULTADOS / "umbrales.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv)
