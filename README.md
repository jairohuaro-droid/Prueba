# Prueba
Repositrio de prueba

## Detector de uso de IA

Ver `docs/fundamentos.md`. Pruebas: `python3 -m unittest discover -s tests -t .`

Comparar los textos de ejemplo:

```
python3 -m detector.heuristicas ejemplos/humano.txt      # fase 1, sin internet
pip install -r requirements.txt
python3 -m detector.perplejidad ejemplos/humano.txt ejemplos/ia.txt   # fase 2
```

La fase 2 descarga Qwen2.5-0.5B y Qwen2.5-0.5B-Instruct (~2 GB) desde
huggingface.co, así que ese dominio tiene que estar permitido en la red.

Fase 3 (prueba con 288 textos reales): ver `docs/resultados_fase3.md`.

```
python3 -m corpus.construir todo
python3 -m corpus.evaluar --recalcular
```

## Revisar un PDF

```
python3 -m detector.revisar tesis.pdf            # crea tesis_revisado.pdf
```

Resalta en rojo/amarillo los tramos de ~200 palabras que son más predecibles
que el 99 %/95 % de las tesis humanas de calibración, y añade una portada con
el resumen. Usa Qwen2.5-1.5B y los umbrales de `resultados/umbrales.json`
(`python3 -m corpus.calibrar 1.5B --guardar`). Es un indicio, no una prueba.
