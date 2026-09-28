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
