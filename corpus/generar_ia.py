"""Genera resúmenes de tesis con un modelo abierto de otra familia
(BSC-LT/salamandra-2b-instruct), para títulos de datos/tesis_extra.jsonl.

Uso: python -m corpus.generar_ia [cantidad]
Salida: corpus/salamandra_ia.jsonl (se puede reanudar).
"""

import json
import random
import sys

from corpus.construir import CORPUS, DATOS, SEMILLA, leer_jsonl, recortar

MODELO = "BSC-LT/salamandra-2b-instruct"
INSTRUCCION = ("Escribe el resumen de una tesis de magíster de la Universidad de Chile "
               "titulada «{titulo}». Debe tener unas {n} palabras, en un solo párrafo, "
               "e incluir el problema, el objetivo, la metodología, los resultados y las "
               "conclusiones. Responde solo con el resumen, sin título.")


def main(argv):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cantidad = int(argv[1]) if len(argv) > 1 else 60
    salida = CORPUS / "salamandra_ia.jsonl"
    hechos = {g["gemelo_de"] for g in leer_jsonl(salida)} if salida.exists() else set()
    tesis = leer_jsonl(DATOS / "tesis_extra.jsonl")
    random.Random(SEMILLA).shuffle(tesis)
    tesis = tesis[:cantidad]

    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16).eval()
    torch.manual_seed(SEMILLA)
    with open(salida, "a", encoding="utf-8") as f:
        for i, t in enumerate(tesis, 1):
            if t["id"] in hechos:
                continue
            n = min(len(t["texto"].split()), 300)
            instruccion = INSTRUCCION.format(titulo=t["titulo"], n=n)
            ids = tok.apply_chat_template([{"role": "user", "content": instruccion}],
                                          add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)["input_ids"]
            with torch.no_grad():
                out = modelo.generate(ids, max_new_tokens=600, do_sample=True, temperature=0.7,
                                      top_p=0.9, repetition_penalty=1.1)
            texto = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({
                "id": "salamandra-" + t["id"], "grupo": "salamandra", "etiqueta": "ia",
                "area": t["area"], "titulo": t["titulo"], "gemelo_de": t["id"],
                "modelo": MODELO, "fuente": "Generado para este proyecto", "url": "",
                "licencia": "Texto generado por IA", "instruccion": instruccion,
                "texto": recortar(texto),
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i}/{len(tesis)} {len(texto.split())} palabras", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
