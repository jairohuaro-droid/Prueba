"""Fase 2: señales estadísticas con un modelo de lenguaje.

Calcula perplejidad, burstiness (variación de la perplejidad entre
oraciones), log-rank / fracción top-10 (GLTR), entropía media y la
puntuación Binoculars (dos modelos que comparten tokenizador).

Requiere: pip install -r requirements.txt

Uso:
    python -m detector.perplejidad archivo.txt
    echo "texto" | python -m detector.perplejidad

Los umbrales todavía no están calibrados; eso se hace en la fase 3.
"""

import json
import math
import re
import statistics
import sys

OBSERVADOR = "Qwen/Qwen2.5-0.5B"
EJECUTOR = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 512

_FIN_ORACION = re.compile(r"(?<=[.!?…])\s+")


def spans_oraciones(texto):
    """Posiciones (inicio, fin) en caracteres de cada oración."""
    spans, inicio = [], 0
    for m in _FIN_ORACION.finditer(texto):
        spans.append((inicio, m.start()))
        inicio = m.end()
    spans.append((inicio, len(texto)))
    return [(a, b) for a, b in spans if texto[a:b].strip()]


class Evaluador:
    def __init__(self, observador=OBSERVADOR, ejecutor=EJECUTOR, dispositivo=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(observador)
        self.observador = self._cargar(AutoModelForCausalLM, observador)
        self.ejecutor = self._cargar(AutoModelForCausalLM, ejecutor) if ejecutor else None

    def _cargar(self, clase, nombre):
        modelo = clase.from_pretrained(nombre, torch_dtype="auto").to(self.dispositivo)
        modelo.eval()
        return modelo

    def analizar(self, texto):
        torch = self.torch
        texto = texto.strip()
        cod = self.tokenizer(texto, return_tensors="pt", return_offsets_mapping=True,
                             truncation=True, max_length=MAX_TOKENS)
        ids = cod["input_ids"].to(self.dispositivo)
        offsets = cod["offset_mapping"][0].tolist()
        if ids.shape[1] < 2:
            raise ValueError("Texto demasiado corto para analizar.")

        with torch.no_grad():
            logits_obs = self.observador(ids).logits[0, :-1].float()
            logits_eje = self.ejecutor(ids).logits[0, :-1].float() if self.ejecutor else None
        objetivo = ids[0, 1:, None]

        logp = torch.log_softmax(logits_obs, dim=-1)
        logp_objetivo = logp.gather(1, objetivo)
        nll = -logp_objetivo[:, 0]
        rangos = (logp > logp_objetivo).sum(1) + 1
        entropia = -(logp.exp() * logp).sum(1)

        senales = {
            "tokens": int(ids.shape[1]),
            "perplejidad": round(math.exp(nll.mean().item()), 3),
            "burstiness": self._burstiness(texto, offsets, nll.tolist()),
            "log_rank_medio": round(torch.log(rangos.float()).mean().item(), 4),
            "fraccion_top10": round((rangos <= 10).float().mean().item(), 4),
            "entropia_media": round(entropia.mean().item(), 4),
            "binoculars": None,
        }
        if logits_eje is not None:
            # Binoculars: log-perplejidad del ejecutor dividida por la
            # entropía cruzada observador→ejecutor. Más bajo = más "IA".
            logp_eje = torch.log_softmax(logits_eje, dim=-1)
            log_ppl = -logp_eje.gather(1, objetivo).mean()
            x_ppl = -(logp.exp() * logp_eje).sum(1).mean()
            senales["binoculars"] = round((log_ppl / x_ppl).item(), 4)
        return senales

    @staticmethod
    def _burstiness(texto, offsets, nll):
        """Coeficiente de variación de la perplejidad por oración."""
        spans = spans_oraciones(texto)
        por_oracion = [[] for _ in spans]
        # nll[i] corresponde al token i + 1 (el primero no tiene predicción).
        for i, valor in enumerate(nll):
            pos = offsets[i + 1][0]
            k = next((k for k, (_, fin) in enumerate(spans) if pos < fin), len(spans) - 1)
            por_oracion[k].append(valor)
        ppls = [math.exp(statistics.mean(v)) for v in por_oracion if len(v) >= 3]
        if len(ppls) < 2:
            return None
        return round(statistics.pstdev(ppls) / statistics.mean(ppls), 4)


def main(argv):
    texto = open(argv[1], encoding="utf-8").read() if len(argv) > 1 else sys.stdin.read()
    print(json.dumps(Evaluador().analizar(texto), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)
