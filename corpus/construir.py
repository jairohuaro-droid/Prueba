"""Fase 3: construye el corpus de evaluación.

Grupos:
  tesis      Resúmenes de tesis de magíster de la Universidad de Chile
             (repositorio.uchile.cl, OAI-PMH), defendidas en 2021 o antes,
             es decir, antes de ChatGPT. Humanos.
  gemelos    Un resumen escrito por IA (Claude) para cada tesis, con el mismo
             título y área. Están en corpus/gemelos_ia.jsonl.
  iberautex  Muestra en español de IberAuTexTification (IberLEF 2024), textos
             humanos y generados por seis modelos, ya etiquetados.
  clasicos   Fragmentos de obras de dominio público en es.wikisource.org.

Los textos descargados se guardan en datos/ (fuera de git: las tesis y
IberAuTexTification no permiten redistribuirlos). En git solo queda
corpus/fuentes.csv con la procedencia de cada texto.

Uso:
    python -m corpus.construir tesis|iberautex|clasicos|unir|todo
"""

import csv
import html
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
CORPUS = RAIZ / "corpus"
SEMILLA = 42
MAX_PALABRAS = 300
UA = {"User-Agent": "detector-ia-prueba/0.1 (investigacion academica)"}

OAI_UCHILE = "https://repositorio.uchile.cl/oai/request"
PAGINAS_OAI = 60
TESIS_POR_AREA = 5
AREAS = [
    ("salud", r"salud|m[eé]dic|enfermer|cl[ií]nic|psicolog|obesidad|nutri|farmac|hospital|odontol|epidemi|pacient"),
    ("educacion", r"educaci|enseñanza|escuela|docente|aprendizaje|pedag|curr[ií]cul|estudiantes"),
    ("derecho", r"derecho|jur[ií]d|ley |leyes|penal|constituc|tribunal|contrat"),
    ("economia", r"econom|negocio|empresa|mercado|finan|gesti[oó]n|marketing|precio|comerc"),
    ("ingenieria", r"ingenier|miner|el[eé]ctric|algoritm|sistemas|dise[ñn]o|optimiz|material|energ|computac|software"),
    ("humanidades", r"literat|novela|poes|histori|filosof|arte|m[uú]sica|cine|ling[uü]|narrativ|memoria"),
    ("sociales", r"social|pol[ií]tic|trabajador|g[eé]nero|comunidad|p[uú]blic|ciudad|urban|migra|sociolog"),
]

IBERAUTEX = "https://huggingface.co/datasets/Genaios/iberautextification/resolve/main/data/subtask_1/"
IBERAUTEX_POR_GRUPO = 25  # por dominio y etiqueta
IBERAUTEX_DOMINIOS = [("train.tsv", "news"), ("train.tsv", "wikipedia"),
                      ("train.tsv", "literary"), ("test.tsv", "chat")]

CLASICOS = [
    ("Gustavo Adolfo Bécquer", "Los ojos verdes"),
    ("Gustavo Adolfo Bécquer", "El rayo de luna"),
    ("Gustavo Adolfo Bécquer", "Maese Pérez el organista"),
    ("Leopoldo Alas «Clarín»", "¡Adiós, Cordera!"),
    ("Horacio Quiroga", "La gallina degollada"),
    ("Rubén Darío", "El rey burgués"),
    ("Benito Pérez Galdós", "Marianela/I"),
]

# Desde Project Gutenberg: (autor, título, n.º de libro, línea con la que empieza).
CLASICOS_GUTENBERG = [
    ("Miguel de Cervantes", "Don Quijote, capítulo primero", 2000,
     "Capítulo primero. Que trata de la condición"),
]

_STOP_ES = set("de la que el en y los las del por para con una se como es su al".split())
_ORACION = re.compile(r"(?<=[.!?…])\s+")


# ---------------------------------------------------------------- utilidades

def descargar(url, reintentos=6):
    for i in range(reintentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or i == reintentos - 1:
                raise
            espera = int(e.headers.get("Retry-After") or 0) or 15 * 2 ** i
            print(f"  {e.code}, reintento en {espera}s", file=sys.stderr)
            time.sleep(espera)


def en_espanol(texto):
    pals = re.findall(r"\w+", texto.lower())
    return bool(pals) and sum(p in _STOP_ES for p in pals) / len(pals) > 0.2


def recortar(texto, max_palabras=MAX_PALABRAS):
    """Oraciones completas hasta max_palabras (al menos una oración)."""
    texto = re.sub(r"\s+", " ", texto).strip()
    salida, n = [], 0
    for oracion in _ORACION.split(texto):
        m = len(oracion.split())
        if salida and n + m > max_palabras:
            break
        salida.append(oracion)
        n += m
    return " ".join(salida)


def escribir_jsonl(ruta, filas):
    ruta.parent.mkdir(exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"{ruta.relative_to(RAIZ)}: {len(filas)} textos")


def leer_jsonl(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


# --------------------------------------------------------------------- tesis

def cosechar_oai(base, paginas, cache):
    if cache.exists():
        return leer_jsonl(cache)
    url, registros = f"{base}?verb=ListRecords&metadataPrefix=oai_dc", []
    for _ in range(paginas):
        xml = descargar(url).decode("utf-8", "replace")
        for r in re.findall(r"<record>(.*?)</record>", xml, re.S):
            campos = {}
            for tag, val in re.findall(r"<dc:(\w+)[^>]*>(.*?)</dc:\1>", r, re.S):
                campos.setdefault(tag, []).append(html.unescape(val))
            registros.append(campos)
        token = re.search(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>", xml)
        if not token:
            break
        url = f"{base}?verb=ListRecords&resumptionToken={urllib.parse.quote(token.group(1))}"
        time.sleep(1)
    escribir_jsonl(cache, registros)
    return registros


def tesis():
    registros = cosechar_oai(OAI_UCHILE, PAGINAS_OAI, DATOS / "oai_uchile.jsonl")
    por_area = {}
    for r in registros:
        if "Tesis" not in r.get("type", []):
            continue
        titulo = r.get("title", [""])[0]
        if not re.search(r"mag[ií]ster", " ".join(r.get("description", []) + [titulo]), re.I):
            continue
        if re.search(r"\b(the|of|and|in)\b", titulo.lower()):
            continue
        # Solo el año de defensa (4 dígitos); las demás fechas son de carga.
        anios = [int(d) for d in r.get("date", []) if re.fullmatch(r"\d{4}", d)]
        if not anios or max(anios) > 2021:
            continue
        resumenes = [d for d in r.get("description", [])
                     if 120 <= len(d.split()) <= 600 and en_espanol(d)]
        if not resumenes:
            continue
        materias = r.get("subject", [])
        clave = (titulo + " " + " ".join(materias)).lower()
        area = next((a for a, patron in AREAS if re.search(patron, clave)), "ciencias")
        url = next((i for i in r.get("identifier", []) if "handle" in i), "")
        por_area.setdefault(area, []).append({
            "id": "tesis-" + url.rsplit("/", 1)[-1],
            "grupo": "tesis", "etiqueta": "humano", "area": area,
            "titulo": re.sub(r"\s+", " ", titulo).strip(), "anio": max(anios),
            "fuente": "Universidad de Chile", "url": url,
            "licencia": "; ".join(r.get("rights", [])),
            "texto": recortar(resumenes[0]),
        })
    rnd = random.Random(SEMILLA)
    elegidas = []
    for area in sorted(por_area):
        elegidas += rnd.sample(por_area[area], min(TESIS_POR_AREA, len(por_area[area])))
    escribir_jsonl(DATOS / "tesis.jsonl", elegidas)


# ----------------------------------------------------------------- iberautex

def iberautex():
    csv.field_size_limit(10 ** 9)
    rnd = random.Random(SEMILLA)
    filas, tablas = [], {}
    for archivo, dominio in IBERAUTEX_DOMINIOS:
        if archivo not in tablas:
            ruta = DATOS / f"iberautex_{archivo}"
            if not ruta.exists():
                ruta.parent.mkdir(exist_ok=True)
                ruta.write_bytes(descargar(IBERAUTEX + archivo))
            with open(ruta, encoding="utf-8") as f:
                tablas[archivo] = [x for x in csv.DictReader(f, delimiter="\t")
                                   if x["language"] == "es"]
        for etiqueta in ("human", "generated"):
            pool = [x for x in tablas[archivo] if x["domain"] == dominio
                    and x["label"] == etiqueta and 120 <= len(x["text"].split()) <= 600]
            if etiqueta == "generated":
                # Mismo número de textos de cada modelo (anónimos: A–F).
                modelos = sorted({x["model"] for x in pool})
                muestra = []
                for i, m in enumerate(modelos):
                    cupo = IBERAUTEX_POR_GRUPO // len(modelos) + (i < IBERAUTEX_POR_GRUPO % len(modelos))
                    muestra += rnd.sample([x for x in pool if x["model"] == m], cupo)
            else:
                muestra = rnd.sample(pool, IBERAUTEX_POR_GRUPO)
            for x in muestra:
                filas.append({
                    "id": f"iberautex-{archivo[:-4]}-{x['id']}", "grupo": "iberautex",
                    "etiqueta": "humano" if etiqueta == "human" else "ia",
                    "area": dominio, "modelo": x["model"],
                    "fuente": "IberAuTexTification 2024",
                    "url": "https://huggingface.co/datasets/Genaios/iberautextification",
                    "licencia": "CC BY-NC-ND 4.0", "texto": recortar(x["text"]),
                })
    escribir_jsonl(DATOS / "iberautex.jsonl", filas)


# ------------------------------------------------------------------ clasicos

def wikisource(titulo):
    q = urllib.parse.urlencode(dict(action="parse", page=titulo, prop="text", redirects=1,
                                    format="json", formatversion=2))
    d = json.loads(descargar("https://es.wikisource.org/w/api.php?" + q))
    if "error" in d:
        return None
    h = re.sub(r"<(style|script|table|sup)[^>]*>.*?</\1>", "", d["parse"]["text"], flags=re.S)
    parrafos = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip()
                for p in re.findall(r"<p>(.*?)</p>", h, re.S)]
    # El primer párrafo suele perder la letra capital; se descarta.
    parrafos = [p for p in parrafos if len(p.split()) >= 8][1:]
    return " ".join(parrafos)


def gutenberg(numero, inicio):
    texto = descargar(f"https://www.gutenberg.org/cache/epub/{numero}/pg{numero}.txt").decode("utf-8").replace("\r\n", "\n")
    texto = texto[texto.index(inicio):]
    # Se salta la línea del título del capítulo.
    return texto.split("\n\n", 1)[1]


def clasicos():
    fuentes = [(a, t, "Wikisource", "https://es.wikisource.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                lambda t=t: wikisource(t)) for a, t in CLASICOS]
    fuentes += [(a, t, "Project Gutenberg", f"https://www.gutenberg.org/ebooks/{n}",
                 lambda n=n, i=i: gutenberg(n, i)) for a, t, n, i in CLASICOS_GUTENBERG]
    filas = []
    for autor, titulo, fuente, url, obtener in fuentes:
        try:
            texto = obtener()
        except urllib.error.HTTPError as e:
            print(f"  error {e.code}: {titulo}", file=sys.stderr)
            texto = None
        time.sleep(5)
        if not texto or len(texto.split()) < 150:
            print(f"  omitido: {titulo}", file=sys.stderr)
            continue
        filas.append({
            "id": "clasico-" + re.sub(r"\W+", "-", titulo.lower()).strip("-"),
            "grupo": "clasicos", "etiqueta": "humano", "area": "literatura",
            "titulo": titulo, "autor": autor, "fuente": fuente, "url": url,
            "licencia": "Dominio público", "texto": recortar(texto),
        })
    escribir_jsonl(DATOS / "clasicos.jsonl", filas)


# ---------------------------------------------------------------------- unir

def unir():
    filas = []
    for nombre in ("tesis", "iberautex", "clasicos"):
        ruta = DATOS / f"{nombre}.jsonl"
        if ruta.exists():
            filas += leer_jsonl(ruta)
    gemelos = CORPUS / "gemelos_ia.jsonl"
    if gemelos.exists():
        ids = {f["id"] for f in filas}
        for g in leer_jsonl(gemelos):
            if g["gemelo_de"] in ids:
                filas.append(g)
    escribir_jsonl(DATOS / "corpus.jsonl", filas)
    campos = ["id", "grupo", "etiqueta", "area", "modelo", "titulo", "autor", "anio",
              "gemelo_de", "palabras", "fuente", "url", "licencia"]
    with open(CORPUS / "fuentes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({**fila, "palabras": len(fila["texto"].split())})


PASOS = {"tesis": tesis, "iberautex": iberautex, "clasicos": clasicos, "unir": unir}

if __name__ == "__main__":
    paso = sys.argv[1] if len(sys.argv) > 1 else "todo"
    for nombre in (PASOS if paso == "todo" else [paso]):
        PASOS[nombre]()
