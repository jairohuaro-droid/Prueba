"""Revisa un PDF y devuelve una copia con los tramos que parecen IA resaltados.

El texto se divide en tramos de ~200 palabras (párrafos seguidos de la misma
sección), a cada tramo se le calcula Binoculars y se compara con umbrales
calibrados en tesis humanas (resultados/umbrales.json):

  rojo      más "predecible" que el 99 % de los tramos de tesis humanas
  amarillo  más "predecible" que el 95 %
  sin color dentro de lo normal para un texto académico humano

Se omiten portada, índices, tablas, pies de página y la bibliografía.
El resultado es un indicio estadístico, no una prueba.

Uso:
    python -m detector.revisar tesis.pdf [salida.pdf]
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
UMBRALES = RAIZ / "resultados" / "umbrales.json"
PALABRAS_TRAMO = 200
MIN_PALABRAS_TRAMO = 80
MIN_PALABRAS_BLOQUE = 12
COLORES = {"alto": (0.96, 0.42, 0.38), "medio": (1.0, 0.85, 0.3)}
NOMBRES = {"alto": "Fuerte", "medio": "Moderado", "bajo": "Sin indicios"}

_FIN_BIBLIO = re.compile(r"^\s*(referencias|bibliograf[ií]a|referencias bibliogr[aá]ficas|"
                         r"anexos?|ap[eé]ndices?)\s*$", re.I)
_INDICE = re.compile(r"(\.\s?){4,}\s*\d+\s*$|^\s*(índice|tabla de contenidos?|contenido)\s*$", re.I | re.M)


@dataclass
class Bloque:
    pagina: int
    rect: tuple
    texto: str


@dataclass
class Tramo:
    bloques: list = field(default_factory=list)
    senales: dict = None
    nivel: str = "bajo"

    @property
    def texto(self):
        return " ".join(b.texto for b in self.bloques)

    @property
    def palabras(self):
        return len(self.texto.split())

    @property
    def paginas(self):
        return sorted({b.pagina + 1 for b in self.bloques})


def es_prosa(texto):
    """Descarta tablas, índices, fórmulas y listas de números."""
    pals = texto.split()
    if len(pals) < MIN_PALABRAS_BLOQUE or _INDICE.search(texto):
        return False
    letras = sum(c.isalpha() for c in texto)
    digitos = sum(c.isdigit() for c in texto)
    return letras > 0.6 * len(texto.replace(" ", "")) and digitos < 0.15 * letras


def extraer_bloques(doc):
    bloques = []
    for n, pagina in enumerate(doc):
        alto = pagina.rect.height
        for x0, y0, x1, y1, texto, *_ in pagina.get_text("blocks", sort=True):
            texto = re.sub(r"-\n(?=[a-záéíóúñ])", "", texto)
            texto = re.sub(r"\s+", " ", texto).strip()
            if _FIN_BIBLIO.match(texto) and n > len(doc) * 0.5:
                return bloques
            # Encabezados y pies de página.
            if y1 < alto * 0.07 or y0 > alto * 0.93:
                continue
            if es_prosa(texto):
                bloques.append(Bloque(n, (x0, y0, x1, y1), texto))
            elif 0 < len(texto.split()) < MIN_PALABRAS_BLOQUE and re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{3}", texto):
                bloques.append(None)  # probable título: corta el tramo
    return bloques


def agrupar(bloques):
    """Tramos de ~PALABRAS_TRAMO palabras que no cruzan títulos (None).
    Un resto corto se suma al tramo anterior de la misma sección; si la
    sección entera es corta, se descarta (poca evidencia)."""
    tramos, actual, en_seccion = [], Tramo(), 0

    def cerrar():
        nonlocal actual, en_seccion
        if actual.bloques:
            if actual.palabras >= MIN_PALABRAS_TRAMO:
                tramos.append(actual)
                en_seccion += 1
            elif en_seccion:
                tramos[-1].bloques += actual.bloques
        actual = Tramo()

    for b in bloques + [None]:
        if b is None:
            cerrar()
            en_seccion = 0
            continue
        actual.bloques.append(b)
        if actual.palabras >= PALABRAS_TRAMO:
            cerrar()
    return tramos


def clasificar(tramos, umbrales):
    for t in tramos:
        b = t.senales["binoculars"]
        if b <= umbrales["alto"]:
            t.nivel = "alto"
        elif b <= umbrales["medio"]:
            t.nivel = "medio"


def resumen(tramos):
    total = sum(t.palabras for t in tramos) or 1
    return {nivel: round(100 * sum(t.palabras for t in tramos if t.nivel == nivel) / total, 1)
            for nivel in ("alto", "medio", "bajo")}


def portada(doc, tramos, umbrales, nombre):
    import pymupdf
    pct = resumen(tramos)
    pagina = doc.new_page(0, width=595, height=842)
    y = 60

    def linea(texto, tam=10.5, negrita=False, color=(0.1, 0.1, 0.1), sangria=0):
        nonlocal y
        fuente = "hebo" if negrita else "helv"
        caja = pymupdf.Rect(56 + sangria, y, 540, y + 400)
        sobra = pagina.insert_textbox(caja, texto, fontsize=tam, fontname=fuente, color=color)
        y += 400 - sobra + tam * 0.5

    linea("Revisión de posibles tramos generados por IA", 17, True)
    linea(nombre, 10, color=(0.35, 0.35, 0.35))
    y += 6
    palabras = f"{sum(t.palabras for t in tramos):,}".replace(",", ".")
    linea(f"Texto analizado: {palabras} palabras en {len(tramos)} tramos "
          f"(se omiten portada, índices, tablas y bibliografía).")
    y += 4
    for nivel, texto in [("alto", "Indicio fuerte (rojo)"), ("medio", "Indicio moderado (amarillo)"),
                         ("bajo", "Sin indicios")]:
        if nivel in COLORES:
            pagina.draw_rect(pymupdf.Rect(56, y + 1, 68, y + 11), color=None, fill=COLORES[nivel])
        linea(f"{texto}: {str(pct[nivel]).replace('.', ',')} % del texto", 11, nivel != "bajo", sangria=18)
    y += 8
    linea("Cómo leer este informe", 12, True)
    linea("Cada tramo de unas 200 palabras se comparó con más de 300 resúmenes de tesis de magíster "
          "escritos por personas antes de 2022. Rojo significa que el tramo es más predecible que el "
          f"99 % de esos textos humanos (Binoculars de {umbrales['alto']:.3f} o menos); amarillo, que lo es más "
          f"que el 95 % ({umbrales['medio']:.3f} o menos).")
    linea(f"En las pruebas del proyecto, con estos umbrales se marcó en amarillo o rojo el "
          f"{umbrales['sensibilidad_medio']:.0%} de los textos académicos generados por IA y el "
          f"{umbrales['fpr_medio_prueba']:.0%} de los escritos por personas.")
    y += 4
    linea("Importante", 12, True)
    linea("Esto es un indicio estadístico, no una prueba. Un texto humano muy formal, muy revisado o "
          "escrito siguiendo plantillas puede salir marcado, y un texto de IA editado a mano puede no "
          "salir. Los tramos marcados son un punto de partida para revisar, no una conclusión. "
          "No debe usarse para sancionar a nadie.", color=(0.45, 0.1, 0.1))
    marcados = [t for t in tramos if t.nivel != "bajo"]
    if marcados:
        y += 4
        linea("Tramos marcados", 12, True)
        for t in marcados[:25]:
            pags = ", ".join(map(str, t.paginas))
            linea(f"Pág. {pags}: {NOMBRES[t.nivel]} (Binoculars {t.senales['binoculars']:.3f}): "
                  f"«{' '.join(t.texto.split()[:14])}...»", 9.5, sangria=10)
            if y > 780:
                break


def resaltar(doc, tramos):
    for t in tramos:
        if t.nivel == "bajo":
            continue
        for b in t.bloques:
            pagina = doc[b.pagina]  # mantener la referencia mientras se edita
            anot = pagina.add_highlight_annot(_lineas(pagina, b.rect))
            anot.set_colors(stroke=COLORES[t.nivel])
            anot.set_info(title="Detector de IA",
                          content=f"{NOMBRES[t.nivel]} · Binoculars {t.senales['binoculars']:.3f}")
            anot.update()


def _lineas(pagina, rect):
    """Rectángulos de cada línea del bloque, para un resaltado limpio."""
    import pymupdf
    zona = pymupdf.Rect(rect)
    rects = []
    for bloque in pagina.get_text("dict", clip=zona)["blocks"]:
        for linea in bloque.get("lines", []):
            r = pymupdf.Rect(linea["bbox"])
            if r.intersects(zona):
                rects.append(r)
    return rects or [zona]


def revisar(entrada, salida=None, evaluador=None, umbrales=None):
    import pymupdf
    entrada = Path(entrada)
    salida = Path(salida) if salida else entrada.with_name(entrada.stem + "_revisado.pdf")
    umbrales = umbrales or json.loads(UMBRALES.read_text())
    doc = pymupdf.open(entrada)
    tramos = agrupar(extraer_bloques(doc))
    if not tramos:
        raise ValueError("No se encontró texto seleccionable (¿es un PDF escaneado?).")
    if evaluador is None:
        from detector.perplejidad import Evaluador
        evaluador = Evaluador(umbrales["observador"], umbrales["ejecutor"])
    for i, t in enumerate(tramos, 1):
        t.senales = evaluador.analizar(t.texto)
        print(f"\r{i}/{len(tramos)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    clasificar(tramos, umbrales)
    resaltar(doc, tramos)
    portada(doc, tramos, umbrales, entrada.name)
    doc.save(salida, garbage=3, deflate=True)
    return salida, tramos


def main(argv):
    salida, tramos = revisar(argv[1], argv[2] if len(argv) > 2 else None)
    print(json.dumps({"salida": str(salida), "porcentajes": resumen(tramos),
                      "tramos": [{"paginas": t.paginas, "palabras": t.palabras, "nivel": t.nivel,
                                  "binoculars": t.senales["binoculars"]} for t in tramos]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
