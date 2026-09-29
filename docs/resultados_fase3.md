# Fase 3: prueba con textos reales

Fecha: septiembre de 2026. Modelo: Qwen2.5-0.5B (observador) y
Qwen2.5-0.5B-Instruct (ejecutor), en CPU.

## Qué textos se usaron (288)

| Grupo | Qué es | Humanos | IA |
|---|---|---|---|
| Tesis | Resúmenes de tesis de magíster de la Universidad de Chile, **defendidas en 2021 o antes** (antes de ChatGPT), 5 por área en 8 áreas | 40 | — |
| Gemelos | Un resumen escrito por IA (Claude) para cada tesis, **con el mismo título y la misma extensión**, sin ver el original | — | 40 |
| IberAuTexTification | Conjunto de investigación (IberLEF 2024) en español: noticias, Wikipedia, literatura y chat, con textos de 6 modelos | 100 | 100 |
| Clásicos | Bécquer, Clarín, Quiroga, Darío, Galdós y el Quijote (dominio público) | 8 | — |

Cada texto se recortó a un máximo de 300 palabras, cortando en un final de
oración. La procedencia de cada uno está en `corpus/fuentes.csv`. Los textos
de las tesis y de IberAuTexTification **no** se guardan en el repositorio
porque sus licencias no lo permiten. Se descargan con
`python -m corpus.construir todo`.

## Resultados en simple

**1. Las señales del modelo funcionan; las de estilo, poco.**
La tabla dice con qué frecuencia la señal pone un texto de IA por delante de
uno humano (50 % = azar, 100 % = perfecto).

| Señal | Tesis vs. gemelos | IberAuTexTification | Total |
|---|---|---|---|
| Perplejidad (sorpresa del modelo) | **90 %** | 81 % | 82 % |
| Binoculars | 88 % | **84 %** | **84 %** |
| Log-rank | 89 % | 82 % | 83 % |
| Palabras entre las 10 más probables | 84 % | 81 % | 82 % |
| Entropía | 79 % | 77 % | 77 % |
| Variación del largo de las frases (fase 1) | 67 % | 68 % | 68 % |
| Puntuación de estilo de la fase 1 | 67 % | 64 % | 65 % |
| Frases típicas de IA ("cabe destacar"…) | 49 % | 54 % | 52 % |
| Burstiness (fase 2) | 59 % | 42 % | 47 % |

- La lista de frases típicas de IA **no sirve**: acierta como una moneda al
  aire. Ni los resúmenes de IA las usaban más, ni las tesis humanas menos.
- La *burstiness* (cambios de sorpresa entre frases) tampoco distingue.

**2. En pares del mismo tema, casi siempre acierta cuál es la IA.**
En 37 de los 40 pares, el resumen de IA salió más "predecible" que la tesis
real con el mismo título.

**3. Pero no hay una línea clara entre humano e IA.**
Si se fija el umbral para acusar como mucho a 1 de cada 20 humanos, el
detector solo atrapa a **un tercio de los textos de IA** (34 % en la mitad
de prueba). En las tesis pasa lo mismo: con un umbral de Binoculars que acusa
a 1 de las 40 tesis, solo se atrapa el 35 % de los gemelos. Si se quiere atrapar más, sube el número de
humanos acusados por error. El umbral elegido con la mitad de los datos, al
probarlo en la otra mitad, acusó al 8,6 % de los humanos, algo más que el 5 %
buscado: con tan pocos textos, el umbral todavía es inestable.

**4. Falla en textos tipo chat.**
Por tipo de texto, Binoculars acierta el 98 % en Wikipedia, el 96 % en
noticias y el 95 % en literatura, pero solo el **48 % en chat** (azar). Los
textos "humanos" de chat de ese conjunto son respuestas de voluntarios que
imitan a un asistente, y el modelo los encuentra tan predecibles como los de
IA.

**5. La trampa de los clásicos no se activó.**
Ningún clásico fue acusado. El castellano antiguo sorprende mucho al modelo
(perplejidad ~30, frente a ~14 en las tesis). El Quijote fue el clásico que
más se acercó a "IA" (Binoculars 0,955), pero quedó lejos del umbral (0,833).

**6. Combinar señales no mejoró.**
En la mitad de prueba, el combinador (regresión logística con 10 señales)
obtuvo un 83 %, prácticamente igual que Binoculars sola en esos mismos
textos (82,5 %).

## Conclusión

El detector sirve para decir que un texto es **más o menos probable** de
ser IA, sobre todo comparando textos del mismo tipo y tema. **No sirve para
acusar a nadie**: con un umbral prudente deja pasar dos de cada tres textos
de IA, y con un umbral agresivo acusa a demasiados humanos. Además, no funciona
con textos de estilo conversacional.

## Límites de esta prueba

- 288 textos es poco: los porcentajes pueden moverse varios puntos con otra
  muestra.
- Los 40 gemelos los escribió un solo modelo (Claude) con una sola
  instrucción. En IberAuTexTification hay 6 modelos, pero de 2023–2024.
- Las tesis son todas de una universidad chilena, y el área se asignó por
  palabras clave del título, así que algunas están mal clasificadas.
- Los textos de IA no fueron editados por personas; un texto de IA
  retocado a mano sería más difícil de detectar.

## Siguientes pasos sugeridos

1. Umbrales por tipo de texto (académico, periodístico, conversacional), en
   lugar de uno solo.
2. Probar un modelo observador más grande (p. ej. Qwen2.5-1.5B) para ver si
   mejora la separación.
3. Ampliar los gemelos con otros modelos y con textos de IA editados por
   personas.
4. Quitar o rehacer la lista de frases típicas de IA.

## Reproducir

```
pip install -r requirements.txt
python -m corpus.construir todo     # descarga y arma datos/corpus.jsonl
python -m corpus.evaluar --recalcular   # ~6 min en CPU
```

Archivos: `resultados/senales.csv` (señales por texto, sin el texto),
`resultados/metricas.json` y `resultados/combinador.json` (pesos y umbral).

## Calibración académica y modelo más grande (revisión de PDF)

Para revisar tesis se calibró solo con texto académico, recortado a ~200
palabras (el tamaño de los tramos que se resaltan):

- Humanos: 345 resúmenes de tesis de magíster (U. de Chile, 2021 o antes).
  Con la mitad se fijan los umbrales y con la otra mitad se mide cuántos
  humanos quedarían marcados.
- IA: los 40 gemelos (Claude) y 40 resúmenes generados con
  BSC-LT/salamandra-2b-instruct (`corpus/generar_ia.py`; de 60 generados,
  20 salieron con menos de 100 palabras y se descartaron).

| | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---|---|
| Acierto general (AUROC) | 92,0 % | **93,9 %** |
| Humanos marcados en amarillo o rojo | 3,5 % | **2,9 %** |
| Humanos marcados en rojo | **0,6 %** | 1,2 % |
| IA detectada (amarillo o rojo) | 67,5 % | 67,5 % |
| — textos de Claude | 40 % | **47,5 %** |
| — textos de Salamandra | **95 %** | 87,5 % |

Se eligió 1.5B: acusa a menos humanos en total y detecta mejor los textos del
modelo más reciente, que son los más difíciles. Umbrales en
`resultados/umbrales.json`.

En un PDF de prueba con 3 capítulos humanos y 3 de IA (Claude), la
herramienta marcó 1 capítulo de IA en rojo y ninguno humano, en línea con
lo esperado.
