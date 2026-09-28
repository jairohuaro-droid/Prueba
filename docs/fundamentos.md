# Detector de uso de IA — Fundamentos

Documento de partida: qué hay que tener en cuenta para construir un detector de
texto generado por IA y qué señales ("reglas") usan los detectores
convencionales.

---

## 1. Factores principales a definir antes de construir

### 1.1 Alcance del problema
- **Qué se detecta**: texto 100 % generado, texto generado y luego editado por
  una persona, texto parafraseado por IA, texto mixto (partes humanas y partes
  IA), traducción automática.
- **Granularidad**: veredicto por documento, por párrafo o por oración. Para
  textos mixtos hace falta llegar al nivel de oración.
- **Idioma(s)**: casi todos los detectores comerciales están optimizados para
  inglés. Si el objetivo es español, los modelos y las listas de marcadores
  deben ser en español.
- **Dominio**: ensayos académicos, correos, código, artículos, redes sociales.
  Cada dominio tiene un "estilo humano" distinto.
- **Longitud mínima**: por debajo de ~150–250 palabras las señales
  estadísticas son muy ruidosas. Hay que definir un umbral y negarse a dar
  veredicto por debajo de él.

### 1.2 Datos
- Corpus **pareado**: mismo tema/prompt escrito por humanos y por varios
  modelos (GPT, Claude, Gemini, Llama, Mistral…), con distintas temperaturas.
- Texto humano **anterior a 2022** para garantizar que no está contaminado.
- Incluir escritores **no nativos**, estudiantes y textos formales: son los
  grupos con más falsos positivos.
- "Hard negatives": texto humano muy pulido y texto IA editado a mano.

### 1.3 Métricas de evaluación
- **Tasa de falsos positivos (FPR)** es la métrica crítica: acusar a una
  persona de usar IA es mucho más grave que dejar pasar un texto IA.
- Reportar **TPR a FPR = 1 %** además de AUROC.
- **Calibración**: que un "80 % IA" signifique realmente eso.
- Evaluar por separado cada modelo generador, cada dominio y cada idioma.

### 1.4 Robustez
Ataques habituales que el detector debe resistir (o al menos reconocer):
parafraseo (QuillBot, otra IA), edición humana ligera, traducción ida y
vuelta, homoglifos (letras cirílicas que parecen latinas), caracteres
invisibles, prompts del tipo "escribe como un estudiante con errores".

### 1.5 Ética y uso
- El resultado **no es prueba**; es una señal probabilística.
- Debe ser **explicable**: mostrar qué oraciones y qué señales pesaron.
- Nunca dar un veredicto binario sin nivel de confianza.

---

## 2. Reglas / señales que usan los detectores convencionales

### A. Señales estadísticas basadas en un modelo de lenguaje
Requieren pasar el texto por un LLM "observador" y mirar las probabilidades
de cada token.

| Señal | Idea | Usado por |
|---|---|---|
| **Perplejidad** | Los LLM eligen palabras muy probables → el texto IA tiene perplejidad baja. | GPTZero, casi todos |
| **Burstiness** | Los humanos alternan oraciones "fáciles" y "sorprendentes"; la IA es uniforme. Se mide como varianza de la perplejidad por oración. | GPTZero |
| **Log-rank / GLTR** | Rango de cada token en la predicción del modelo. La IA usa casi siempre tokens del top-10. | GLTR, Log-Rank |
| **Entropía** | Entropía de la distribución predicha en cada posición. | Varios |
| **Curvatura de probabilidad** | El texto IA está en un máximo local de probabilidad: perturbarlo baja mucho su log-prob; el humano no. | DetectGPT, Fast-DetectGPT |
| **Binoculars** | Cociente entre perplejidad y "cross-perplejidad" de dos modelos; sin entrenamiento, muy buen FPR. | Binoculars |
| **Regeneración** | Cortar el texto, pedir al LLM que lo continúe y medir solapamiento de n-gramas con el original. | DNA-GPT |

### B. Señales estilométricas (no necesitan LLM)
- Longitud media de oración y, sobre todo, su **desviación** (proxy barato de
  burstiness).
- **Riqueza léxica**: type-token ratio, MTLD, proporción de hapax legomena.
- Frecuencia de **palabras funcionales** (de, que, el, y…) y conectores.
- **Puntuación**: abuso de raya (—), dos puntos, punto y coma, listas con
  viñetas; escasez de puntos suspensivos, exclamaciones, paréntesis.
- **Legibilidad** (Fernández-Huerta / Szigriszt-Pazos en español): la IA
  tiende a un rango estrecho.
- **Repetición estructural**: estructuras paralelas, "regla de tres",
  párrafos de longitud casi idéntica.

### C. Marcadores léxicos y discursivos
- Frases muletilla: "es importante destacar", "cabe señalar", "en
  conclusión", "en el panorama actual", "desempeña un papel crucial",
  "sumergirse en", "un tapiz de", "no solo… sino también", "en resumen".
  (En inglés: *delve, tapestry, testament, multifaceted, navigate the
  complexities*.)
- Hedging y descargos: "es posible que", "depende de varios factores",
  "como modelo de lenguaje…".
- Estructura formulaica: introducción → 3–5 puntos con encabezado/negrita →
  conclusión que repite la introducción.
- Tono uniforme y neutro, **ausencia de errores** ortográficos, pocos
  detalles concretos (nombres, fechas, anécdotas personales).
- Restos de la interfaz: "¡Claro! Aquí tienes…", "Espero que esto te
  ayude", markdown pegado (`**`, `###`).

### D. Clasificadores supervisados
Un transformer (RoBERTa, DeBERTa, XLM-R para multilingüe) ajustado con
ejemplos etiquetados humano/IA. Es lo que usan Turnitin, Originality.ai,
Pangram, Copyleaks. Suelen ser los más precisos dentro de su dominio, pero
generalizan mal a modelos o dominios nuevos si no se reentrenan.

### E. Marcas de agua y metadatos
- **Watermarking** estadístico (lista verde/roja de tokens, SynthID de
  Google): solo detectable si el proveedor lo activó y se tiene la clave.
- **Caracteres invisibles** (zero-width space, espacios no estándar) y
  **homoglifos**.
- Metadatos de documento / C2PA.

### F. Señales de proceso (si hay acceso al editor)
Historial de versiones, velocidad de escritura, pegados masivos de texto,
tiempo total frente a longitud. Es de las señales más fiables cuando existe.

---

## 3. Arquitectura propuesta

```
texto ──► preprocesado ──► extractores de señales ──► combinador ──► reporte
          (normalizar,      A. estadísticas LLM       (regresión     (puntuación
           segmentar,       B. estilometría            logística /    calibrada +
           detectar         C. marcadores léxicos      boosting,      señales por
           idioma)          E. invisibles/homoglifos   calibrado)     oración)
```

Cada extractor devuelve un conjunto de *features* numéricas y evidencias
legibles; el combinador aprende los pesos con el corpus etiquetado.

## 4. Hoja de ruta

1. **Fase 1 – Heurísticas sin dependencias** (este commit): estilometría,
   marcadores léxicos en español, caracteres invisibles. Ver
   `detector/heuristicas.py`.
2. **Fase 2 – Señales con LLM**: perplejidad, burstiness y log-rank con un
   modelo pequeño multilingüe; luego Binoculars.
3. **Fase 3 – Datos y combinador**: construir el corpus pareado, entrenar el
   combinador y calibrarlo. Primera versión con 288 textos: ver
   `docs/resultados_fase3.md`.
4. **Fase 4 – Clasificador supervisado** (XLM-R) y evaluación de robustez.
5. **Fase 5 – Interfaz**: reporte con resaltado por oración.
