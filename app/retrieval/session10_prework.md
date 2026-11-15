## Resumen

Ataca el problema de "similar ≠ relevante" en la recuperación con dos técnicas, y mide si compensan:

1. **Búsqueda híbrida**: rama léxica (full-text en PostgreSQL) + rama vectorial existente, fusionadas con Reciprocal Rank Fusion.
2. **Reranking** con cross-encoder siguiendo el patrón recall-then-rerank (top-50 → top-5).
3. **Golden set** de 5 consultas anotadas a mano y evaluación de las configuraciones A-D.

## Cambios

| Paso | Qué | Dónde |
|---|---|---|
| 1 | Columna generada `content_tsv` (`to_tsvector('english', content)`, STORED) + índice GIN | `migrations/versions/0a616b6e9003_*.py`, `app/models.py` |
| 2 | Rama léxica (`ts_rank`, consulta en OR), rama vectorial y RRF | `app/retrieval/{lexical,vector,fusion,service,types}.py` |
| 3 | Wrapper de cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU) + script de verificación | `app/retrieval/reranker.py`, `app/retrieval/verify_reranker.py` |
| 4 | Golden set + script de evaluación A-D vía API | `data/golden_set.json`, `scripts/evaluate_retrieval.py`, `retrieval_evaluation.md` |

`POST /search` acepta ahora `mode` (`vector` | `hybrid`) y `rerank` (bool) por petición. Si no se envían, usa `SEARCH_MODE` y `RERANK_ENABLED` del `.env`, así que el reranking se activa y desactiva sin tocar código. La respuesta incluye `timings_ms` por etapa (embed / search / rerank / total) y `score` en lugar de `distance` (más alto = mejor, en cualquier modo).

## Decisiones

- **Text search config `'english'` y no `'spanish'`**: en este repo los presupuestos del dataset y el texto de los chunks están en inglés. Con `'spanish'` las stopwords y el stemming serían incorrectos (`payments`, `authentication` no se normalizarían bien). Por coherencia, las consultas del golden set están en inglés. Si el corpus pasa a español, basta con una migración que cambie la config.
- **Consulta léxica en OR**: `plainto_tsquery` une los términos con AND, y con descripciones de proyecto largas casi nunca hay un chunk que los contenga todos (0 resultados). Se reescribe a OR y `ts_rank` premia a los chunks con más coincidencias.
- **RRF con k=60** (valor del paper original), configurable con `RRF_K`. Solo usa posiciones, así que la similitud coseno y `ts_rank` se combinan sin normalizar.
- **Relevancia anotada por presupuesto**: un chunk devuelto es un acierto si su `budget_id` está en `relevant_budget_ids`. Cada consulta tiene ≥5 chunks relevantes (precisión@5 máxima = 1,0) y una "trampa" deliberada (un presupuesto parecido en el espacio vectorial pero inútil para estimar).
- **La evaluación pasa por la API** (sin `--reload`): mide la latencia que vería un cliente real, con el modelo ejecutándose dentro del servidor.

## Resultados

5 consultas · top_k=5 · recall_k=50 · rrf_k=60 · 5 ejecuciones medidas por consulta tras calentar.

| Config | Búsqueda | Reranking | Precisión@5 | Latencia p50 (ms) | Latencia p95 (ms) | Rerank p50 (ms) |
|---|---|---|---|---|---|---|
| A | Vectorial | No | 0,84 | 232 | 276 | – |
| B | Híbrida | No | **0,88** | 232 | 264 | – |
| C | Vectorial | Sí | 0,84 | 3740 | 3892 | 3505 |
| D | Híbrida | Sí | 0,84 | 3976 | 4528 | 3723 |

Detalle por consulta y top-5 de cada configuración en `retrieval_evaluation.md`.

## Conclusiones

**Configuración elegida: B (híbrida sin reranking).** Obtiene la mejor precisión@5 (0,88 frente a 0,84) con la misma latencia que la vectorial pura (p50 de 232 ms en ambas; la rama léxica solo añade unos 3 ms de base de datos). La mejora es pequeña, un chunk de 25, pero el coste es prácticamente nulo, y la rama léxica protege frente a lo que los embeddings diluyen: términos exactos como PSD2, FHIR o nombres de tecnologías, que pesarán más cuando crezca el corpus.

**El reranking no compensa en este caso:** multiplica la latencia por unas 16 (de ~230 ms a ~3,7–4,0 s) sin mejorar la precisión, y en q4 y q1 promociona falsos amigos (detección de fraude, devoluciones de moda), porque un modelo genérico entrenado con MS MARCO juzga si un pasaje responde a una pregunta, no si dos proyectos son comparables para estimarlos. Además, con 58 chunks y `recall_k=50`, el reranker reordena casi todo el corpus, así que C y D son idénticas. En este caso de uso la latencia por sí sola no sería decisiva (la generación con el LLM ya tarda varios segundos); lo decisivo es que no hay ganancia que pagar. Revisaría la decisión con un corpus de cientos de presupuestos y un golden set más difícil, probando un reranker multilingüe o ajustado al dominio, con `recall_k≈20` y un modelo cuantizado o en ONNX.

## Limitaciones y puntos abiertos

- **Corpus pequeño (17 presupuestos, 58 chunks) y golden set fácil**: la vectorial ya está cerca del máximo, así que la diferencia A→B (1 chunk de 25) no es estadísticamente significativa.
- **q4 se queda en 0,6 en todas las configuraciones**: ninguna recupera BUD-2024-015. La anotación de 015 como relevante (ingesta industrial de series temporales + forecasting) es discutible. No se ha cambiado a posteriori para no ajustar el examen a los resultados.
- **Latencia del reranker dentro del servidor (~3,5 s) frente a fuera (~1,3 s)** con los mismos 50 chunks. Se ha descartado que sea por el threadpool o por el número de hilos de torch. Causa pendiente de investigar.
- **Bug previo en el chunker** (`\\n` en lugar de `\n`): los chunks contienen el texto literal `\n` y el índice léxico genera tokens como `ndescription`. No se corrige aquí porque exige reingerir el corpus.
