# Session 11 pre-work: line-level verifiable citations and RAGAS baseline

## Punto de partida y alcance

El enunciado da por hecho un generador estructurado de la sesión 9 (Responses API, estimación en JSON con citación). En este repo no existía: la generación devolvía texto libre con ejemplos fijos y `/search` terminaba en una lista de chunks. En este pre-work se construye la capa de augmentation + generación ya con citación por línea:

```
query ──► retrieve() (hybrid, sesión 10) ──► assemble_context() ──► gpt-4.1-mini (responses.parse → Estimate) ──► verify_citations()
```

Fuera de alcance (no implementado en este repo): reformulación de la transcripción, expansión de consultas, routing multi-índice y filtrado temporal. Las consultas del golden set hacen de consulta ya reformulada.

## Parte 1 — Citación verificable por línea

### Decisiones

- **`hours: float | None`**: una línea con `grounded=false` lleva `hours=null`. Con un campo obligatorio, el modelo tendría que poner un número, y sin fuente cualquier número sería inventado.
- **La regla de integridad se comprueba después de generar, no en un `@model_validator`**: un validador haría fallar `responses.parse` entero por una sola línea incoherente. `verify_citations` informa línea a línea.
- **`<source chunk_id document_id sector year>`**: cada chunk llega al prompt con su id pegado al contenido. `chunk_id` es el id de la tabla `chunks`; `document_id` es el `budget_id` (p. ej. `BUD-2024-005`).
- **`verify_citations` recibe el contexto ensamblado, no solo un `set` de ids**: así detecta tres formas de cita falsa, no una.

### Estados de cada línea

| Estado | Significado | ¿Cuentan sus horas? |
|---|---|---|
| `grounded` | todas sus fuentes estaban en el contexto, con su `document_id` correcto, y la `evidence` aparece literalmente en el chunk | sí |
| `dangling_citation` | cita un `chunk_id` que no se recuperó, o un chunk real con el presupuesto equivocado | no |
| `unverified_evidence` | la fuente existe, pero lo copiado en `evidence` no está en ella | no |
| `insufficient_data` | marcada `grounded=false`, sin horas ni fuentes (correcto) | — |
| `integrity_violation` | rompe la regla grounded ↔ fuentes/horas | no |

"Rechazar" significa que las horas de esa línea no cuentan en `verified_hours`. La estimación se devuelve entera junto con el informe, para que el fallo quede visible. Todo se loguea con structlog y lleva el mismo `request_id` (middleware + `structlog.contextvars`; se acepta `X-Request-ID` del cliente).

### Pruebas

`app/tests/test_citations.py`: 7 tests deterministas, sin LLM (válida, id inventado, chunk real con documento equivocado, evidence que no está en la fuente, sin datos suficientes, y dos violaciones de integridad).

### Informe de verificación sobre estimaciones reales

Generado con `scripts/generate_estimate.py` (hybrid, top_k=8, gpt-4.1-mini):

**(a) q1-fashion-store, estimación real**

```
[grounded            ] Product catalog with search and filters (150h)
[grounded            ] Shopping cart (24h)
[grounded            ] Card checkout (40h)
[grounded            ] Order confirmation emails (12h)
clean: True | verified hours: 226
```

**(b) la misma, con una citación colgante inyectada a propósito (`--inject-dangling`)**

```
[dangling_citation   ] Product catalog with search and filters (150h)
    ! chunk_id 999999 was not in the retrieved context
[grounded            ] Shopping cart (24h)
[grounded            ] Card checkout (40h)
[grounded            ] Order confirmation emails (12h)
clean: False | verified hours: 76
```

Log: `citation_rejected ... status=dangling_citation request_id=6ca0bca1d9034894af651da1edd1d68c`

**(c) petición con partes sin histórico** (gourmet shop + puntos de fidelización + recogida con QR)

```
[grounded            ] Product catalog (150h)
[grounded            ] Card checkout (40h)
[insufficient_data   ] Customer loyalty points program (-)
[insufficient_data   ] QR-code based in-store pickup (-)
clean: True | verified hours: 190
```

### Observaciones

- **La `evidence` es débil**: el modelo copia casi siempre `"Estimated hours: X"`. Es literal y verificable, pero respalda la cifra, no que el componente sea comparable.
- **Citar bien no es estimar bien**: en q1 todas las líneas están bien citadas, pero mezclan escalas: el catálogo sale de un headless empresarial (150 h) y el carrito, el checkout y el email de un MVP (24, 40 y 12 h). Ninguna verificación de citas detecta eso.

## Parte 2 — Evaluación RAGAS

### Configuración

- RAGAS 0.4.4, métricas de `ragas.metrics.collections` (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall).
- Generador `gpt-4.1-mini`; juez `gpt-4.1`, distinto y más capaz, para evitar el sesgo de autopreferencia; embeddings `text-embedding-3-small`.
- Golden set de la sesión 10 ampliado con `reference` (estimación de experto en texto) y `reference_hours`. Las referencias están ancladas a componentes y horas del corpus, para que `context_recall` mida la recuperación y no si la cifra ajustada del experto aparece en algún chunk.
- La respuesta se pasa a RAGAS como texto sin el total, porque el total lo calcula el código y el juez lo marcaría como afirmación no respaldada.

### Resultados

| Query | Faithfulness | Answer relevancy | Context precision | Context recall | Citations | Verified h | Reference h |
|---|---|---|---|---|---|---|---|
| q1-fashion-store | 0.92 | 0.67 | 1.00 | 0.57 | 4/4 grounded | 226 | 216 |
| q2-marketplace | 1.00 | 0.46 | 1.00 | 0.62 | 0/4 grounded, 4 unverified_evidence | 0 | 830 |
| q3-telehealth | 1.00 | 0.65 | 0.33 | 1.00 | 5/5 grounded | 790 | 850 |
| q4-predictive-maintenance | 1.00 | 0.77 | 0.83 | 0.80 | 3/3 grounded | 500 | 500 |
| q5-warehouse-logistics | 0.62 | 0.74 | 0.00 | 1.00 | 0/3 grounded, 3 unverified_evidence | 0 | 640 |
| **Average** | **0.91** | **0.66** | **0.63** | **0.80** | | | |


## Cómo reproducir

```bash
uv run pytest app/tests/test_citations.py -v
uv run python scripts/generate_estimate.py --golden q1-fashion-store
uv run python scripts/generate_estimate.py --golden q1-fashion-store --inject-dangling
uv run python scripts/evaluate_generation.py --output generation_evaluation.md
```
