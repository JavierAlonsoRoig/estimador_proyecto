# estimador_proyecto

uv run uvicorn app.main:app --reload
uv run streamlit run app/streamlit_app.py

http://127.0.0.1:8000/docs
## Decisiones del schema

**Dos tablas (`documents` y `chunks`) en vez de una.** Un presupuesto produce N chunks. Con una sola tabla, la metadata del documento se repetiría en cada fila y no habría integridad referencial. Con dos tablas y una FK con `ON DELETE CASCADE`, borrar un presupuesto borra automáticamente todos sus chunks.

**`source_path` único.** La referencia solo pone un índice normal; usamos `UNIQUE`. Así no se puede ingerir dos veces el mismo fichero, y la restricción ya crea el índice para buscar por ruta. Para reingerir un documento, se borra y se vuelve a insertar: el `CASCADE` se encarga de los chunks antiguos.

**`metadata` JSONB en ambas tablas.** La metadata estable (tipo de documento, tipo de chunk, fechas) va en columnas tipadas. La variable, o la que el chunker puede enriquecer (tags, alcance, tecnologías mencionadas), va en JSONB con `'{}'` por defecto. El índice GIN sobre `chunks.metadata` permite filtrar por cualquier clave sin migrar el schema cada vez.

**`token_count` en `chunks`.** Es un añadido nuestro. Guardar el número de tokens al ingerir permite controlar el presupuesto de contexto al montar el prompt sin volver a tokenizar cada chunk en cada consulta.

**`vector(1536)`.** Es la dimensión de `text-embedding-3-small`. Va fija en el schema (`EMBEDDING_DIM` en `app/models.py`) porque cambiarla obliga a volver a calcular los embeddings de todo el corpus; no es algo que vaya a cambiar en caliente.

**`embedding` nullable.** Permite insertar un chunk y calcular su embedding después, por ejemplo si la llamada a la API falla. En este ejercicio ingerimos chunk y embedding a la vez, en la misma transacción, pero así queda preparada la ingesta asíncrona de sesiones posteriores.

**Índices sobre `chunks`.** Hay uno en `document_id` para la FK y los borrados en cascada, otro en `chunk_type` porque filtraremos por tipo, y el GIN sobre `metadata`.

**Sin índice vectorial.** Es deliberado: con pocos chunks, la búsqueda exacta es suficiente. El índice (HNSW o IVFFlat) se añade en el directo.

**Extensión `vector`.** La primera migración ejecuta `CREATE EXTENSION IF NOT EXISTS vector` antes de crear las tablas. La imagen `pgvector/pgvector` trae la extensión instalada, pero hay que activarla en cada base de datos.
