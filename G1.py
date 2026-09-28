import psycopg2
import time
import statistics
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "dbname": "vectorlab1",
    "user": "postgres",
    "password": "contrasenya_DB",
    "host": "localhost",
    "port": "5432"
}

# 1. Connexió a la base de dades
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 2. Llegir totes les files (id + text)
cur.execute("SELECT id, text FROM sentences_pgvector")
rows = cur.fetchall()
print(f"{len(rows)} files llegides de la base de dades.")

# 3. Carregar el model Transformer
print("Carregant model all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Bucle: generar embedding, formatar-lo per a pgvector i actualitzar
update_times = []

for fila_id, text in rows:
    t0 = time.perf_counter()

    embedding = model.encode(text).tolist()
    
    # pgvector accepta el format d'string '[val1, val2, ...]' o la conversió de pgvector.python
    embedding_str = str(embedding)

    cur.execute(
        "UPDATE sentences_pgvector SET embedding = %s WHERE id = %s",
        (embedding_str, fila_id)
    )
    conn.commit()

    t1 = time.perf_counter()
    update_times.append(t1 - t0)

    if fila_id % 2000 == 0:
        print(f"Fila {fila_id} processada...")

# 5. Crear un índex HNSW sobre la columna de vectors per maximitzar el rendiment de cerca
print("\nCreant índex HNSW a Pgvector (distància cosinus)...")
cur.execute("""
    CREATE INDEX IF NOT EXISTS sentences_vec_hnsw_idx 
    ON sentences_pgvector 
    USING hnsw (embedding vector_cosine_ops);
""")
conn.commit()

# 6. Estadístiques de temps per a G1
print("\n--- G1: Temps de generació + inserció d'EMBEDDINGS a Pgvector ---")
print(f"Min:     {min(update_times):.6f} s")
print(f"Max:     {max(update_times):.6f} s")
print(f"Mitjana: {statistics.mean(update_times):.6f} s")
print(f"Stdev:   {statistics.stdev(update_times):.6f} s")

cur.close()
conn.close()
