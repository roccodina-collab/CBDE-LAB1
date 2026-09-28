import psycopg2
import numpy as np
import time
import statistics

DB_CONFIG = {
    "dbname": "vectorlab1",
    "user": "postgres",
    "password": "810808",
    "host": "localhost",
    "port": "5432"
}

# 1. Conexión
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 2. Cargar datos en memoria desde PostgreSQL
cur.execute("SELECT id, text, embedding FROM sentences")
rows = cur.fetchall()

all_ids = [r[0] for r in rows]
all_texts = [r[1] for r in rows]
all_embeddings = np.array([r[2] for r in rows], dtype=np.float32)

# Normalizar vectores una sola vez para acelerar el producto escalar (similitud coseno)
norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
all_embeddings_norm = all_embeddings / norms

query_ids = [56, 72, 33, 598, 711, 678, 979, 110, 222, 396]

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    return np.dot(a, b)

query_times = []

print("=== INICIANDO CONSULTAS P2 (PostgreSQL Data in Memory) ===")

for qid in query_ids:
    t0 = time.perf_counter()

    idx_query = all_ids.index(qid)
    query_vector = all_embeddings[idx_query]
    query_vector_norm = all_embeddings_norm[idx_query]

    euclidean_scores = []
    cosine_scores = []

    for i in range(len(all_ids)):
        if all_ids[i] == qid:
            continue  # Excluir la propia frase de la consulta

        dist = euclidean_distance(query_vector, all_embeddings[i])
        sim = cosine_similarity(query_vector_norm, all_embeddings_norm[i])

        euclidean_scores.append((all_ids[i], all_texts[i], dist))
        cosine_scores.append((all_ids[i], all_texts[i], sim))

    top2_euclidean = sorted(euclidean_scores, key=lambda x: x[2])[:2]
    top2_cosine = sorted(cosine_scores, key=lambda x: x[2], reverse=True)[:2]

    t1 = time.perf_counter()
    query_times.append(t1 - t0)

    print(f"\n--- Consulta (ID={qid}): {all_texts[idx_query]} ---")
    print("  Top-2 L2 (Euclidiana):")
    for r in top2_euclidean:
        print(f"    id={r[0]} | dist={r[2]:.4f} | {r[1]}")
    print("  Top-2 Coseno:")
    for r in top2_cosine:
        print(f"    id={r[0]} | sim={r[2]:.4f} | {r[1]}")

print("\n--- Tiempos de consulta P2 (Top-2, 10 frases) ---")
print(f"Min:     {min(query_times):.6f} s")
print(f"Max:     {max(query_times):.6f} s")
print(f"Mitjana: {statistics.mean(query_times):.6f} s")
print(f"Stdev:   {statistics.stdev(query_times):.6f} s")

cur.close()
conn.close()

