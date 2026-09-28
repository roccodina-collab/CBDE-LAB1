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

# 1. Connexió a PostgreSQL
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 2. Carregar el model per codificar les consultes
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Els mateixos 10 IDs utilitzats a P2 i C2
query_ids = [56, 72, 33, 598, 711, 678, 979, 110, 222, 396]

query_times = []

print("=== INICIANDO CONSULTAS G2 (Pgvector Nativo con HNSW) ===")

for qid in query_ids:
    # Obtenir el text i l'embedding de la frase de consulta
    cur.execute("SELECT text, embedding FROM sentences_pgvector WHERE id = %s", (qid,))
    res = cur.fetchone()
    if not res:
        continue
    
    q_text, q_vec_str = res

    t0 = time.perf_counter()

    # Consulta Top-2 per Distància Euclidiana (operador <->) excloent la pròpia frase
    cur.execute("""
        SELECT id, text, embedding <-> %s AS dist
        FROM sentences_pgvector
        WHERE id != %s
        ORDER BY embedding <-> %s
        LIMIT 2;
    """, (q_vec_str, qid, q_vec_str))
    top2_l2 = cur.fetchall()

    # Consulta Top-2 per Distància Cosinus (operador <=>) excloent la pròpia frase
    cur.execute("""
        SELECT id, text, embedding <=> %s AS dist_cosine
        FROM sentences_pgvector
        WHERE id != %s
        ORDER BY embedding <=> %s
        LIMIT 2;
    """, (q_vec_str, qid, q_vec_str))
    top2_cosine = cur.fetchall()

    t1 = time.perf_counter()
    query_times.append(t1 - t0)

    print(f"\n--- Consulta Pgvector (ID={qid}): {q_text} ---")
    print("  Top-2 L2 (Euclidiana) <->:")
    for r in top2_l2:
        print(f"    id={r[0]} | dist={r[2]:.4f} | {r[1]}")
    print("  Top-2 Cosinus <==>:")
    for r in top2_cosine:
        print(f"    id={r[0]} | dist_cos={r[2]:.4f} | {r[1]}")

# 4. Estadístiques de temps per a G2
print("\n--- Tiempos de consulta G2 (Pgvector Top-2, 10 frases) ---")
print(f"Min:     {min(query_times):.6f} s")
print(f"Max:     {max(query_times):.6f} s")
print(f"Mitjana: {statistics.mean(query_times):.6f} s")
print(f"Stdev:   {statistics.stdev(query_times):.6f} s")

cur.close()
conn.close()
