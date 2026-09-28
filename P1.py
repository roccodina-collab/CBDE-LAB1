import psycopg2
import json
import time
import statistics
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "dbname": "vectorlab1",
    "user": "postgres",
    "password": "810808",
    "host": "localhost",
    "port": "5432"
}

# 1. Connectar-se 
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 2. Llegir totes les files (id + text) de la taula
cur.execute("SELECT id, text FROM sentences")
rows = cur.fetchall()  # llista de tuples (id, text)
print(f"{len(rows)} files llegides de la base de dades.")

# 3. Carregar el model UNA SOLA VEGADA, fora del bucle
print("Carregant model all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Bucle: generar embedding + actualitzar BD + mesurar temps
update_times = []

for row in rows:
    fila_id, text = row  # "desempaquetar" la tupla en dues variables

    t0 = time.perf_counter()

    embedding = model.encode(text)          # genera l'embedding de la frase
    embedding_list = embedding.tolist()      # numpy array -> llista de Python

    cur.execute(
        "UPDATE sentences SET embedding = %s WHERE id = %s",
        (embedding_list, fila_id)
    )
    conn.commit()

    t1 = time.perf_counter()
    update_times.append(t1 - t0)

    if fila_id % 1000 == 0:
        print(f"Fila {fila_id} processada...")

# 5. Estadístiques 
print("\n--- Temps de generació + inserció d'EMBEDDINGS ---")
print(f"Min:    {min(update_times):.6f} s")
print(f"Max:    {max(update_times):.6f} s")
print(f"Mitjana:{statistics.mean(update_times):.6f} s")
print(f"Stdev:  {statistics.stdev(update_times):.6f} s")

cur.close()
conn.close()
print("\nFet. Connexió tancada.")
