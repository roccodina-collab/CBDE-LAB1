import psycopg2
import json
import time
import statistics

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

# 2. Habilitar l'extensió pgvector i crear la taula
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("""
    DROP TABLE IF EXISTS sentences_pgvector;
    CREATE TABLE sentences_pgvector (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding vector(384)
    );
""")
conn.commit()
print("Taula 'sentences_pgvector' creada amb columna de tipus VECTOR(384).")

# 3. Carregar les frases del fitxer JSON
with open("chunk_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

print(f"{len(sentences)} frases carregades del fitxer JSON.")

# 4. Inserir cada frase (només text) mesurant el temps individual
insert_times = []

for s in sentences:
    t0 = time.perf_counter()
    cur.execute("INSERT INTO sentences_pgvector (text) VALUES (%s)", (s,))
    conn.commit()
    t1 = time.perf_counter()
    insert_times.append(t1 - t0)

# 5. Estadístiques de temps per a G0
print("\n--- G0: Temps d'inserció de TEXT a Pgvector ---")
print(f"Min:     {min(insert_times):.6f} s")
print(f"Max:     {max(insert_times):.6f} s")
print(f"Mitjana: {statistics.mean(insert_times):.6f} s")
print(f"Stdev:   {statistics.stdev(insert_times):.6f} s")

cur.close()
conn.close()
