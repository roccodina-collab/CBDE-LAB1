import psycopg2
import json
import time
import statistics

# --- Configuració de connexió ---
DB_CONFIG = {
    "dbname": "vectorlab1",
    "user": "postgres",
    "password": "810808", 
    "host": "localhost",
    "port": "5432"
}

# --- Connectar-se a PostgreSQL ---
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# --- Crear la taula ---
cur.execute("""
    DROP TABLE IF EXISTS sentences;
    CREATE TABLE sentences (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding FLOAT4[]
    );
""")
conn.commit()
print("Taula 'sentences' creada.")

# --- Carregar les frases del fitxer JSON ---
with open("chunk_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)
print(f"{len(sentences)} frases carregades del fitxer.")

# --- Inserir cada frase, mesurant el temps individual ---
insert_times = []

for s in sentences:
    t0 = time.perf_counter()
    cur.execute("INSERT INTO sentences (text) VALUES (%s)", (s,))
    conn.commit()
    t1 = time.perf_counter()
    insert_times.append(t1 - t0)

# --- Estadístiques de temps ---
print("\n--- Temps d'inserció de TEXT ---")
print(f"Min:    {min(insert_times):.6f} s")
print(f"Max:    {max(insert_times):.6f} s")
print(f"Mitjana:{statistics.mean(insert_times):.6f} s")
print(f"Stdev:  {statistics.stdev(insert_times):.6f} s")

cur.close()
conn.close()
print("\nFet. Connexió tancada.")
