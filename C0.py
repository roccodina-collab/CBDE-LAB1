import json
import time
import statistics
import chromadb

# 1. Inicializar cliente persistente de Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Eliminar la colección previa si existe
try:
    chroma_client.delete_collection(name="book_corpus")
except Exception:
    pass

# Crear la colección en Chroma
collection = chroma_client.create_collection(
    name="book_corpus",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"}
)

# 2. Cargar las frases del archivo JSON
with open("chunk_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

print(f"{len(sentences)} frases cargadas del archivo JSON.")

# 3. Insertar solo el texto plano midiendo tiempos individualmente
insert_times = []

# Para insertar solo texto sin que falle la validación de Chroma, le pasamos un vector de ceros ficticio
dummy_embedding = [0.0] * 384

print("Iniciando inserción de texto en Chroma...")

for idx, sentence in enumerate(sentences):
    t0 = time.perf_counter()
    
    # Insertamos el documento con un embedding neutro para medir SOLO el tiempo de almacenamiento de texto
    collection.add(
        documents=[sentence],
        embeddings=[dummy_embedding],
        ids=[str(idx)]
    )
    
    t1 = time.perf_counter()
    insert_times.append(t1 - t0)

    # Feedback en consola cada 2.000 frases para ver que avanza
    if (idx + 1) % 2000 == 0:
        print(f"  Progreso: {idx + 1} / {len(sentences)} frases procesadas...")

# 4. Estadísticas de tiempo para C0
print("\n--- C0: Temps d'inserció de TEXT a Chroma ---")
print(f"Min:     {min(insert_times):.6f} s")
print(f"Max:     {max(insert_times):.6f} s")
print(f"Mitjana: {statistics.mean(insert_times):.6f} s")
print(f"Stdev:   {statistics.stdev(insert_times):.6f} s")