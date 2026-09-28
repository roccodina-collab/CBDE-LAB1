import time
import statistics
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Conectar al almacenamiento persistente de Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="book_corpus",
    embedding_function=None
)

# 2. Obtener los documentos guardados previamente en C0
data = collection.get()
ids = data["ids"]
documents = data["documents"]

print(f"{len(documents)} registros leídos de Chroma.")

# 3. Cargar el modelo Transformer (una sola vez)
print("Cargando modelo all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Generar embeddings y actualizar la colección midiendo tiempos
update_times = []

print("Iniciando generación e inserción de embeddings reales...")

for idx, (doc_id, doc_text) in enumerate(zip(ids, documents)):
    t0 = time.perf_counter()
    
    # Generar el embedding de la frase
    embedding = model.encode(doc_text).tolist()
    
    # Actualizar el registro en Chroma con su vector correspondiente
    collection.update(
        ids=[doc_id],
        embeddings=[embedding]
    )
    
    t1 = time.perf_counter()
    update_times.append(t1 - t0)

    # Feedback visual cada 2.000 frases
    if (idx + 1) % 2000 == 0:
        print(f"  Progreso: {idx + 1} / {len(documents)} embeddings procesados...")

# 5. Estadísticas de tiempo para C1
print("\n--- C1: Temps de generació + inserció d'EMBEDDINGS a Chroma ---")
print(f"Min:     {min(update_times):.6f} s")
print(f"Max:     {max(update_times):.6f} s")
print(f"Mitjana: {statistics.mean(update_times):.6f} s")
print(f"Stdev:   {statistics.stdev(update_times):.6f} s")