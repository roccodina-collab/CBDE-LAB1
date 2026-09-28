import time
import statistics
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Conectar a Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Cargar el modelo para codificar las 10 frases de consulta
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Las 10 frases de consulta (usando los mismos IDs que en P2)
query_ids = ["56", "72", "33", "598", "711", "678", "979", "110", "222", "396"]

collection_cosine = chroma_client.get_collection(
    name="book_corpus",
    embedding_function=None
)

query_times = []

print("=== INICIANDO CONSULTAS C2 (Chroma HNSW) ===")

for qid in query_ids:
    # Obtener el texto del ID de consulta
    doc_data = collection_cosine.get(ids=[qid])
    if not doc_data["documents"]:
        continue
    
    query_text = doc_data["documents"][0]
    query_vector = model.encode(query_text).tolist()

    t0 = time.perf_counter()

    # Pedimos 3 resultados para descartar la propia frase (que estará en la pos 0)
    results = collection_cosine.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    t1 = time.perf_counter()
    query_times.append(t1 - t0)

    # Extraer descartando el elemento 0 (sí mismo)
    res_ids = results["ids"][0][1:3]
    res_docs = results["documents"][0][1:3]
    res_dists = results["distances"][0][1:3]

    print(f"\n--- Consulta Chroma (ID={qid}): {query_text} ---")
    for r_id, r_doc, r_dist in zip(res_ids, res_docs, res_dists):
        print(f"  Top Neighbor -> id={r_id} | dist/score={r_dist:.4f} | {r_doc}")

print("\n--- Tiempos de consulta C2 (Chroma, Top-2, 10 frases) ---")
print(f"Min:     {min(query_times):.6f} s")
print(f"Max:     {max(query_times):.6f} s")
print(f"Mitjana: {statistics.mean(query_times):.6f} s")
print(f"Stdev:   {statistics.stdev(query_times):.6f} s")