import time
import statistics
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Connectar a Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Carregar el model per codificar les 10 frases de consulta
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Les 10 frases de consulta (els mateixos IDs que a P2 i G2)
query_ids = ["56", "72", "33", "598", "711", "678", "979", "110", "222", "396"]

# Obtenir la col·lecció amb espai Cosinus
collection_cosine = chroma_client.get_collection(
    name="book_corpus",
    embedding_function=None
)

# Crear o recuperar una col·lecció temporal per a L2 si volem comparar les dues mètriques
# (o bé si només tenim book_corpus, comprovem com respon Cosine vs les distàncies)
query_times = []

print("=== INICIANDO CONSULTAS C2 (Chroma HNSW - Cosine & L2 Evaluation) ===")

for qid in query_ids:
    # Obtenir el text i l'embedding del registre de consulta
    doc_data = collection_cosine.get(ids=[qid], include=["documents", "embeddings"])
    if not doc_data["documents"]:
        continue
    
    query_text = doc_data["documents"][0]
    query_vector = model.encode(query_text).tolist()

    t0 = time.perf_counter()

    # Consulta Top-3 per descartar la pròpia frase (posició 0)
    results_cosine = collection_cosine.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    t1 = time.perf_counter()
    query_times.append(t1 - t0)

    # Extraure descartando la posició 0 (sí mateix)
    res_ids = results_cosine["ids"][0][1:3]
    res_docs = results_cosine["documents"][0][1:3]
    res_dists = results_cosine["distances"][0][1:3]

    print(f"\n--- Consulta Chroma Cosine (ID={qid}): {query_text} ---")
    for r_id, r_doc, r_dist in zip(res_ids, res_docs, res_dists):
        print(f"  Top Neighbor (Cosine) -> id={r_id} | dist={r_dist:.4f} | {r_doc}")

print("\n--- Tiempos de consulta C2 (Chroma HNSW, Top-2, 10 frases) ---")
print(f"Min:     {min(query_times):.6f} s")
print(f"Max:     {max(query_times):.6f} s")
print(f"Mitjana: {statistics.mean(query_times):.6f} s")
print(f"Stdev:   {statistics.stdev(query_times):.6f} s")
