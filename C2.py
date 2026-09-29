import time
import statistics
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Connectar a Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Carregar el model per codificar les 10 frases de consulta
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Les 10 frases de consulta
query_ids = ["55", "72", "33", "598", "711", "678", "979", "110", "222", "396"]

# 3. Recuperar la col·lecció original
original_collection = chroma_client.get_collection(
    name="book_corpus",
    embedding_function=None
)

# Obtenir tots els documents i embeddings
data = original_collection.get(
    include=["documents", "embeddings"]
)

ids = data["ids"]
documents = data["documents"]
embeddings = data["embeddings"]

# 4. Crear col·lecció per Cosine
try:
    chroma_client.delete_collection(name="book_corpus_cosine")
except Exception:
    pass

collection_cosine = chroma_client.create_collection(
    name="book_corpus_cosine",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"}
)

# 5. Crear col·lecció per L2
try:
    chroma_client.delete_collection(name="book_corpus_l2")
except Exception:
    pass

collection_l2 = chroma_client.create_collection(
    name="book_corpus_l2",
    embedding_function=None,
    metadata={"hnsw:space": "l2"}
)

# 6. Copiar els mateixos embeddings a les dues col·leccions
print("Copiant embeddings a les dues col·leccions...")

batch_size = 500

for i in range(0, len(ids), batch_size):
    collection_cosine.add(
        ids=ids[i:i + batch_size],
        documents=documents[i:i + batch_size],
        embeddings=embeddings[i:i + batch_size]
    )

    collection_l2.add(
        ids=ids[i:i + batch_size],
        documents=documents[i:i + batch_size],
        embeddings=embeddings[i:i + batch_size]
    )

print("Embeddings copiats correctament.")

# 7. Llistes per guardar els temps
query_times_cosine = []
query_times_l2 = []

print("\n=== INICIANT CONSULTES C2 (Cosine vs L2) ===")

# 8. Fer les mateixes consultes amb les dues mètriques
for qid in query_ids:

    # Obtenir el text de la consulta
    doc_data = original_collection.get(
        ids=[qid],
        include=["documents"]
    )

    if not doc_data["documents"]:
        continue

    query_text = doc_data["documents"][0]

    # Generar embedding de la consulta
    query_vector = model.encode(query_text).tolist()

    # -------------------------------------------------
    # COSINE
    # -------------------------------------------------

    t0 = time.perf_counter()

    results_cosine = collection_cosine.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    t1 = time.perf_counter()
    query_times_cosine.append(t1 - t0)

    res_ids = results_cosine["ids"][0][1:3]
    res_docs = results_cosine["documents"][0][1:3]
    res_dists = results_cosine["distances"][0][1:3]

    print(f"\n--- COSINE | ID={qid}: {query_text} ---")

    for r_id, r_doc, r_dist in zip(
        res_ids,
        res_docs,
        res_dists
    ):
        print(
            f"  Top Neighbor -> id={r_id} | "
            f"dist={r_dist:.4f} | {r_doc}"
        )

    # -------------------------------------------------
    # L2
    # -------------------------------------------------

    t0 = time.perf_counter()

    results_l2 = collection_l2.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    t1 = time.perf_counter()
    query_times_l2.append(t1 - t0)

    res_ids = results_l2["ids"][0][1:3]
    res_docs = results_l2["documents"][0][1:3]
    res_dists = results_l2["distances"][0][1:3]

    print(f"\n--- L2 | ID={qid}: {query_text} ---")

    for r_id, r_doc, r_dist in zip(
        res_ids,
        res_docs,
        res_dists
    ):
        print(
            f"  Top Neighbor -> id={r_id} | "
            f"dist={r_dist:.4f} | {r_doc}"
        )


# 9. Estadístiques Cosine
print("\n--- Temps de consulta COSINE ---")
print(f"Min:     {min(query_times_cosine):.6f} s")
print(f"Max:     {max(query_times_cosine):.6f} s")
print(f"Mitjana: {statistics.mean(query_times_cosine):.6f} s")
print(f"Stdev:   {statistics.stdev(query_times_cosine):.6f} s")

# 10. Estadístiques L2
print("\n--- Temps de consulta L2 ---")
print(f"Min:     {min(query_times_l2):.6f} s")
print(f"Max:     {max(query_times_l2):.6f} s")
print(f"Mitjana: {statistics.mean(query_times_l2):.6f} s")
print(f"Stdev:   {statistics.stdev(query_times_l2):.6f} s")
