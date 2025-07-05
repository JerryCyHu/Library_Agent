# convert_and_embed_booksummaries.py
import json, pathlib, itertools
from tqdm import tqdm
import chromadb

# --- embedding model ---------------------------------------------------------
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base",
    trust_remote_code=True,
    device="cuda"
).half() 


chroma_client = chromadb.PersistentClient("./books_chroma")
collection = chroma_client.get_or_create_collection("booksums_gte")


data_path = pathlib.Path("booksummaries.json")
books = json.loads(data_path.read_text(encoding="utf-8"))


batched_texts, batched_metas, batched_ids = [], [], []
batch_size = 512

def flush_batch():
    if not batched_texts:
        return
    embs = model.encode(batched_texts, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True).tolist()
    collection.add(
        embeddings = embs,
        documents  = batched_texts,
        metadatas  = batched_metas,
        ids        = batched_ids,
    )
    batched_texts.clear(); batched_metas.clear(); batched_ids.clear()

for idx, book in enumerate(tqdm(books, desc="Embedding")):
    book_id = str(idx + 1)  # start from 1
    
    meta_line = f'{book["bookname"]} by {book["author"]}. Genres: {", ".join(book["genre"])}'
    batched_texts.append(meta_line)
    batched_metas.append({"summary": book["summary"]})
    batched_ids.append(f"{book_id}.1")

    
    batched_texts.append(f'{book["bookname"]} summary: {book["summary"]}')
    batched_metas.append({
        "bookname": book["bookname"],
        "author"  : book["author"],
        "genre"   : ", ".join(book["genre"]),
    })
    batched_ids.append(f"{book_id}.2")


    if len(batched_texts) >= batch_size:
        flush_batch()

flush_batch()
print(f"Done! {collection.count()} vectors now in collection '{collection.name}'.")
