
import re, json, signal, chromadb, torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextStreamer
)


encoder = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base",
    trust_remote_code=True, device="cuda"
).half()


LM_NAME = "OuteAI/Lite-Oute-2-Mamba2Attn-250M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(LM_NAME)
llm = AutoModelForCausalLM.from_pretrained(
    LM_NAME, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True
).eval()


client = chromadb.PersistentClient("./books_chroma")
collection = client.get_collection("booksums_gte")



def build_prompt(user_query, books):
    """拼成系统 + 用户模板"""
    cat_blocks = "\n\n".join(
        f"<BOOK>\n{b}\n</BOOK>"
        for b in books
    )
    sys_block = (
        "You are a literary recommendation assistant.\n\n"
        "Given the catalogue below and a user request, select the most relevant books and explain why each one fits.\n\n"
        "Your Guidelines\n"
        "1. Use **only** titles that appear in the catalogue—do not invent books or facts.\n"
        "2. Recommend **1 – 2** books, ordered by relevance.\n"
        "3. For every title, include:\n"
        "   • The exact book name\n"
        "   • A concise justification grounded in the catalogue’s author, genre, and summary information.\n"
        "4. Write your answer as a numbered Markdown list.\n\n"
        "## Catalogue entry schema\n"
        "<BOOK>\nBOOKNAME by AUTHOR. Genres: GENRE. SUMMARY.\n</BOOK>\n\n"
    )

    user_block = (
        "### USER QUERY\n"
        f"{user_query}"
        "### END QUERY\n"
        "### BOOK CATALOGUE\n"
        f"{cat_blocks}\n"
        "### END CATALOGUE"
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sys_block},
            {"role": "user", "content": user_block},
        ],
        add_generation_prompt=True,
        tokenize=False
    )

def query_books(user_query, top_k=3):
    """向量检索 + 去重 + prompt 构造"""
    q_emb = encoder.encode(
        user_query, normalize_embeddings=True, device="cuda"
    ).tolist()
    res = collection.query(
        query_embeddings=[q_emb], n_results=top_k,
        include=["documents", "metadatas"]
    )
    # zip back
    docs_map = {}
    for i in range(len(res['ids'][0])):
        id = res['ids'][0][i]
        idx = id.split('.', 1)[1]
        id = id.split('.', 1)[0]
        if id in docs_map: continue
        if idx == '1':
            book_data = f"{res['documents'][0][i]}. summary: {res['metadatas'][0][i]['summary']}."
            docs_map[id] = book_data
        else:
            book_data = f'{res["metadatas"][0][i]["bookname"]} by {res["metadatas"][0][i]["author"]}. Genres: {res["metadatas"][0][i]["genre"]}. {res["documents"][0][i]}.'
            docs_map[id] = book_data

    merged = list(docs_map.values())
    return merged

def generate_answer(prompt, max_new=512):
    ids = tokenizer(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(
        **ids, max_new_tokens=max_new,
        repetition_penalty=1.12,
        do_sample=False
    )
    return tokenizer.decode(out[0, ids["input_ids"].shape[-1]:], skip_special_tokens=True)


def main():
    print("Bibliotheca ready. Ask me anything!\n")
    try:
        while True:
            q = input("\nYou: ").strip()
            if not q:
                continue
            books = query_books(q, top_k=2)
            prompt = build_prompt(q, books)
            answer = generate_answer(prompt)
            print(answer)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")

if __name__ == "__main__":
    main()
