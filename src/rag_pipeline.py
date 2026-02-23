import os
from langchain_core.prompts import PromptTemplate
from .model_loader import load_llm


# -----------------------------
# 3️⃣ Create Retriever
# -----------------------------
def get_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

# -----------------------------
# 4️⃣ Prompt Template
# -----------------------------
PROMPT = PromptTemplate.from_template("""
You are an AI assistant answering questions about {name}.
Paraphrase naturally instead of copying sentences directly.
If the answer is not in the context, say:
'I do not have information about that.'

Context:
{context}

Question:
{question}

Answer:
""")


# -----------------------------
# 5️⃣ RAG Answer Function
# -----------------------------
def rag_answer(name, query, retriever):
    llm = load_llm()

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.format(
        name=name,
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)

    print("📌 Retrieved:", len(docs))
    print("🤖 Answer:", response)

    return response