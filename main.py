from src.vectorstore import load_vectorstore
from src.rag_pipeline import get_retriever, rag_answer
from src.preprocessing_data import load_data , split_docs

def main():
    print("🚀 Starting RAG Test...\n")

    # 1️⃣ Load & Split documents
    print("📄 Loading documents...")
    doc = load_data()
    print("✅ Number of documents:", len(doc))

    chunks = split_docs(doc)
    print("✅ Number of chunks after splitting:", len(chunks))

    # 2️⃣ Load or Create Vectorstore
    print("\n🗄 Loading vectorstore...")
    vectorstore = load_vectorstore(chunks)

    # 3️⃣ Create Retriever
    retriever = get_retriever(vectorstore, k=5)

    # 4️⃣ Ask Question
    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = rag_answer(
            name="Asmaa",
            query=query,
            retriever=retriever
        )

        print("\n🧠 Final Answer:\n", answer)


if __name__ == "__main__":
    main()