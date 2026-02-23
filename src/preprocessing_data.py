from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter , RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE , CHUNK_OVERLAP , file_path

def load_data():
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section")]
    )

    smart_chunks = []

    for doc in documents:
        chunks = markdown_splitter.split_text(doc.page_content)

        for chunk_doc in chunks:
            text = chunk_doc.page_content

            if len(text) > chunk_size:
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", ".", " "]
                )

                sub_chunks = sub_splitter.split_text(text)

                for sub in sub_chunks:
                    smart_chunks.append(
                        type(doc)(
                            page_content=sub,
                            metadata=chunk_doc.metadata
                        )
                    )
            else:
                smart_chunks.append(chunk_doc)

    return smart_chunks