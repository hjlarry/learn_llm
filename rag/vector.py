import chromadb

from llm import get_embeddings
from utils import split_text, extract_text_from_pdf


class MyVectorDBConnector:
    def __init__(self, collection_name):
        chroma_client = chromadb.PersistentClient(path="my_vector_db")
        self.collection = chroma_client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents):
        """向 collection 中添加文档与向量"""
        self.collection.add(
            embeddings=get_embeddings(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        """检索向量数据库"""
        results = self.collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=top_n
        )
        return results


vector_db = MyVectorDBConnector("demo_text_split")

if __name__ == "__main__":
    paragraphs = extract_text_from_pdf("llama2.pdf")
    chunks = split_text(paragraphs, 300, 100)
    vector_db.add_documents(chunks)
