from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv())

llm = OpenAI(temperature=0, model_kwargs={"seed": 42})
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def load_docs(filename: str):
    ext = filename.split(".")[-1]
    if ext == "pdf":
        file_loader = PyPDFLoader(filename)
    elif ext == "docx" or ext == "doc":
        file_loader = UnstructuredWordDocumentLoader(filename)
    else:
        raise NotImplementedError(f"File extension {ext} not supported.")

    pages = file_loader.load_and_split()
    return pages


def ask_document(filename: str, query: str) -> str:
    """根据一个文档的内容，回答一个问题"""
    raw_docs = load_docs(filename)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    documents = text_splitter.split_documents(raw_docs)
    if documents is None or len(documents) == 0:
        return "无法读取文档内容"
    db = Chroma.from_documents(documents, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    response = qa_chain.invoke(query + "(请用中文回答)")
    return response


if __name__ == "__main__":
    filename = "../data/供应商资格要求.pdf"
    query = "销售额达标的标准是多少？"
    response = ask_document(filename, query)
    print(response)
