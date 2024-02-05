from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

vectorstore = Chroma.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()
chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "language": itemgetter("language"),
        }
        | prompt
        | model
        | StrOutputParser()
)
res = chain.invoke({"question": "where did harrison work", "language": "chinese"})
print(res)
