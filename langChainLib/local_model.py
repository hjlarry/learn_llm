from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

CURRENT_MODEL = "llama2"
# CURRENT_MODEL = "gemma:2b"
llm = Ollama(model=CURRENT_MODEL)

# 一、只使用本地模型本身的能力回答这个问题

from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])
# chain = prompt | llm | output_parser
# res = chain.invoke({"input": "how the langsmith pricing?"})
# print(res, 1221321)




# 二、加入外部知识来回答这个问题
# 把这篇文章的内容转换为向量数据库
loader = WebBaseLoader("https://docs.smith.langchain.com/pricing", verify_ssl=False)
docs = loader.load()
embeddings = OllamaEmbeddings(model=CURRENT_MODEL)
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = Chroma.from_documents(documents, embeddings)

# 然后测试让模型基于一句话（代码中的page_content）回答问题
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")


document_chain = create_stuff_documents_chain(llm, prompt)
res = document_chain.invoke({
    "input": "how the langsmith pricing?",
    "context": [Document(page_content="priseDesigned for hobbyists who want to start their adventure soloEverything in Developer, plus team features, higher rate limits, and longer data retentionDesigned for teams with more security, deployment, and support needsFree for 1 user3,000 free traces per monthAdditional traces billed @ $0.005/trace$39/user8,000 free traces per monthAdditional traces billed @ $0.005/traceCustomKey features:1 Developer seatDebugging ")]
})
print(res, 1221321)

# 让模型基于之前的向量数据库回答问题
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "how the langsmith pricing?"})
print(response["answer"])


# 三、之前的都只能回答一个问题，现在做一个聊天机器人

# from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage
#
#
# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
# ])
# retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
#
#
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
# ])
# document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
#
# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# res = retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })
#
# print(res)