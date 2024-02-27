from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv())

chat_client = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, model_kwargs={"seed": 42})
# chat_client = ChatOpenAI(model="gpt-4", temperature=0, model_kwargs={"seed": 42})
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
