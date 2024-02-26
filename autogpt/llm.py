from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv())

openai_client = OpenAI(temperature=0, model_kwargs={"seed": 42})
chat_client = ChatOpenAI(temperature=0, model_kwargs={"seed": 42})
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
