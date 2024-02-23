from operator import itemgetter

from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase

load_dotenv()
model = ChatOpenAI()

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)
db = SQLDatabase.from_uri("sqlite:///./chinook.db")


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
)

# res = sql_response.invoke({"question": "How many employees are there?"})
# print(res) # SELECT COUNT(*) FROM employees


template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}
"""

prompt_response = ChatPromptTemplate.from_template(template)
full_chain = (
        RunnablePassthrough.assign(query=sql_response).assign(
            schema=get_schema,
            response=lambda x: db.run(x["query"]),
        )
        | prompt_response
        | model
)

res = full_chain.invoke({"question": "How many employees are there?"})
# content='There are 8 employees.'
print(res)
