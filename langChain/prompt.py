from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

load_dotenv()

# PromptTemplate / ChatPromptTemplate -> LLM / ChatModel -> OutputParser
prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model | StrOutputParser()
res = chain.invoke({"foo": "bears"})
print(res)

# A complicated example
functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]

chain = (
        prompt
        | model.bind(function_call={"name": "joke"}, functions=functions)
        | JsonOutputFunctionsParser()
)
res = chain.invoke({"foo": "bears"})
print(res)
