import re

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI()


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.invoke(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)


# count_tokens(llm_math, "What is 13 raised to the .3432 power?")
# print(llm_math.prompt.template)

def transform_func(inputs: dict) -> dict:
    text = inputs["text"]

    # replace multiple new lines and multiple spaces with a single one
    text = re.sub(r'(\r\n|\r|\n){2,}', r'\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return {"output_text": text}


clean_extra_spaces_chain = TransformChain(input_variables=["text"], output_variables=["output_text"],
                                          transform=transform_func)
# res = clean_extra_spaces_chain.invoke(
#     {"text": 'A random text  with   some irregular spacing.\n\n\n     Another one   here as well.'})
# print(res)

template = """Paraphrase this text:

{output_text}

In the style of a {style}.

Paraphrase: """
prompt = PromptTemplate(input_variables=["style", "output_text"], template=template)
style_paraphrase_chain = LLMChain(llm=llm, prompt=prompt, output_key='final_output')
sequential_chain = SequentialChain(chains=[clean_extra_spaces_chain, style_paraphrase_chain],
                                   input_variables=['text', 'style'], output_variables=['final_output'])
input_text = """
Chains allow us to combine multiple 


components together to create a single, coherent application. 

For example, we can create a chain that takes user input,       format it with a PromptTemplate, 

and then passes the formatted response to an LLM. We can build more complex chains by combining     multiple chains together, or by 


combining chains with other components.
"""
res = count_tokens(sequential_chain, {'text': input_text, 'style': 'a 90s rapper'})
print(res)