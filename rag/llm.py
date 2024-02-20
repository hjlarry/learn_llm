from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = OpenAI()


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def get_embeddings(texts, model="text-embedding-ada-002"):
    """封装 OpenAI 的 Embedding 模型接口"""
    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]
