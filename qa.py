import ast

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy import spatial
import tiktoken

from data.wiki_article_part import wikipedia_article_on_curling

load_dotenv()
client = OpenAI()

# GPT学习知识有两种途径, 一种是通过fine-tune, 另一种就是把知识插入到输入信息中去
# fine-tune更适合去教AI一些特别的任务, 是一种长期记忆, 好像是考试要来了,模型临时学习了一个礼拜,可能会丢掉或记错一些细节
# 输入信息更适合去学习知识, 是一种短期记忆, 像是开卷考试
# 案例一、对比是否输入额外知识GPT回答的区别
"""
query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'
response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
        {'role': 'user', 'content': query},
    ],
    model="gpt-3.5-turbo",
    temperature=0,
)
# 此时gpt不知道2022年发生的事, 会回答As an AI language model, I don't have real-time data...
print(response.choices[0].message.content)
"""

query = f"""Use the below article on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found, write "I don't know."

Article:
\"\"\"
{wikipedia_article_on_curling}
\"\"\"

Question: Which athletes won the gold medal in curling at the 2022 Winter Olympics?"""
"""
response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
        {'role': 'user', 'content': query},
    ],
    model="gpt-3.5-turbo",
    temperature=0,
)
# 此时具备了外部知识, 它能回答正确 In the men's curling event, the gold medal was won by Sweden...
print(response.choices[0].message.content)
"""


# 案例二、通过相似度高的外部知识和问题一起传给GPT

# 200MB太大, 存本地了
# embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
embeddings_path = "data/winter_olympics_2022.csv"
df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)
GPT_MODEL = "gpt-3.5-turbo"


# 把要搜索的字符串和datafile传入, 得到最相关的top_n个字符串
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query,
    )
    # 得到query字符串的向量embedding
    query_embedding = query_embedding_response.data[0].embedding
    # 计算每一行的向量余弦相似度
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    # 按相似度排序
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# 得到外部文件中和查询字符串余弦相似度相近的文字后, 结合token数量限制, 加上prompt, 组装成一个发给GPT的消息
def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
        query: str,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    messages = [
        {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


# 用GPT4的答案更准确
msg1 = ask('Which athletes won the gold medal in curling at the 2022 Winter Olympics?')
print(msg1)
print("=====================================")
msg2 = ask('Did Jamaica or Cuba have more athletes at the 2022 Winter Olympics?')
print(msg2)
print("=====================================")
