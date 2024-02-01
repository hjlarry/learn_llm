import pandas as pd
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_embedding(text: str, model="text-embedding-ada-002", **kwargs):
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 从数据集里读取数据, 把combined这列通过大模型得到vector embeddings 追加到每一列
# 模型一般可选择text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
"""
df = pd.read_csv("data/fine_food_reviews_1k.csv", index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

df["embedding"] = df.combined.apply(lambda x: get_embedding(x))
df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")
"""

# 之后可以再去读取这个文件, 使用它进行语义搜索
df = pd.read_csv('data/fine_food_reviews_with_embeddings_1k.csv')
df['embedding'] = df.embedding.apply(eval).apply(np.array)


def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(product_description)
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


search_reviews(df, "delicious beans")
