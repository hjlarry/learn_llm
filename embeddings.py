import pandas
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_embedding(text: str, model="text-embedding-ada-002", **kwargs):
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


# 从数据集里读取数据, 把combined这列通过大模型得到vector embeddings 追加到每一列
# 模型一般可选择text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
df = pandas.read_csv("data/fine_food_reviews_1k.csv", index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

df["embedding"] = df.combined.apply(lambda x: get_embedding(x))
df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")


# 之后可以再去读取这个文件,便于后续使用
# df = pandas.read_csv('data/fine_food_reviews_with_embeddings_1k.csv')
# df['embedding'] = df.embedding.apply(eval).apply(np.array)
