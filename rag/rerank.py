from sentence_transformers import CrossEncoder

from vector import vector_db


user_query = "how safe is llama 2"
search_results = vector_db.search(user_query, 5)

# 直接从向量数据库搜索的结果并没有使相关性高的结果排在前面
for doc in search_results['documents'][0]:
    print(doc + "\n")

print("===========================")
# 通过排序模型可以对结果重排序得到更准确的结果
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
scores = model.predict([[user_query, doc]
                        for doc in search_results['documents'][0]])
# 按得分排序
sorted_list = sorted(
    zip(scores, search_results['documents'][0]), key=lambda x: x[0], reverse=True)
for score, doc in sorted_list:
    print(f"{score}\t{doc}\n")
