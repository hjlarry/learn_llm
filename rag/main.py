"""
先通过执行`python vector.py`把文档灌入向量数据库中
再执行`python main.py`做想要的查询
"""
from utils import build_prompt, prompt_template
from llm import get_completion
from vector import vector_db


class RAG:
    def __init__(self, db, n_results=2):
        self.vector_db = db
        self.n_results = n_results

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. 构建 Prompt
        prompt = build_prompt(
            prompt_template, info=search_results['documents'][0], query=user_query)

        # 3. 调用 LLM
        response = get_completion(prompt)
        return response


if __name__ == '__main__':
    rag = RAG(db=vector_db)
    res = rag.chat('llama 2 安全吗')
    print(res)

