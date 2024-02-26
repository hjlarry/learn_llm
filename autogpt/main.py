from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from llm import embeddings, chat_client
from tools import document_qa_tool, document_generation_tool, email_tool, excel_inspection_tool, excel_analyze_tool, \
    directory_inspection_tool, finish_placeholder
from agent import Agent


def launch_agent(agent):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"

    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, verbose=True)
        print(f"{ai_icon}：{reply}\n")


def main():
    # 存储长时记忆的向量数据库
    db = Chroma.from_documents([Document(page_content="")], embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})

    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        excel_analyze_tool
    ]

    agent = Agent(chat_client, tools, memery_retriever=retriever)
    launch_agent(agent)


if __name__ == "__main__":
    main()
