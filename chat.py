import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT")
)

deployment_name = os.getenv("DEPLOYMENT_NAME")


prompt = """
你是一个Python领域的专家,给我推荐一个初学者的课程,使用如下的格式:

格式:
- 概念:
- 简介:
- 示例代码
"""
messages = [{"role": "user", "content": prompt}]
completion = client.chat.completions.create(model=deployment_name, messages=messages)
old_prompt_result = completion.choices[0].message.content
print(old_prompt_result)
prompt = input("现在你有什么要说的?")

new_prompt = f"{old_prompt_result} {prompt}"
messages = [{"role": "user", "content": new_prompt}]
completion = client.chat.completions.create(model=deployment_name, messages=messages, max_tokens=1200)

# print response
print("最终结果:")
print(completion.choices[0].message.content)