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

# 1.发送基本文字
# prompt = "Complete the following: Once upon a time there was a"
# response = client.completions.create(model=deployment_name, prompt=prompt)
# print(response.choices[0].text)

# 2.发送聊天消息
# response = client.chat.completions.create(model=deployment_name, messages=[{"role": "user", "content": "Hello world"}])
# print(response.choices[0].message.content)

prompt = "Complete the following: Once upon a time there was a"
messages = [{"role": "user", "content": prompt}]

response = client.chat.completions.create(model=deployment_name, messages=messages)
print(response.choices[0].message.content)
