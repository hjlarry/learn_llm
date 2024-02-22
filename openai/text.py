from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()
client = OpenAI()

# 1.发送聊天消息
"""
response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)
"""

# 2.发送聊天消息, 要求json回复
"""
# 只有gpt-3.5-turbo-1106或gpt4支持
# 需要发出去的第一句话要求系统使用JSON回复
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
print(response.choices[0].message.content)
"""


# 3.算token
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not presently implemented for model {model}.")


messages = [
    {"role": "system",
     "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
    {"role": "system", "name": "example_user", "content": "New synergies will help drive top-line growth."},
    {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
    {"role": "system", "name": "example_user",
     "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
    {"role": "system", "name": "example_assistant",
     "content": "Let's talk later when we're less busy about how to do better."},
    {"role": "user",
     "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
]

compute_token = num_tokens_from_messages(messages)
response = client.chat.completions.create(
  model="gpt-3.5-turbo-0613",
  messages=messages,
  temperature=0,
)
# 计算126, 实际129
print(f"Computed token is: {compute_token}, actural usage token is {response.usage.prompt_tokens}")
