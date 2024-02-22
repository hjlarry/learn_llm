import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "seattle" in location.lower():
        return json.dumps({"location": "Seattle", "temperature": "10", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


# 1. 发送消息和可以选择的方法给model
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [
    {"role": "user", "content": "What's the weather like today in Seattle?"}
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    tools=tools
)
response_msg = response.choices[0].message
tool_calls = response_msg.tool_calls

# 2. 判断模型回复是否想用一个方法
if tool_calls:
    available_functions = {
        "get_current_weather": get_current_weather
    }
    messages.append(response_msg)
    # 3. 调用每个可以调用的方法
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions.get(function_name)
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(location=function_args.get("location"), unit=function_args.get("unit"))
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response
        })

    # 4. 把调用的方法及方法的返回 一起返回给模型
    second_response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
    )
    print(second_response.choices[0].message.content)
