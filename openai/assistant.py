import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-3.5-turbo"
)

print("Created Assistant:", assistant.model_dump_json())
print("===================================================")

# 每个thread代表一个对话
thread = client.beta.threads.create()
print("Created Thread:", thread.model_dump_json())
print("===================================================")

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)
print("Created Message:", message.model_dump_json())
print("===================================================")

# 创建的thread和assistant是无关的, 需要用run把它们连接起来
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
print("Created Run:", run.model_dump_json())
print("===================================================")


# run是异步的, 需要判断它的状态来确定有没有运行完
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


run = wait_on_run(run, thread)
print("Completed Run:", run.model_dump_json())
print("===================================================")


# 消息列表倒序排列
messages = client.beta.threads.messages.list(thread_id=thread.id)
print("All Messages:", messages.model_dump_json())

time.sleep(5)

# 给thread里再追加一条消息并等待其运行完
message = client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content="Could you explain this to me?"
)
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
wait_on_run(run, thread)

# 这次只显示追加的消息
messages = client.beta.threads.messages.list(
    thread_id=thread.id, order="asc", after=message.id
)
print("Append Messages:", messages.model_dump_json())