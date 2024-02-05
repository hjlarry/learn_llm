from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain_community.chat_models import ChatAnthropic

# 需要ChatAnthropic的APIkey, 暂未运行成功
model = ChatAnthropic(model="claude-2")


@tool
def search(query: str) -> str:
    """Search things about current events."""
    return "32 degrees"


tool_list = [search]
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/xml-agent-convo")


# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: convert_intermediate_steps(
                x["intermediate_steps"]
            ),
        }
        | prompt.partial(tools=convert_tools(tool_list))
        | model.bind(stop=["</tool_input>", "</final_answer>"])
        | XMLAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
res = agent_executor.invoke({"input": "whats the weather in New york?"})
print(res)
