from typing import Tuple

from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import ValidationError

from prompt_builder import prompt_builder, final_prompt
from action import Action
from utils import pprint, THOUGHT_COLOR, ROUND_COLOR, OBSERVATION_COLOR


class Agent:
    def __init__(self, llm, tools, work_dir="./data", max_thought_steps=10, memery_retriever=None, ):
        self.llm = llm
        self.tools = tools
        self.work_dir = work_dir
        self.max_thought_steps = max_thought_steps
        self.memery_retriever = memery_retriever

        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)

    @staticmethod
    def _format_long_term_memory(task_description: str, memory: BaseChatMemory) -> str:
        return memory.load_memory_variables(
            {"prompt": task_description}
        )["history"]

    @staticmethod
    def _format_short_term_memory(memory: BaseChatMemory) -> str:
        messages = memory.chat_memory.messages
        string_messages = [messages[i].content for i in range(1, len(messages))]
        return "\n".join(string_messages)

    def _step(self,
              reason_chain,
              task_description,
              short_term_memory,
              long_term_memory,
              verbose=False
              ) -> Tuple[Action, str]:
        response = ""
        short_memory = self._format_short_term_memory(short_term_memory)
        long_memory = self._format_long_term_memory(task_description, long_term_memory) if long_term_memory else ""
        memory = {"short_term_memory": short_memory, "long_term_memory": long_memory}
        for s in reason_chain.stream(memory):
            if verbose:
                pprint(s, THOUGHT_COLOR, end="")
            response += s

        action = self.robust_parser.parse(response)
        return action, response

    def _final_step(self, short_term_memory, task_description):
        """生成最终输出"""
        f_prompt = final_prompt.partial(task_description=task_description, short_term_memory=short_term_memory)
        chain = f_prompt | self.llm | StrOutputParser()
        return chain.invoke({})

    def _exec_action(self, action):
        tool = next((tool for tool in self.tools if tool.name == action.name), None)
        if tool is None:
            return f"Error: 找不到工具或指令 '{action.name}'. 请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
        try:
            observation = tool.run(action.args)
        except ValidationError as e:
            # 工具的入参异常
            observation = (f"Validation Error in args: {str(e)}, args: {action.args}")
        except Exception as e:
            # 工具执行异常
            observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    def run(self, task_description, verbose=False):
        prompt_template = prompt_builder(self.tools, self.output_parser).partial(work_dir=self.work_dir,
                                                                                 task_description=task_description)
        chain = prompt_template | self.llm | StrOutputParser()
        short_term_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=4000,
        )

        short_term_memory.save_context(
            {"input": "\n初始化"},
            {"output": "\n开始"}
        )

        # 如果有长时记忆，加载长时记忆
        if self.memery_retriever is not None:
            long_term_memory = VectorStoreRetrieverMemory(
                retriever=self.memery_retriever,
            )
        else:
            long_term_memory = None

        reply = ""
        thought_steps_count = 0
        while thought_steps_count < self.max_thought_steps:
            if verbose:
                pprint(f">>>>Round: {thought_steps_count}<<<<", ROUND_COLOR)

            action, response = self._step(chain, task_description, short_term_memory, long_term_memory, verbose)
            if action.name == "FINISH":
                if verbose:
                    pprint(f"\n----\nFINISH", OBSERVATION_COLOR)

                reply = self._final_step(short_term_memory, task_description)
                break

            observation = self._exec_action(action)
            if verbose:
                pprint(f"\n----\n结果:\n{observation}", OBSERVATION_COLOR)
            short_term_memory.save_context(
                {"input": response},
                {"output": "返回结果:\n" + observation}
            )

            thought_steps_count += 1

        if not reply:
            reply = "抱歉，我没能完成您的任务。"

        if long_term_memory is not None:
            long_term_memory.save_context(
                {"input": task_description},
                {"output": reply}
            )

        return reply
