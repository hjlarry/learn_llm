from operator import itemgetter

from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
model = ChatOpenAI()

# 示例一
"""
prompt1 = ChatPromptTemplate.from_template(
    "生成一个属性为{attribute}的颜色. 只返回颜色的名称,别的不要返回:"
)
prompt2 = ChatPromptTemplate.from_template(
    "什么水果的颜色是{color}的. 只返回水果的名称,别的不要返回:"
)
prompt3 = ChatPromptTemplate.from_template(
    "哪个国家的国旗是{color}的. 只返回国家的名称,别的不要返回:"
)
prompt4 = ChatPromptTemplate.from_template(
    "{fruit}和{country}的国旗是什么颜色?"
)

model_parser = model | StrOutputParser()
color_generator = (
        {"attribute": RunnablePassthrough()} | prompt1 | {"color": model_parser}
)
color_to_fruit = prompt2 | model_parser
color_to_country = prompt3 | model_parser
question_generator = (
        color_generator | {"fruit": color_to_fruit, "country": color_to_country} | prompt4
)

prompt = question_generator.invoke("温暖")
res = model.invoke(prompt)
# 返回content='橙子和印度国旗的颜色都是橙色。'
print(res)
"""


# 示例二
"""
     Input
      / \
     /   \
 Branch1 Branch2
     \   /
      \ /
      Combine
"""
planner = (
    ChatPromptTemplate.from_template("生成一个关于{input}的辩论")
    | ChatOpenAI()
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

arguments_for = (
    ChatPromptTemplate.from_template(
        "列出{base_response}的优点"
    )
    | ChatOpenAI()
    | StrOutputParser()
)
arguments_against = (
    ChatPromptTemplate.from_template(
        "列出{base_response}的缺点"
    )
    | ChatOpenAI()
    | StrOutputParser()
)

final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "优点:\n{results_1}\n\n缺点:\n{results_2}"),
            ("system", "生成一个最终的结论"),
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)

chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)

res = chain.invoke({"input": "敏捷开发"})
# 最终的结论是，敏捷开发是一种有效的软件开发方法，但需要认真权衡其优点和缺点，并采取相应的措施来解决潜在的问题，以确保能够最大程度地发挥其效益。团队需要具备良好的沟通和协作能力，同时也需要有足够的技术能力和经验来保证项目的质量和成功。
print(res)