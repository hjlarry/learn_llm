from langchain_core.tools import Tool
from langchain.output_parsers import PydanticOutputParser

from action import Action
from tools.excel import analyze_excel
from tools.doc import ask_document
from tools.write import write
from prompt_builder import prompt_builder, final_prompt, excel_prompt


def test_excel_analyze():
    res = analyze_excel(
        query="8月销售额",
        filename="data/2023年8月-9月销售记录.xlsx",
        verbose=True
    )
    print(res)


def test_ask_doc():
    filename = "data/供应商资格要求.pdf"
    query = "销售额达标的标准是多少？"
    response = ask_document(filename, query)
    print(response)


def test_write():
    response = write("写一封邮件给张三，内容是：你好，我是李四。")
    print(response)


def test_prompt_template():
    tools = [
        Tool(name="FINISH", func=lambda: None, description="任务完成")
    ]
    output_parser = PydanticOutputParser(pydantic_object=Action)
    prompt_template = prompt_builder(tools, output_parser)

    print(prompt_template.format(
        task_description="解决问题",
        work_dir=".",
        short_term_memory="",
        long_term_memory="",
    ))

    print("====")
    print(final_prompt)
    print("====")
    print(excel_prompt)


test_excel_analyze()
# test_prompt_template()
# test_ask_doc()
# test_write()
