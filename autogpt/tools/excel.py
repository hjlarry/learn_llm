import warnings

import pandas as pd
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

from utils import PythonCodeParser, pprint, CODE_COLOR
from prompt_builder import excel_prompt

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

def get_sheet_names(filename: str) -> str:
    """获取 Excel 文件的工作表名称"""
    excel_file = pd.ExcelFile(filename)
    sheet_names = excel_file.sheet_names
    return f"这是 '{filename}' 文件的工作表名称：\n\n{sheet_names}"


def get_column_names(filename: str) -> str:
    """获取 Excel 文件的列名"""
    df = pd.read_excel(filename, sheet_name=0)
    column_names = '\n'.join(df.columns.to_list())
    result = f"这是 '{filename}' 文件第一个工作表的列名：\n\n{column_names}"
    return result


def get_first_n_rows(filename: str, n: int = 3) -> str:
    """获取 Excel 文件的前 n 行"""
    result = get_sheet_names(filename) + "\n\n"
    result += get_column_names(filename) + "\n\n"

    df = pd.read_excel(filename, sheet_name=0)  # sheet_name=0 表示第一个工作表
    n_lines = '\n'.join(df.head(n).to_string(index=False, header=True).split('\n'))

    result += f"这是 '{filename}' 文件第一个工作表的前{n}行样例：\n\n{n_lines}"
    return result


def analyze_excel(query: str, filename: str, verbose=False) -> str:
    inspections = get_first_n_rows(filename, 3)
    llm = ChatOpenAI(temperature=0, model_kwargs={"seed": 42})
    chain = excel_prompt | llm | PythonCodeParser()

    if verbose:
        pprint("\n#!/usr/bin/env python", CODE_COLOR, end="\n")

    code = ""
    for c in chain.stream({
        "query": query,
        "filename": filename,
        "inspections": inspections
    }):
        if verbose:
            pprint(c, CODE_COLOR, end="")
        code += c

    if not code:
        return "没有找到可执行的Python代码"

    res = PythonREPL().run(code)
    return res
