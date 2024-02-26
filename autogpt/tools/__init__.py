from langchain.tools import StructuredTool
from .doc import ask_document
from .dir import list_files_in_directory
from .write import write
from .my_email import send_email
from .excel import get_first_n_rows, get_column_names, analyze_excel

document_qa_tool = StructuredTool.from_function(
    func=ask_document,
    name="AskDocument",
    description="根据一个Word或PDF文档的内容，回答一个问题。考虑上下文信息，确保问题对相关概念的定义表述完整。",
)

document_generation_tool = StructuredTool.from_function(
    func=write,
    name="GenerateDocument",
    description="根据需求描述生成一篇正式文档",
)

email_tool = StructuredTool.from_function(
    func=send_email,
    name="SendEmail",
    description="给指定的邮箱发送邮件。确保邮箱地址是xxx@xxx.xxx的格式。多个邮箱地址以';'分割。",
)

excel_inspection_tool = StructuredTool.from_function(
    func=get_first_n_rows,
    name="InspectExcel",
    description="探查表格文件的内容和结构，展示它的列名和前n行，n默认为3",
)

excel_analyze_tool = StructuredTool.from_function(
    func=analyze_excel,
    name="AnalyzeExcel",
    description="通过程序脚本分析一个结构化文件（例如excel文件）的内容。输人中必须包含文件的完整路径和具体分析方式和分析依据，阈值常量等。如果输入信息不完整，你可以拒绝回答",
)

directory_inspection_tool = StructuredTool.from_function(
    func=list_files_in_directory,
    name="ListDirectory",
    description="探查文件夹的内容和结构，展示它的文件名和文件夹名",
)

finish_placeholder = StructuredTool.from_function(
    func=lambda: None,
    name="FINISH",
    description="用于表示任务完成的占位符工具"
)
