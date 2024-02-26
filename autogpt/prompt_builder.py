from pathlib import Path
import json

from langchain.prompts import load_prompt
from langchain.tools.render import render_text_description


def json_with_ascii(text):
    # 传入一段文字，如果包含一行json，则使json中能显示中文而非\uXXXX转义序列
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return '\n'.join(lines)


def prompt_builder(tools, output_parser):
    # main.txt需要是gbk编码
    main_prompt_template = load_prompt("prompts/main.json")
    variables = main_prompt_template.input_variables
    partial_variables = {}

    for var in variables:
        file = Path(f"prompts/{var}.txt")
        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                partial_variables[var] = f.read()

    if tools is not None:
        partial_variables["tools"] = render_text_description(tools)

    partial_variables["format_instructions"] = json_with_ascii(output_parser.get_format_instructions())

    prompt_template = main_prompt_template.partial(**partial_variables)
    return prompt_template


# final_step.txt需要是gbk编码
final_prompt = load_prompt("prompts/final_step.json")
excel_prompt = load_prompt("prompts/excel_analyser.json")
