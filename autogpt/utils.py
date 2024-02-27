import sys
import re

from colorama import Fore, Style
from langchain_core.output_parsers import BaseOutputParser

THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
ROUND_COLOR = Fore.BLUE
CODE_COLOR = Fore.CYAN


def pprint(text, color=None, end="\n"):
    if color is not None:
        content = color + text + Style.RESET_ALL + end
    else:
        content = text + end
    sys.stdout.write(content)
    sys.stdout.flush()


class PythonCodeParser(BaseOutputParser):
    """从OpenAI返回的文本中提取Python代码。"""

    def parse(self, text: str) -> str:
        return re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)[0]
