import re

from nltk.tokenize import sent_tokenize
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


def extract_text_from_pdf(filename, min_line_length=10):
    """从 PDF文件中提取文字, 以 min_line_length为一段"""
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本, pdf中其他元素舍弃掉
    for page_layout in extract_pages(filename):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


def sent_tokenize_zh(input_string):
    """对于中文按标点断句"""
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    return [sentence for sentence in sentences if sentence.strip()]


def split_text(paragraphs, chunk_size=300, overlap_size=100):
    """按指定 chunk_size 和 overlap_size 交叠割文本"""
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev = i - 1
        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap + chunk
        next = i + 1
        # 向后计算当前chunk
        while next < len(sentences) and len(sentences[next]) + len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    return chunks


prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
__INFO__

用户问：
__QUERY__

请用中文回答用户问题。
"""


def build_prompt(prompt_template, **kwargs):
    """将 Prompt 模板赋值"""
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt


if __name__ == '__main__':
    paragraphs = extract_text_from_pdf("llama2.pdf")
    for para in paragraphs[:3]:
        print(para + "\n")
