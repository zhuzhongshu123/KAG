import re
from string import Template
from typing import List
import logging

from kag.interface.common.prompt import PromptABC

logger = logging.getLogger(__name__)

@PromptABC.register("resp_think_generator")
class RethinkRespGenerator(PromptABC):
    template_zh = """
# task
给定的信息不能够回答问题，你的任务是基于背景知识和用户给定的问题，思考背后需要获取的信息，并基于已有信息经可能给出一些相关的回答
# 背景知识
$dk

# output format
纯文本，不要包含markdown格式。

# context
$memory

# question
$question

# your answer
"""
    template_en = """# task
The provided information is insufficient to answer the question. Your task is to, based on background knowledge and the user's given question, identify the information that needs to be obtained and potentially provide some related answers based on the existing information.
# Background Knowledge
$dk

# output format
Plain text, do not include markdown formatting.

# context
$memory

# question
$question

# your answer"""

    def __init__(self, language: str):
        super().__init__(language)

    @property
    def template_variables(self) -> List[str]:
        return ["memory", "question", "dk"]

    def parse_response(self, response: str, **kwargs):
        return response
