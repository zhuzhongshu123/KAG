import re
from string import Template
from typing import List
import logging

from kag.interface.common.prompt import PromptABC

logger = logging.getLogger(__name__)

@PromptABC.register("llm_backup")
class RespGenerator(PromptABC):
    template_zh = """
# 背景知识
$dk

# 问题
$question
"""
    template_en = template_zh

    def __init__(self, language: str):
        super().__init__(language)

    @property
    def template_variables(self) -> List[str]:
        return ["dk", "question"]

    def parse_response(self, response: str, **kwargs):
        return response
