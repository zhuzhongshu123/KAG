import logging
import json
from typing import List


logger = logging.getLogger(__name__)

from kag.interface.common.prompt import PromptABC

@PromptABC.register("logic_form_plan_table")
class LogicFormPlanPrompt(PromptABC):
    template_zh = """
{
  "task": "拆解子问题",
  "instruction": [
    "找出解决问题的核心关键步骤，总结为子问题。",
    "参考函数能力，将子问题分配给合适的函数进行处理。",
    "参考failed_cases中失败的尝试，改变思路尝试其他拆解方式，生成新的子问题！"
  ],
  "pay_attention": [
    "你的数学计算能力能力很差，必须使用PythonCoder求解。",
    "拆解子问题要详略得当，每个子问题可独立求解；子问题必须是核心关键步骤，而不是极度详细的执行步骤。",
    "子问题描述信息要完整，不要遗漏任何关键字，语义要清晰，利于理解。",
    "你有领域知识domain_knowledge可供参考"
  ],
  "output_format": [
    "输出json格式，output给出子问题列表",
    "每个子问题包含sub_question和process_function"
  ],
  "domain_knowledge": "$dk",
  "functions": [
    {
      "functionName": "Retrieval",
      "description": "包含一个知识库，根据给出的检索条件(自然语言)，返回检索结果。",
      "pay_attention": [
        "Retrieval的问题必须是具体明确的；不合格的问题：查找财务报表。具体明确的问题：查找2024年全年的净利润值。"
      ],
      "knowledge_base_content": "$kg_content",
      "examples": [
        {
          "knowledge_base_content": "中芯国际2024第3季度财务报表",
          "input": "从资产负债信息中召回流动资产的所有子项",
          "output": "略"
        }
      ]
    },
    {
      "functionName": "PythonCoder",
      "description": "对给出的问题，编写python代码求解。",
      "pay_attention": "只使用python基础库",
      "examples": [
        {
          "input": "9.8和9.11哪个大？",
          "internal_processing_logic": "编写python代码```python\nanswer=max(9.8, 9.11)\nprint(answer)```, 调用执行器获得结果",
          "output": "9.8"
        },
        {
          "input": "今天星期几？",
          "internal_processing_logic": "```python\nimport datetime\n\n# 获取当前日期\ntoday = datetime.datetime.now()\n\n# 将日期格式化为星期几，%A会给出完整的星期名称\nday_of_week = today.strftime(\"%A\")\n\n# 打印结果\nprint(\"今天是:\", day_of_week)\n```",
          "output": "例子中无法给出答案，取决于具体的运行时间"
        }
      ]
    }
  ],
  "examples": [
    {
      "input": "如果游戏收入按照目前的速度增长，2020年的游戏收入是多少？",
      "output": [
        {
          "sub_question": "查找2018年和2019年游戏收入",
          "process_function": "Retrieval"
        },
        {
          "sub_question": "根据2018年和2019年游戏收入，计算2019年游戏收入增长率；再根据增长率，计算2020年游戏收入",
          "process_function": "PythonCoder"
        }
      ]
    },
    {
      "input": "找到两个数，他们的乘积为1038155，他们的和为2508",
      "output": [
        {
          "sub_question": "解方程组以找到满足条件的数：\n1. 两数乘积为 X * Y = 1038155\n2. 两数和为 X + Y = 2508\n使用数学方法或编程计算 X 和 Y 的具体值。",
          "process_function": "PythonCoder"
        }
      ]
    },
    {
      "input": "阿里巴巴财报中最新的资产负债信息中流动资产最高的子项是哪个？其占流动资产的比例是多少？",
      "output": [
        {
          "sub_question": "召回阿里巴巴最新的资产负债信息中流动资产总值",
          "process_function": "Retrieval"
        },
        {
          "sub_question": "查询阿里巴巴最新的资产负债信息中所有流动资产详情",
          "process_function": "Retrieval"
        },
        {
          "sub_question": "根据召回的阿里巴巴流动资产详情，计算最高的子项是哪个？并计算最高子项占总流动资产的比例是多少？",
          "process_function": "PythonCoder"
        }
      ]
    },
    {
      "input": "在A公司担任高管又同时担任风险委员会主席的人，是哪个国家的人？",
      "output": [
        {
          "sub_question": "查找A公司的高管名单",
          "process_function": "Retrieval"
        },
        {
          "sub_question": "查找A公司风险委员会主席的人有哪些",
          "process_function": "Retrieval"
        },
        {
          "sub_question": "以上子问题答案中共同人员有谁，是哪个国家的人",
          "process_function": "Retrieval"
        }
      ]
    }
  ],
  "input": "$input",
  "failed_cases": "$history"
}
"""
    template_en = """{
"task": "Decompose into Sub-Questions",
"instruction": [
"Identify the core key steps to solve the problem and summarize them as sub-questions.",
"Refer to function capabilities and assign sub-questions to appropriate functions for processing.",
"Refer to failed attempts in failed_cases, change the approach, try other decomposition methods, and generate new sub-questions!"
],
"pay_attention": [
"Your mathematical calculation abilities are poor; you must use PythonCoder to solve.",
"The decomposition of sub-questions must be appropriately detailed, and each sub-question can be solved independently; sub-questions must be core key steps, not extremely detailed execution steps.",
"Descriptions of sub-questions must be complete, without omitting any keywords, with clear semantics for ease of understanding."
],
"output_format": [
"Just Output in JSON format, without any other description after JSON Data, with 'output' providing the list of sub-questions.",
"Each sub-question includes 'sub_question' and 'process_function'.",
"The value of 'sub_question' is a string type. Note: Do not include line breaks to avoid JSON format issues.",
"The value of 'process_function' should be a 'functionName' from 'functions'."
],
"functions": [
{
  "functionName": "PythonCoder",
  "description": "Write Python code to solve the given problem.",
  "pay_attention": "Use only Python's standard libraries.",
  "examples": [
    {
      "input": "Which is larger, 9.8 or 9.11?",
      "internal_processing_logic": "Write Python code ```python\nanswer = max(9.8, 9.11)\nprint(answer)``` and call the executor to obtain the result.",
      "output": "9.8"
    },
    {
      "input": "What day of the week is it today?",
      "internal_processing_logic": "```python\nimport datetime\n\n# Get the current date\ntoday = datetime.datetime.now()\n\n# Format the date to get the day of the week, %A returns the full weekday name\nday_of_week = today.strftime(\"%A\")\n\n# Print the result\nprint(\"Today is:\", day_of_week)\n```",
      "output": "The example cannot provide an answer; it depends on the specific execution time."
    }
  ]
}
],
"examples": [
{
  "input": "Find two numbers whose product is 1,038,155 and whose sum is 2,508.",
  "output": [
    {
      "sub_question": "Solve the system of equations to find the numbers that satisfy the conditions: 1. The product of the two numbers is X * Y = 1,038,155. 2. The sum of the two numbers is X + Y = 2,508. Use mathematical methods or programming to calculate the specific values of X and Y.",
      "process_function": "PythonCoder"
    }
  ]
}
],
"input": "$input",
"failed_cases": "$history"
}
"""

    def __init__(self, language: str):
        super().__init__(language)

    @property
    def template_variables(self) -> List[str]:
        return ["input", "kg_content", "history", "dk"]

    def parse_response(self, response: str, **kwargs):
        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        return rsp
