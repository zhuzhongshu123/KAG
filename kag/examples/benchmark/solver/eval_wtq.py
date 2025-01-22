# -*- coding: utf-8 -*-
import re
import json
import shlex
import subprocess
from pathlib import Path
from colorama import Fore

import numpy as np
from tabulate import tabulate
from kag.interface import LLMClient
from tenacity import retry, stop_after_attempt
from camel.interpreters import SubprocessInterpreter
from concurrent.futures import ThreadPoolExecutor, as_completed
from kag.common.checkpointer import CheckpointerManager


class MySubprocessInterpreter(SubprocessInterpreter):
    def run_file(
        self,
        file: Path,
        code_type: str,
    ) -> str:
        r"""Executes a code file in a subprocess and captures its output.

        Args:
            file (Path): The path object of the file to run.
            code_type (str): The type of code to execute (e.g., 'python',
                'bash').

        Returns:
            str: A string containing the captured stdout and stderr of the
                executed code.

        Raises:
            RuntimeError: If the provided file path does not point to a file.
            InterpreterError: If the code type provided is not supported.
        """
        if not file.is_file():
            raise RuntimeError(f"{file} is not a file.")
        code_type = self._check_code_type(code_type)
        cmd = shlex.split(
            self._CODE_EXECUTE_CMD_MAPPING[code_type].format(file_name=str(file))
        )
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
        if stderr:
            raise RuntimeError(f"\n{stderr}")
        exec_result = f"{stdout}"
        exec_result += f"(stderr: {stderr})" if stderr else ""
        return exec_result


class Solver:
    def __init__(self):
        self.prompt = """
I have a reading comprehension task in which each question is associated with a Wikipedia table. The table is given in the format of a list of dicts, where each dict represents a row in the table, and the dict's keys and values correspond to the cell titles and their respective values. Accurately answering these questions requires performing various calculations on the table data, such as table lookup, aggregation, comparison (maximum, minimum), arithmetic operations, joins, and unions. Please follow these steps to provide the answer:

1. Identify all rows and columns in the table that are helpful for answering the question.
2. Determine the series of calculation operations that need to be performed on the table to answer the question.
3. Execute the calculations identified in step two on the table data and provide the answer.

!!!IMPORTANT: You must conclude you response with "Answer: " to present a concise, definitive response, devoid of additional elaboration in step 3. Acceptable formats such as "Answer: 14 May", "Answer: 1832" or "Answer: yes". THE SHORTER, THE BETTER!!!


Here is my question:
{}

Here is my table data:
{}
        """

    def render_template(self, data):
        question = data["question"]
        table = data["support_doc"]
        return self.prompt.format(question, table)

    def parse_response(self, response, data):
        if "Answer:" not in response:
            raise ValueError(
                "unrecognized response format, should provide `Answer:` section"
            )
        return response.split("Answer: ")[1]


class CodeSolver:
    def __init__(self):
        #         self.prompt = """
        # I have a reading comprehension task in which each question is associated with a Wikipedia table. The table is given in the format of a list of python dicts, where each dict represents a row in the table, and the dict's keys and values correspond to the cell titles and their respective values. Accurately answering these questions requires performing various calculations on the table data, such as table lookup, aggregation, comparison (maximum, minimum), arithmetic operations, joins, and unions. Please follow these steps to provide the answer:

        # 1. Identify all rows and columns in the table that are helpful for answering the question.
        # 2. Determine the series of calculation operations that need to be performed on the table to answer the question.
        # 3. Write a Python function that takes a JSON file path as input, which stores the content of a data table, and outputs the answer to the question. Your code must ensure that the function syntax is correct and callable, and all necessary dependencies are imported. Additionally, only generate the function code, do not generate the calling code. Following is an example:

        # def func(table_file: str):
        #     table=json.load(open(table_file, "r"))
        #     # implement your solution here, please add necessary comments.
        #     pass

        # Here is my question:
        # {}

        # Here is my table data:
        # {}
        #         """
        self.prompt = """
I have a reading comprehension task in which each question is associated with a Wikipedia table. The table is given in the format of a list of python dicts, where each dict represents a row in the table, and the dict's keys and values correspond to the cell titles and their respective values. Accurately answering these questions requires performing various calculations on the table data, such as table lookup, aggregation, comparison (maximum, minimum), arithmetic operations, joins, and unions. Please follow these steps to provide the answer:


1. Analysis: Identify all rows and columns in the table that are helpful for answering the question.
2. Plan: Determine the series of calculation operations that need to be performed on the table to answer the question.
3. Reason: Execute the calculations identified in step two on the table data and provide the answer.
4. Verify: Write a Python function that takes a JSON file path as input, which stores the content of a data table, and outputs the answer to the question. Your code must ensure that the function syntax is correct and callable, and all necessary dependencies are imported. Additionally, only generate the function code, do not generate the calling code. Following is an example:
5. Answer: Based on above steps, conclude you response with "Answer: " to present a concise, definitive response, devoid of additional elaboration. Acceptable formats such as "Answer: 14 May", "Answer: 1832" or "Answer: yes". THE SHORTER, THE BETTER!!!


def func(table_file: str):
    table=json.load(open(table_file, "r"))
    # implement your solution here, please add necessary comments.
    pass

Here is my question:
{}

Here is my table data:
{}
        """

    def extract_python_code(self, api_response: str) -> str:
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, api_response, re.DOTALL)

        if matches:
            return matches[0].strip()
        else:
            return None

    def run_python_code(self, function_code, file_path):

        intp = MySubprocessInterpreter(require_confirm=False)
        execute_code = f"print(func('{file_path}'))"
        code = f"{function_code}\n{execute_code}"
        output = intp.run(code, code_type="python3")
        try:
            return eval(output.strip())
        except:
            return output.strip()

    def render_template(self, data):
        question = data["question"]
        table = data["support_doc"]
        return self.prompt.format(question, table)

    def parse_response(self, response, data):

        try:
            function_code = self.extract_python_code(response)
            data_id = data["id"]
            file_path = f"tmp/{data_id}.json"
            with open(file_path, "w") as writer:
                writer.write(json.dumps(data["support_doc"], ensure_ascii=False))
            output = self.run_python_code(function_code, file_path)
            return output
        except:
            if "**Answer**: " in response:
                return response.split("**Answer**: ")[1]
            elif "Answer: " in response:
                return response.split("Answer: ")[1]
            else:
                raise ValueError(
                    f"Both the Python code and the LLM reasoning failed to answer the question."
                )


def calculate_precision_recall_f1(prediction, label):

    TP = len(set(prediction) & set(label))
    FP = len(set(prediction) - set(label))
    FN = len(set(label) - set(prediction))
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    # print(f"{prediction} vs {label}, {precision}, {recall}, {f1}")
    return precision, recall, f1


def is_digit_and_chinese(s):
    # 正则表达式匹配规则：开始位置有一个或多个数字，后面跟着一个或多个汉字，直到字符串结束
    pattern = r"^\d+\p{Script=Han}+$"
    if re.match(pattern, s):
        return True
    else:
        return False


def get_score(gold, submit):
    # 计算得分
    precision_list = []
    recall_list = []
    f1_list = []

    for k, v in gold.items():
        label = v
        if k in submit:
            prediction = submit[k]

            if (
                label == ["没有"]
                or label == ["不是"]
                or label == ["否"]
                or label == ["无"]
                or label == ["不同"]
                or label == ["知识库未提及"]
            ):
                label = ["0"]
            if (
                prediction == ["没有"]
                or prediction == ["No"]
                or prediction == ["no"]
                or prediction == ["否"]
                or prediction == ["不属于"]
                or prediction == ["不是"]
                or prediction == ["不相同"]
                or prediction == ["No."]
                or prediction == ["不同"]
                or prediction == ["无"]
                or prediction == ["知识库未提及"]
            ):
                prediction = ["0"]

            # 处理 ["数字+单位"] 的情况, 例如 ["1个"]
            if (
                re.match(r"(\d+).*", label[0]) != None
                and re.match(r"(\d+).*", prediction[0]) != None
            ):
                label[0] = re.match(r"(\d+).*", label[0]).group(1)
                prediction[0] = re.match(r"(\d+).*", prediction[0]).group(1)

            precision, recall, f1 = calculate_precision_recall_f1(prediction, label)
        else:
            precision, recall, f1 = 0, 0, 0

        # if f1 != 1:
        #     print("问题：", k)
        #     print("标准答案：", label)
        #     #print("选手提交：", prediction)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
    return f1, f1_list


def get_llm_client():
    config = {
        "type": "maas",
        "api_key": "",
        "base_url": "https://api.deepseek.com/beta",
        "model": "deepseek-chat",
    }
    return LLMClient.from_config(config)


def load_data(file_path):
    return json.load(open(file_path, "r"))


@retry(stop=stop_after_attempt(3))
def qa(llm_client, data):
    # solver = Solver()
    solver = CodeSolver()
    prompt = solver.render_template(data)
    # print(f"prompt = {prompt}")
    data_id = data["id"]
    question = data["question"]
    response = llm_client(prompt)
    # print(f"response = {response}")
    answer = solver.parse_response(response, data)
    print(f"answer = {answer}")
    return (
        data_id,
        {
            "id": data_id,
            "question": question,
            "answer": answer,
            "input": prompt,
            "output": response,
        },
    )


def process(llm_client, data):
    try:
        return qa(llm_client, data)
    except:
        return None


def main():
    file_path = "./data/src/data/WTQ/test_full.json"
    ckpt = CheckpointerManager.get_checkpointer({"type": "zodb", "ckpt_dir": "ckpt"})
    llm_client = get_llm_client()
    data = load_data(file_path)
    print(f"llm_client = {llm_client}")
    futures = []
    with ThreadPoolExecutor(16) as executor:
        for item in data:
            item_id = item["id"]
            if ckpt.exists(item_id):
                continue
            fut = executor.submit(
                process,
                llm_client,
                item,
            )
            futures.append(fut)
    success = 0
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            data_id, answer_dict = result
            ckpt.write_to_ckpt(data_id, answer_dict)
            success += 1
    print(f"Done process all records, total: {len(data)}, success: {success}.")

    gold_answer_path = "data/src/gold_answer/WTQ.json"
    gold_answers = json.load(open(gold_answer_path, "r"))
    gold = {}
    for item in gold_answers:
        gold[item["id"]] = item["answer"]

    answers = []
    pred_test = {}
    gold_test = {}
    for k in ckpt.keys():
        answer = ckpt.read_from_ckpt(k)
        data_id = answer["id"]
        label = gold.get(data_id, None)
        answer["gold_answer"] = label
        pred_test[data_id] = (
            answer["answer"]
            if isinstance(answer["answer"], list)
            else [answer["answer"]]
        )
        answers.append(answer)
        if k in gold:
            gold_test[k] = gold[k]
    with open("answer.json", "w") as writer:
        writer.write(json.dumps(answers, indent=4, ensure_ascii=False))
    macro_f1, f1_list = get_score(gold_test, pred_test)
    incorrect = []
    for f1, answer in zip(f1_list, answers):
        if f1 != 1:
            incorrect.append(answer)
    keys_to_print = ["id", "question", "answer", "gold_answer"]
    # 提取数据并转换为表格格式
    table_data = [[item[key] for key in keys_to_print] for item in incorrect]

    # 打印表格
    print(
        tabulate(table_data, headers=keys_to_print, tablefmt="html", maxcolwidths=100)
    )

    print(f"score = {macro_f1}")

    CheckpointerManager.close()


if __name__ == "__main__":
    main()
