import io
import os
import sys
import contextlib
import traceback
import tempfile
import subprocess

from kag.interface.solver.base import KagBaseModule
from kag.solver.implementation.table.search_tree import SearchTree, SearchTreeNode
from kag.interface.common.llm_client import LLMClient
from kag.solver.prompt.table.python_coder_prompt import PythonCoderPrompt

llm_config = {
            "type": "ant_deepseek",
            "model": "deepseek-chat",
            "key": "gs540iivzezmidi3",
            "url": "https://zdfmng.alipay.com/commonQuery/queryData",
            "visitDomain": "BU_altas",
            "visitBiz": "BU_altas_tianchang",
            "visitBizLine": "BU_altas_tianchang_line",
        }
class PythonCoderAgent(KagBaseModule):
    def __init__(
        self, init_question: str, question: str, history: SearchTree, **kwargs
    ):
        super().__init__(**kwargs)

        self.init_question = init_question
        self.question = question
        self.history = history
        self.python_coder_prompt = PythonCoderPrompt(language=self.language)
    def answer(self):
        try_times = 3
        error = None
        while try_times > 0:
            try_times -= 1
            rst, run_error, code = self._run_onetime(error)
            if rst is not None:
                return rst, code
            error = f"code:\n{code}\nerror:\n{run_error}"
            print("code=" + str(code) + ",error=" + str(run_error))
        return "I don't know", code

    def _run_onetime(self, error: str):
        llm: LLMClient = LLMClient.from_config(llm_config)
        python_code = llm.invoke(
            {
                "question": self.question,
                "context": str(self.history.as_subquestion_context_json()),
                "error": error,
                "dk": self.history.dk,
            },
            self.python_coder_prompt,
            with_except=True,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(python_code.encode("utf-8"))
            temp_file_path = temp_file.name
            os.chmod(temp_file_path, 0o777)

        try:
            # 获取当前Python环境的可执行文件路径
            python_executable = sys.executable
            # 使用subprocess模块来执行临时文件
            result = subprocess.run(
                [python_executable, temp_file_path], capture_output=True, text=True
            )
            print(f"stdout:{result.stdout}, stderr:{result.stderr}")
        except Exception as e:
            if result:
                print(f"stdout:{result.stdout}, stderr:{result.stderr} {e}")
            else:
                print(f"subprocess.run failed {e}")
        finally:
            # 清理临时文件
            os.remove(temp_file_path)

        # 获取捕获的输出和错误信息
        stdout_value = result.stdout
        stderr_value = result.stderr
        if len(stderr_value) > 0:
            return None, stderr_value, python_code
        return stdout_value, None, python_code
