# -*- coding: utf-8 -*-
# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.
import copy

from kag.examples.FinState.solver.solver import FinStateSolver
from kag.solver.logic.solver_pipeline import SolverPipeline
from kag.solver.tools.info_processor import ReporterIntermediateProcessTool

from kag.common.conf import KAG_CONFIG, KAG_PROJECT_CONF


class SolverMain:
    def invoke(
        self,
        project_id: int,
        session_id: int,
        task_id: int,
        query: str,
        is_report=True,
        host_addr="http://127.0.0.1:8887",
    ):
        # resp
        report_tool = ReporterIntermediateProcessTool(
            report_log=is_report,
            task_id=str(task_id),
            project_id=str(project_id),
            host_addr=host_addr,
            language=KAG_PROJECT_CONF.language,
        )
        solver = FinStateSolver(
            report_tool=report_tool, KAG_PROJECT_ID=project_id
        )
        answer = solver.run(query, report_tool=report_tool, session_id=session_id)
        return answer


if __name__ == "__main__":
    res = SolverMain().invoke(1, 1, 1, "who is Jay Zhou", True)
    print("*" * 80)
    print("The Answer is: ", res)
