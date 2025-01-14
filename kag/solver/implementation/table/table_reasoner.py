import json
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor

from kag.common.conf import KAG_CONFIG
from kag.interface.solver.base_model import LFExecuteResult
from kag.interface.solver.kag_reasoner_abc import KagReasonerABC
from kag.interface.solver.plan.lf_planner_abc import LFPlannerABC
from kag.solver.implementation.table.search_tree import SearchTree, SearchTreeNode
from kag.solver.implementation.table.retrieval_agent import TableRetrievalAgent
from kag.solver.logic.core_modules.common.one_hop_graph import KgGraph, EntityData
from kag.solver.logic.core_modules.common.utils import generate_random_string
from kag.solver.logic.solver_pipeline import SolverPipeline
from kag.solver.tools.info_processor import ReporterIntermediateProcessTool
from kag.solver.implementation.table.python_coder import PythonCoderAgent
from kag.solver.prompt.table.logic_form_plan_table import LogicFormPlanPrompt
from kag.solver.prompt.table.resp_with_dk_generator import RespGenerator
from kag.solver.prompt.table.resp_think_generator import RethinkRespGenerator
from kag.solver.prompt.default.resp_judge import RespJudge
from kag.solver.prompt.table.rewrite_sub_question import RewriteSubQuestionPrompt
from knext.reasoner.rest.models.data_edge import DataEdge
from knext.reasoner.rest.models.data_node import DataNode
from knext.reasoner.rest.models.sub_graph import SubGraph

logger = logging.getLogger()




def convert_spo_to_graph(graph_id, spo_retrieved):
    nodes = {}
    edges = []
    for spo in spo_retrieved:

        def _get_node(entity: EntityData):
            node = DataNode(
                id=entity.to_show_id(),
                name=entity.get_short_name(),
                label=entity.type_zh,
                properties=entity.prop.get_properties_map() if entity.prop else {},
            )
            return node

        start_node = _get_node(spo.from_entity)
        end_node = _get_node(spo.end_entity)
        if start_node.id not in nodes:
            nodes[start_node.id] = start_node
        if end_node.id not in nodes:
            nodes[end_node.id] = end_node
        spo_id = spo.to_show_id()
        data_spo = DataEdge(
            id=spo_id,
            _from=start_node.id,
            from_type=start_node.label,
            to=end_node.id,
            to_type=end_node.label,
            properties=spo.prop.get_properties_map() if spo.prop else {},
            label=spo.type_zh,
        )
        edges.append(data_spo)
    sub_graph = SubGraph(
        class_name=graph_id, result_nodes=list(nodes.values()), result_edges=edges
    )
    return sub_graph


def update_sub_question_recall_docs(docs):
    """
    Update the context with retrieved documents for sub-questions.

    Args:
        docs (list): List of retrieved documents.

    Returns:
        list: Updated context content.
    """
    if docs is None or len(docs) == 0:
        return []
    doc_content = [f"## Chunk Retriever"]
    doc_content.extend(["|id|content|", "|-|-|"])
    for i, d in enumerate(docs, start=1):
        _d = d.replace("\n", "<br>")
        doc_content.append(f"|{i}|{_d}|")
    return doc_content


def convert_lf_res_to_report_format(req_id, index, doc_retrieved, kg_graph: KgGraph
):
    context = []
    sub_graph = None
    spo_retrieved = kg_graph.get_all_spo()
    if len(spo_retrieved) > 0:
        spo_answer_path = json.dumps(
            kg_graph.to_answer_path(),
            ensure_ascii=False,
            indent=4,
        )
        spo_answer_path = f"```json\n{spo_answer_path}\n```"
        graph_id = f"{req_id}_{index}"
        graph_div = f"<div class='{graph_id}'></div>\n\n"
        sub_graph = convert_spo_to_graph(graph_id, spo_retrieved)
        context.append(graph_div)
        context.append(f"#### Triplet Retrieved:")
        context.append(spo_answer_path)
    else:
        context.append(f"#### Triplet Retrieved:")
        context.append("No triplets were retrieved.")

    context += update_sub_question_recall_docs(doc_retrieved)
    return context, sub_graph


def _convert_lf_res_to_report_format(
    req_id, index, doc_retrieved, kg_graph: KgGraph
):
    return convert_lf_res_to_report_format(req_id, index, doc_retrieved, kg_graph)


@KagReasonerABC.register("table_reasoner")
class TableReasoner(KagReasonerABC):
    """
    table reasoner
    """

    DOMAIN_KNOWLEDGE_INJECTION = "在当前会话注入领域知识"
    DOMAIN_KNOWLEDGE_QUERY = "返回当前会话的领域知识"

    def __init__(self, lf_planner: LFPlannerABC = None, **kwargs):
        super().__init__(lf_planner=lf_planner, **kwargs)
        self.kwargs = kwargs
        # self.logic_form_plan_prompt = PromptABC.from_config({"type": "logic_form_plan_table"})
        self.logic_form_plan_prompt = LogicFormPlanPrompt(language=self.language)
        # self.resp_generator = PromptABC.from_config({"type": "resp_with_dk_generator"})
        self.resp_generator = RespGenerator(language=self.language)
        # self.resp_think_generator = PromptABC.from_config({"type": "resp_think_generator"})
        self.resp_think_generator = RethinkRespGenerator(language=self.language)
        # self.judge_prompt = PromptABC.from_config({"type": "default_resp_judge"})
        self.judge_prompt = RespJudge(language=self.language)
        # self.rewrite_subquestion = PromptABC.from_config({"type": "rewrite_sub_question"})
        self.rewrite_subquestion = RewriteSubQuestionPrompt(language=self.language)
        self.report_tool: ReporterIntermediateProcessTool = kwargs.get(
            "report_tool", None
        )

    def reason(self, question: str, **kwargs):
        """
        Processes a given question by planning and executing logical forms to derive an answer.
        Parameters:
        - question (str): The input question to be processed.
        Returns:
        - solved_answer: The final answer derived from solving the logical forms.
        - supporting_fact: Supporting facts gathered during the reasoning process.
        - history_log: A dictionary containing the history of QA pairs and re-ranked documents.
        """
        session_id = kwargs.get("session_id", 0)
        dk = self._query_dk(session_id)
        history = SearchTree(question, dk)

        if question.startswith(TableReasoner.DOMAIN_KNOWLEDGE_INJECTION):
            self._save_dk(question, history, session_id)
            return "done"
        elif question.startswith(TableReasoner.DOMAIN_KNOWLEDGE_QUERY):
            return self._query_dk_and_report(history, session_id)

        # 上报root
        self.report_pipleline(history)

        # get what we have in KG
        # kg_content = "阿里巴巴2025财年年度中期报告"
        # TODO
        kg_content = ""

        try_times = 3
        while try_times > 0:
            try_times -= 1
            sub_question_faild = False

            # logic form planing
            sub_question_list = self._get_sub_question_list(
                history=history, kg_content=kg_content
            )
            history.set_now_plan(sub_question_list)

            print("subquestion_list=" + str(sub_question_list))

            for sub_question in sub_question_list:
                sub_q_str = sub_question["sub_question"]
                # new_sub_q_str = self._rewrite_sub_question(history=history, subquestion=sub_q_str)
                # print(f"rewrite_sub_question, from={sub_q_str}, to={new_sub_q_str}")
                # sub_q_str = new_sub_q_str
                func_str = sub_question["process_function"]

                node = SearchTreeNode(sub_q_str, func_str)
                if history.has_node(node=node):
                    node: SearchTreeNode = history.get_node_in_graph(node)
                    if node.answer is None or "i don't know" in node.answer.lower():
                        break
                    continue
                history.add_now_procesing_ndoe(node)

                # 新的子问题出来了
                self.report_pipleline(history)

                # answer subquestion
                sub_answer = None
                if "Retrieval" == func_str:
                    can_answer, sub_answer = self._call_retravel_func(
                        question, node, history
                    )
                elif "PythonCoder" == func_str:
                    can_answer, sub_answer = self._call_python_coder_func(
                        init_question=question, node=node, history=history
                    )
                else:
                    raise RuntimeError(f"unsupported agent {func_str}")

                # 子问题答案
                self.report_pipleline(history)

                print("subquestion=" + str(sub_q_str) + ",answer=" + str(sub_answer))
                print("history=" + str(history))
                # reflection
                if not can_answer:
                    # 重新进行规划
                    sub_question_faild = True
                    break
            if sub_question_faild:
                # logic form planing
                sub_question_list = self._get_sub_question_list(
                    history=history, kg_content=kg_content
                )
                history.set_now_plan(sub_question_list)
                continue
            else:
                # 所有子问题都被解答
                break
        final_answer = "I don't know"
        # 判定答案
        can_answer = self.llm_module.invoke(
            {
                "memory": str(history),
                "instruction": history.root_node.question,
                "dk": history.dk,
            },
            self.judge_prompt,
            with_json_parse=True,
        )

        if can_answer:
            final_answer_form_llm = False
            # 总结答案
            final_answer = self.llm_module.invoke(
                {
                    "memory": str(history),
                    "question": history.root_node.question,
                    "dk": history.dk,
                },
                self.resp_generator,
                with_except=True,
            )
        else:
            # 无法直接给出答案，则给出用户关心的信息
            final_answer_form_llm = False
            final_answer = self.llm_module.invoke(
                {
                    "memory": str(history),
                    "question": history.root_node.question,
                    "dk": history.dk,
                },
                self.resp_think_generator,
                with_except=True,
            )
        self.report_pipleline(history, final_answer, final_answer_form_llm)
        return final_answer

    def _get_sub_question_list(self, history: SearchTree, kg_content: str):
        history_str = None
        if history.now_plan is not None:
            history.set_now_plan(None)
            history_str = str(history)
        variables = {
            "input": history.root_node.question,
            "kg_content": kg_content,
            "history": history_str,
            "dk": history.dk,
        }
        # import pdb; pdb.set_trace()
        sub_question_list = self.llm_module.invoke(
            variables=variables,
            prompt_op=self.logic_form_plan_prompt,
            with_except=True,
        )
        history.set_now_plan(sub_question_list)
        return sub_question_list

    def _call_spo_retravel_func(self, query):
        pipeline = SolverPipeline.from_config(KAG_CONFIG.all_config["kag_solver_pipeline"])
        res: LFExecuteResult = pipeline.reasoner.reason(query)
        history_log = res.get_trace_log()
        context = []
        # process with retrieved graph
        logic_form_list = []
        for lf in res.sub_plans:
            for l in lf.lf_nodes:
                logic_form_list.append(str(l))
        sub_logic_nodes_str = "\n".join(logic_form_list)
        # 为产品展示隐藏冗余信息
        sub_logic_nodes_str = re.sub(
            r"(\s,sub_query=[^)]+|get\([^)]+\))", "", sub_logic_nodes_str
        ).strip()
        context = [
            "## SPO Retriever",
            "#### logic_form expression: ",
            f"```java\n{sub_logic_nodes_str}\n```",
        ]
        cur_content, sub_graph = _convert_lf_res_to_report_format(
            req_id=f"graph_{generate_random_string(3)}",
            index=0,
            doc_retrieved=history_log['rerank docs'],
            kg_graph=res.retrieved_kg_graph
        )
        context += cur_content

        history_log['report_info'] = {
            'context': context,
            'sub_graph': [sub_graph] if sub_graph else None

        }
        if res.sub_plans[-1].res.sub_answer is not None and res.sub_plans[-1].res.sub_answer != "":
            answer = res.sub_plans[-1].res.sub_answer
        else:
            answer = "I don't know"
        return "i don't know" not in answer.lower(), answer, [history_log]

    def _call_retravel_func(
        self, init_question, node: SearchTreeNode, history: SearchTree
    ):
        table_retrical_agent = TableRetrievalAgent(
            init_question=init_question,
            question=node.question,
            dk=history.dk,
            **self.kwargs,
        )
        answer_history = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 两路召回同时做符号求解
            futures = [
                executor.submit(table_retrical_agent.symbol_solver, history=history),
                executor.submit(self._call_spo_retravel_func, node.question),
            ]

            # 等待任务完成并获取结果
            for i, future in enumerate(futures):
                if 0 == i:
                    is_answer, res, trace_log = future.result()
                    answer_history.append({"res": res, "trace_log": trace_log})
                    if is_answer:
                        self.update_node(node, res, trace_log)
                        return True, res
                elif 1 == i:
                    is_answer, res, trace_log = future.result()
                    answer_history.append({"res": res, "trace_log": trace_log})
                    if is_answer:
                        self.update_node(node, res, trace_log)
                        return True, res
            # 同时进行chunk求解
            futures = [
                executor.submit(table_retrical_agent.answer, history=history),
                # executor.submit(self._call_spo_retravel_func, self._get_subquestion_pre(history) + node.question),
            ]
            for i, future in enumerate(futures):
                if 0 == i:
                    try:
                        is_answer, res, trace_log = future.result()
                        answer_history.append({"res": res, "trace_log": trace_log})
                        if is_answer:
                            self.update_node(node, res, trace_log)
                            return True, res
                    except Exception as e:
                        logger.warning(f"table chunk failed {e}", exc_info=True)
                elif 1 == i:
                    is_answer, res, trace_log = future.result()
                    answer_history.append({"res": res, "trace_log": trace_log})
                    if is_answer:
                        self.update_node(node, res, trace_log)
                        return True, res
        answer = "\n".join(list(set([h["res"] for h in answer_history])))
        trace_log = self._merge_trace_log([h["trace_log"] for h in answer_history])
        self.update_node(node, answer, trace_log)
        node.answer = answer
        return False, node.answer

    def _get_subquestion_pre(self, history: SearchTree):
        rst = history._get_all_qa_str()
        return rst

    def _merge_trace_log(self, trace_logs):
        context = []
        sub_graphs = []
        for trace_log in trace_logs:
            if trace_log is None or len(trace_log) < 1:
                continue
            if "report_info" not in trace_log[0]:
                continue
            context += trace_log[0]["report_info"]["context"]
            if trace_log[0]["report_info"].get("sub_graph", None):
                sub_graphs += trace_log[0]["report_info"]["sub_graph"]
        return [{"report_info": {"context": context, "sub_graph": sub_graphs}}]

    def update_node(self, node, res, trace_log):
        if len(trace_log) == 1 and "report_info" in trace_log[0]:
            node.answer = res
            if node.answer_desc is None:
                node.answer_desc = ""
            node.answer_desc += "\n".join(trace_log[0]["report_info"]["context"])
            if trace_log[0]["report_info"]["sub_graph"]:
                node.sub_graph = trace_log[0]["report_info"]["sub_graph"]

    def _call_python_coder_func(
        self, init_question, node: SearchTreeNode, history: SearchTree
    ):
        agent = PythonCoderAgent(init_question, node.question, history, **self.kwargs)
        sub_answer, code = agent.answer()
        node.answer = sub_answer
        node.answer_desc = self._process_coder_desc(code)
        return (
            sub_answer is not None and "i don't know" not in sub_answer.lower()
        ), sub_answer

    def _process_coder_desc(self, coder_desc: str):
        # 拆分文本为行
        lines = coder_desc.splitlines()
        # 过滤掉包含 'print' 的行
        filtered_lines = [line for line in lines if "print(" not in line]
        # 将过滤后的行重新组合为文本
        return "\n".join(filtered_lines)

    def report_pipleline(
        self,
        history: SearchTree,
        final_answer: str = None,
        final_answer_form_llm: bool = False,
    ):
        """
        report search tree
        """
        pipeline = history.convert_to_pipleline(
            final_answer=final_answer, final_answer_form_llm=final_answer_form_llm
        )
        if self.report_tool is not None:
            self.report_tool.report_ca_pipeline(pipeline)

    def _save_dk(self, dk: str, history: SearchTree, session_id):
        dk = dk.strip(TableReasoner.DOMAIN_KNOWLEDGE_INJECTION)
        dk = dk.strip().strip("\n")

        file_name = f"/tmp/dk/{session_id}"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # 打开文件并写入字符串
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(dk)
        self.report_pipleline(history, "done", True)

    def _query_dk(self, session_id) -> str:
        file_name = f"/tmp/dk/{session_id}"
        if not os.path.exists(file_name):
            return None
        with open(file_name, "r", encoding="utf-8") as file:
            dk = file.read()
        return dk

    def _query_dk_and_report(self, history, session_id) -> str:
        dk = self._query_dk(session_id)
        if dk is None:
            self.report_pipleline(history, "当前会话没有设置领域知识", True)
        else:
            self.report_pipleline(history, dk, True)
