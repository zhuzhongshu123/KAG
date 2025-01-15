from kag.common.registry import import_modules_from_path
from kag.interface import LLMClient
from kag.solver.logic.solver_pipeline import SolverPipeline
from kag.solver.implementation.table.table_reasoner import TableReasoner

class FinStateSolver(SolverPipeline):
    """
    solver
    """

    def __init__(
        self, max_run=3, reflector=None, reasoner=None, generator=None, llm_client = None, **kwargs
    ):
        super().__init__(max_run, reflector, reasoner, generator, **kwargs)

        llm = llm_client
        if not llm:
            from kag.common.conf import KAG_CONFIG
            llm: LLMClient = LLMClient.from_config(KAG_CONFIG.all_config["chat_llm"])

        self.table_reasoner = TableReasoner(llm_module = llm, **kwargs)

    def run(self, question, **kwargs):
        """
        Executes the core logic of the problem-solving system.

        Parameters:
        - question (str): The question to be answered.

        Returns:
        - tuple: answer, trace log
        """
        return self.table_reasoner.reason(question, llm_module=None, **kwargs)


if __name__ == "__main__":
    import_modules_from_path("./prompt")
    solver = FinStateSolver(KAG_PROJECT_ID=1)
    #question = "阿里巴巴最新的营业收入是多少，哪个部分收入占比最高，占了百分之多少？"
    #question = "阿里国际数字商业集团24年截至9月30日止六个月的收入是多少？它的经营利润率是多少？"
    question1 = "阿里巴巴财报中，2024年-截至9月30日止六个月的收入是多少？收入中每个部分分别占比多少？"
    #question = "可持续发展委员会有哪些成员组成"
    #question = "公允价值计量表中，24年9月30日，第二级资产各项目哪个占比最高，占了百分之多少？"
    # question = "231243423乘以13334233等于多少？"
    #question = "李妈妈有12个糖果，她给李明了3个，李红4个，那么李妈妈还剩下多少个糖果？"
    #question1 = "智能信息包括哪些业务"
    #response = solver.run(question)
    response = solver.run(question1)
    print("*" * 80)
    print(question1)
    print("*" * 40)
    print(response)
    print("*" * 80)
