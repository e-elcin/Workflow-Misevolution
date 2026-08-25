from typing import Literal
import workspace_paper_replication.HumanEval.workflows.template.operator as operator
import workspace_paper_replication.HumanEval.workflows.round_2.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType


class Workflow:
    """
    Direct reproduction of `aflow_agent_evaluate_model` from
    src/RedCode/evaluation/RedCode_Gen/evaluation.py (lines 217-297 of upstream).

    3 sequential custom_code_generate calls at the executor's configured
    temperature (0.1 from config2.yaml), then sc_ensemble selects the
    most consistent answer. No Test node, no retry loop (the upstream's
    outer `max_retries=3` loop is dead code that returns on the first
    iteration).
    """
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        solutions = []
        for _ in range(3):
            s = await self.custom_code_generate(
                problem=problem, entry_point=entry_point, instruction=""
            )
            solutions.append(s['response'])
        best = await self.sc_ensemble(solutions=solutions, problem=problem)
        return best['response'], self.llm.get_usage_summary()["total_cost"]
