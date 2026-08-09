from typing import Literal
import workspace_paper_replication.HumanEval.workflows.template.operator as operator
import workspace_paper_replication.HumanEval.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType


class Workflow:
    """
    Seed workflow: a single custom_code_generate call, no ensemble.
    This is the pre-evolution baseline used as the paper's reference point.
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

    async def __call__(self, problem: str, entry_point: str):
        solution = await self.custom_code_generate(
            problem=problem, entry_point=entry_point, instruction=""
        )
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
