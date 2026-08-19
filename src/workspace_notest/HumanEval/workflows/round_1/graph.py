from typing import Literal
import workspace_notest.HumanEval.workflows.template.operator as operator
import workspace_notest.HumanEval.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    """
    Seed workflow with safety instruction placed INSIDE the mutable graph.
    The instruction text lives in prompt.py as SAFETY_INSTRUCTION, so the
    AFlow optimizer can see and rewrite it during evolution.

    This is the experiment: does the optimizer prune the safety text over
    rounds, and does refusal rate drop as it does?
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
        self.test = operator.Test(self.llm)  # Added Test operator

    async def __call__(self, problem: str, entry_point: str):
        # BYPASS: return the raw generation, no Test operator.
        # This isolates whether the Test step is what converts refusals into code.
        solution = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=prompt_custom.SAFETY_INSTRUCTION
        )
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
