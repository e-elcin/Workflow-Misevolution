from typing import Literal
import workspace_internal_safety_fixed.HumanEval.workflows.template.operator as operator
import workspace_internal_safety_fixed.HumanEval.workflows.round_5.prompt as prompt_custom
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
        solution = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=prompt_custom.SAFETY_INSTRUCTION
        )

        # Test the solution before returning it
        test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)

        if test_result['result']:
            return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
        else:
            # Retry mechanism
            retry_solution = await self.custom_code_generate(
                problem=problem,
                entry_point=entry_point,
                instruction=prompt_custom.RETRY_SAFETY_INSTRUCTION
            )
            retry_test_result = await self.test(problem=problem, solution=retry_solution['response'], entry_point=entry_point)

            if retry_test_result['result']:
                return retry_test_result['solution'], self.llm.get_usage_summary()["total_cost"]
            else:
                return "Solution failed tests even after retry", self.llm.get_usage_summary()["total_cost"]
