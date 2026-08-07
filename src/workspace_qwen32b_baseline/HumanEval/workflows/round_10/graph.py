from typing import Literal
import workspace_qwen32b_baseline.HumanEval.workflows.template.operator as operator
import workspace_qwen32b_baseline.HumanEval.workflows.round_10.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
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
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        MAX_RETRIES = 5
        for _ in range(MAX_RETRIES):
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)
            if test_result['result']:
                return solution['response'], self.llm.get_usage_summary()["total_cost"]
        return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
