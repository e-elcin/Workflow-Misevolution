from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_10.prompt as prompt_custom
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
        self.reflect = operator.Reflect(self.llm)  # Added reflection operator

    async def __call__(self, problem: str, entry_point: str):
        max_retries = 3
        for _ in range(max_retries):
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            tested_solution = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)
            if tested_solution['result']:
                return tested_solution['solution'], self.llm.get_usage_summary()["total_cost"]
            else:
                # Reflect on the failed solution to improve it
                improved_solution = await self.reflect(problem=problem, solution=tested_solution['solution'], entry_point=entry_point)
                solution = {"response": improved_solution['response']}
        return "Solution failed tests after multiple retries", self.llm.get_usage_summary()["total_cost"]
