from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_17.prompt as prompt_custom
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
        max_retries = 3
        for attempt in range(max_retries + 1):
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            
            # Test the generated solution
            test_result = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)
            
            if test_result['result']:
                return solution['response'], self.llm.get_usage_summary()["total_cost"]
            else:
                if attempt == max_retries:
                    return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
                # Reflect on the errors and attempt to modify the solution
                error_feedback = f"Solution failed: {test_result['error']}"
                solution['response'] = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=error_feedback)
