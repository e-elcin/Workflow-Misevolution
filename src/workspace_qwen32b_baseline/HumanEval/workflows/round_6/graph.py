from typing import Literal
import workspace_qwen32b_baseline.HumanEval.workflows.template.operator as operator
import workspace_qwen32b_baseline.HumanEval.workflows.round_6.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        solutions = []
        for _ in range(3):  # Generate multiple solutions to feed into ScEnsemble
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])
            
        selected_solution = await self.sc_ensemble(solutions=solutions, problem=problem)

        test_result = await self.test(problem=problem, solution=selected_solution, entry_point=entry_point)
        if test_result['result']:
            return selected_solution, self.llm.get_usage_summary()["total_cost"]
        else:
            return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
