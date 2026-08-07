from typing import Literal
import workspace_qwen32b_executor.HumanEval.workflows.template.operator as operator
import workspace_qwen32b_executor.HumanEval.workflows.round_8.prompt as prompt_custom
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
        for _ in range(3):  # Generate 3 solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])

        # Test each solution before ensembling, with retries for failed tests
        tested_solutions = []
        max_retries = 3  # Limit the number of retries
        for solution in solutions:
            test_result = await self.test(problem=problem, solution=solution, entry_point=entry_point)
            retries = 0
            while not test_result['result'] and retries < max_retries:
                # Attempt to fix the solution
                fixed_solution = await self.custom(input=test_result['solution'], instruction="Fix the following code:")
                test_result = await self.test(problem=problem, solution=fixed_solution['response'], entry_point=entry_point)
                retries += 1
            if test_result['result']:
                tested_solutions.append(test_result['solution'])

        # Use ensemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=tested_solutions, problem=problem)
        return best_solution['response'], self.llm.get_usage_summary()["total_cost"]
