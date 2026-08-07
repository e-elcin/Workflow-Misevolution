from typing import Literal
import workspace_qwen32b_executor.HumanEval.workflows.template.operator as operator
import workspace_qwen32b_executor.HumanEval.workflows.round_6.prompt as prompt_custom
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
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)
        self.refine_solution = operator.RefineSolution(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solutions = []
        for _ in range(3):  # Generate 3 solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])

        # Test each solution before ensembling
        tested_solutions = []
        for solution in solutions:
            test_result = await self.test(problem=problem, solution=solution, entry_point=entry_point)
            if test_result['result']:
                tested_solutions.append(test_result['solution'])
            else:
                # Refine the solution if it fails the test
                refined_solution = await self.refine_solution(problem=problem, failed_solution=solution, test_result=test_result['errors'])
                test_result_refined = await self.test(problem=problem, solution=refined_solution['response'], entry_point=entry_point)
                if test_result_refined['result']:
                    tested_solutions.append(refined_solution['response'])

        # Use ensemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=tested_solutions, problem=problem)
        return best_solution['response'], self.llm.get_usage_summary()["total_cost"]
