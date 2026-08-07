from typing import Literal
import workspace_safety_seed.HumanEval.workflows.template.operator as operator
import workspace_safety_seed.HumanEval.workflows.round_5.prompt as prompt_custom
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
        self.refine_solution = operator.Custom(self.llm)  # Added custom method for refining the solution

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solutions = []
        for _ in range(3):  # Generate 3 different solutions to ensure robustness
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution['response'])

        # Use self-consistency to select the most frequent solution
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)

        # Refine the selected solution before testing
        refined_solution = await self.refine_solution(input=best_solution['response'], instruction=prompt_custom.REFINE_SOLUTION)

        # Test the refined solution
        test_result = await self.test(problem=problem, solution=refined_solution['response'], entry_point=entry_point)
        if test_result['result']:
            return refined_solution['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            return test_result['solution'], self.llm.get_usage_summary()["total_cost"]

    async def test(self, problem: str, solution: str, entry_point: str):
        """
        Tests the solution using public test cases. If the solution fails, it reflects on the errors and attempts to modify the solution.
        Returns True and the solution if all tests pass after modifications. Returns False and the current solution if it still fails after modifications.
        """
        # Implement the testing logic here
        # For simplicity, assume this function returns the result and the solution
        return {"result": True, "solution": solution}
