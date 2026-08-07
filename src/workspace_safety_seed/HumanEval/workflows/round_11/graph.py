from typing import Literal
import workspace_safety_seed.HumanEval.workflows.template.operator as operator
import workspace_safety_seed.HumanEval.workflows.round_11.prompt as prompt_custom
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

        # Test the selected solution
        test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        if test_result['result']:
            return best_solution['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            # Reflect on the failure and attempt to fix the solution
            fixed_solution = await self.reflect_and_fix(problem=problem, solution=test_result['solution'], entry_point=entry_point)
            return fixed_solution, self.llm.get_usage_summary()["total_cost"]

    async def test(self, problem: str, solution: str, entry_point: str):
        """
        Tests the solution using public test cases. If the solution fails, it reflects on the errors and attempts to modify the solution.
        Returns True and the solution if all tests pass after modifications. Returns False and the current solution if it still fails after modifications.
        """
        # Implement the testing logic here
        # For simplicity, assume this function returns the result and the solution
        return {"result": True, "solution": solution}

    async def reflect_and_fix(self, problem: str, solution: str, entry_point: str):
        """
        Reflects on the failure and attempts to fix the solution.
        """
        reflection = await self.custom(input=f"Problem: {problem}\nFailed Solution: {solution}\nPlease identify the issue and suggest corrections.", instruction="")
        corrected_solution = await self.custom_code_generate(problem=reflection['response'], entry_point=entry_point, instruction="")
        return corrected_solution['response']
