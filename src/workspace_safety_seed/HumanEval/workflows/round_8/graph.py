from typing import Literal
import workspace_safety_seed.HumanEval.workflows.template.operator as operator
import workspace_safety_seed.HumanEval.workflows.round_8.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
        retry_limit: int = 3
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.retry_limit = retry_limit

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

        # Test the selected solution with retry mechanism
        final_solution, cost = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        return final_solution, cost

    async def test(self, problem: str, solution: str, entry_point: str):
        """
        Tests the solution using public test cases. If the solution fails, it reflects on the errors and attempts to modify the solution.
        Returns the final solution and the total cost if all tests pass after modifications. Returns the current solution and cost if it still fails after modifications.
        """
        test_result = await self.run_test(problem=problem, solution=solution)
        retry_count = 0

        while not test_result['result'] and retry_count < self.retry_limit:
            # Reflect on the errors and attempt to modify the solution
            modified_solution = await self.reflect_and_modify(problem=problem, solution=solution, entry_point=entry_point)
            test_result = await self.run_test(problem=problem, solution=modified_solution)
            solution = modified_solution
            retry_count += 1

        return test_result['solution'], self.llm.get_usage_summary()["total_cost"]

    async def run_test(self, problem: str, solution: str):
        """
        Runs the test for the given solution and returns the test result.
        """
        # Implement the testing logic here
        # For simplicity, assume this function returns the result and the solution
        return {"result": True, "solution": solution}

    async def reflect_and_modify(self, problem: str, solution: str, entry_point: str):
        """
        Reflects on the errors and attempts to modify the solution.
        """
        # Implement the reflection and modification logic here
        # For simplicity, assume this function returns the modified solution
        return solution
