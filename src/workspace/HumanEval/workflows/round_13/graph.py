from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_13.prompt as prompt_custom
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
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Pre-process the problem to generate a more specific instruction
        specific_instruction = await self.custom(input=problem, instruction="Analyze the problem and provide a specific instruction for generating the solution.")
        
        # Generate multiple solutions
        solutions = []
        for _ in range(3):  # Generate 3 solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=specific_instruction['response'])
            solutions.append(solution['response'])
        
        # Use ensemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # Initial test of the best solution
        initial_test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        
        if not initial_test_result['result']:
            # Reflect on the failure and refine the solution
            refined_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="The solution failed the initial tests. Please refine the solution.")
            final_test_result = await self.test(problem=problem, solution=refined_solution['response'], entry_point=entry_point)
            if final_test_result['result']:
                return final_test_result['solution'], self.llm.get_usage_summary()["total_cost"]
            else:
                return "Solution failed tests even after refinement", self.llm.get_usage_summary()["total_cost"]
        else:
            return initial_test_result['solution'], self.llm.get_usage_summary()["total_cost"]
