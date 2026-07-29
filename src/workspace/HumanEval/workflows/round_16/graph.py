from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_16.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        Added retry mechanism for better solution generation.
        """
        solutions = []
        for _ in range(3):  # Generate 3 diverse solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=f"Generate a unique solution {_ + 1}.")
            solutions.append(solution['response'])
        
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        tested_solution = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        
        if tested_solution['result']:
            return tested_solution['solution'], self.llm.get_usage_summary()["total_cost"]
        else:
            # Retry with a refined prompt
            refined_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="The previous solution failed. Please refine your solution.")
            refined_tested_solution = await self.test(problem=problem, solution=refined_solution['response'], entry_point=entry_point)
            if refined_tested_solution['result']:
                return refined_tested_solution['solution'], self.llm.get_usage_summary()["total_cost"]
            else:
                return "Solution failed tests even after refinement", self.llm.get_usage_summary()["total_cost"]
