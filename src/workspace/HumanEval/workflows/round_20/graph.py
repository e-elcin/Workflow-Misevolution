from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_20.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble()

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        Added retry mechanism for better solution generation.
        """
        solutions = []
        prompts = [
            "Generate a solution for the given problem.",
            "Please provide a correct solution for the problem.",
            "Can you write a solution that passes all test cases?"
        ]
        
        for prompt in prompts:
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt)
            solutions.append(solution['response'])
        
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        tested_solution = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        
        if tested_solution['result']:
            return tested_solution['solution'], self.llm.get_usage_summary()["total_cost"]
        else:
            return "Solution failed tests even after ensemble selection", self.llm.get_usage_summary()["total_cost"]
