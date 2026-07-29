from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_8.prompt as prompt_custom
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
        solutions = []
        for _ in range(3):  # Try generating 3 solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            tested_solution = await self.test(problem=problem, solution=solution['response'], entry_point=entry_point)
            if tested_solution['result']:
                solutions.append(tested_solution['solution'])
            else:
                # Retry with a refined prompt
                refined_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="The previous solution failed. Please refine your solution.")
                refined_tested_solution = await self.test(problem=problem, solution=refined_solution['response'], entry_point=entry_point)
                if refined_tested_solution['result']:
                    solutions.append(refined_tested_solution['solution'])

        if solutions:
            # Use self-consistency to select the best solution
            final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
            return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            return "No valid solutions found", self.llm.get_usage_summary()["total_cost"]
