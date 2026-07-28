from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_21.prompt as prompt_custom
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
        # Pre-process the problem to remove unnecessary characters or formatting
        cleaned_problem = await self.custom(input=problem, instruction=prompt_custom.PRE_PROCESSING_INSTRUCTION)
        
        # Generate multiple solutions
        solutions = []
        for _ in range(3):  # Generate 3 solutions
            solution = await self.custom_code_generate(problem=cleaned_problem['response'], entry_point=entry_point, instruction="")
            solutions.append(solution['response'])
        
        # Use ensemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=solutions, problem=cleaned_problem['response'])
        
        # Validate the best solution using test cases
        test_result = await self.test(problem=cleaned_problem['response'], solution=best_solution['response'], entry_point=entry_point)
        
        if test_result['result']:
            return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
        else:
            return "Solution failed test cases", self.llm.get_usage_summary()["total_cost"]
