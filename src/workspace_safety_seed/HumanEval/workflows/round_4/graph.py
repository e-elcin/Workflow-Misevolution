from typing import Literal
import workspace_safety_seed.HumanEval.workflows.template.operator as operator
import workspace_safety_seed.HumanEval.workflows.round_4.prompt as prompt_custom
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
        # Generate initial solution
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        
        # Perform self-consistency check
        solutions_list = [solution['response']] * 3  # Generate multiple copies for consistency check
        consistent_solution = await self.sc_ensemble(solutions=solutions_list, problem=problem)
        
        # Test the solution
        test_result = await self.test(problem=problem, solution=consistent_solution['response'], entry_point=entry_point)
        
        if test_result['result']:
            final_solution = test_result['solution']
        else:
            final_solution = consistent_solution['response']
        
        return final_solution, self.llm.get_usage_summary()["total_cost"]
