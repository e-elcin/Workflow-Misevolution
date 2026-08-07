from typing import Literal
import workspace_qwen32b_baseline.HumanEval.workflows.template.operator as operator
import workspace_qwen32b_baseline.HumanEval.workflows.round_8.prompt as prompt_custom
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
        self.refine_solution = operator.RefineSolution(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        solutions_list = [solution['response']]
        
        # Generate additional potential solutions and use ScEnsemble to select the best one
        for _ in range(4):  # Generate 4 additional solutions
            additional_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions_list.append(additional_solution['response'])
        
        selected_solution = await self.sc_ensemble(solutions=solutions_list, problem=problem)
        
        refined_solution = await self.refine_solution(solution=selected_solution['response'], problem=problem)
        
        test_result = await self.test(problem=problem, solution=refined_solution['response'], entry_point=entry_point)
        if test_result['result']:
            return refined_solution['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
