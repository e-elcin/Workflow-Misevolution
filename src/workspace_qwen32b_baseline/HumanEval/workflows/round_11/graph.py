from typing import Literal
import workspace_qwen32b_baseline.HumanEval.workflows.template.operator as operator
import workspace_qwen32b_baseline.HumanEval.workflows.round_11.prompt as prompt_custom
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
        self.validate = operator.Validate(self.llm)
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
        
        validated_solution = await self.validate(problem=problem, solution=selected_solution['response'], entry_point=entry_point)
        if validated_solution['valid']:
            test_result = await self.test(problem=problem, solution=validated_solution['response'], entry_point=entry_point)
        else:
            return selected_solution['response'], self.llm.get_usage_summary()["total_cost"]

        if test_result['result']:
            return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
        else:
            return test_result['solution'], self.llm.get_usage_summary()["total_cost"]
