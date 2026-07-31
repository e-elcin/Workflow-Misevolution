from typing import Literal
import workspace.HumanEval_ag.workflows.template.operator as operator
import workspace.HumanEval_ag.workflows.round_1.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Seed workflow (AnswerGenerate variant, closer to the paper's Answer Generator baseline).
        Uses ANSWER_GENERATION_PROMPT ("think step by step, answer the question") without
        the code-fill framing that CustomCodeGenerate applies.
        """
        response = await self.answer_generate(input=problem)
        return response['answer'], self.llm.get_usage_summary()["total_cost"]
