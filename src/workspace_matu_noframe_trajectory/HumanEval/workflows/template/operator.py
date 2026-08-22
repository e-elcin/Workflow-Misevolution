# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : operator demo of aflow
import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.HumanEval.workflows.template.operator_an import *
from workspace.HumanEval.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio

from scripts.utils.code import extract_test_cases_from_jsonl, test_case_2_test_function


from scripts.operators import Operator
from workspace_matu_noframe_trajectory.HumanEval.workflows.template.trace import add_event



class TracedOperator(Operator):
    """Record every underlying LLM call without changing its result."""

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        operation = getattr(op_class, "__name__", str(op_class))
        try:
            response = await super()._fill_node(
                op_class,
                prompt,
                mode=mode,
                **extra_kwargs,
            )
        except Exception as error:
            add_event(
                "llm_error",
                self.name,
                operation=operation,
                mode=mode,
                error=repr(error),
            )
            raise

        add_event(
            "llm_call",
            self.name,
            operation=operation,
            mode=mode,
            response=response,
        )
        return response


class Custom(TracedOperator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        add_event(
            "generation",
            self.name,
            instruction=instruction,
            response=response.get("response"),
            formatted_response=response,
        )
        return response
    
class CustomCodeGenerate(TracedOperator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        add_event(
            "generation",
            self.name,
            entry_point=entry_point,
            instruction=instruction,
            response=response.get("response"),
            formatted_response=response,
        )
        return response


class ScEnsemble(TracedOperator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()
        selected_index = answer_mapping[answer]
        selected_response = solutions[selected_index]

        add_event(
            "ensemble",
            self.name,
            candidate_count=len(solutions),
            unique_candidate_count=len(set(map(str, solutions))),
            candidates=solutions,
            selector_response=response,
            selected_letter=answer,
            selected_index=selected_index,
            selected_response=selected_response,
        )

        return {"response": selected_response}

class Test(TracedOperator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution, entry_point):

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="HumanEval")

        # RedCode prompts have no HumanEval test cases. Mark Test as skipped
        # instead of treating missing tests as a pass or iterating over None.
        if not test_cases:
            return {"no_test_cases": True}
                
        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 3
    ):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if isinstance(result, dict) and result.get("no_test_cases"):
                add_event(
                    "test",
                    self.name,
                    entry_point=entry_point,
                    status="skipped_no_test_cases",
                    result=True,
                )
                return {
                    "result": True,
                    "solution": solution,
                    "test_status": "skipped_no_test_cases",
                }
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]
        
        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}