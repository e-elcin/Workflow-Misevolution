"""Context-local event tracing for the MATU trajectory experiment."""

from contextvars import ContextVar
from copy import deepcopy
from typing import Any

_CURRENT_TRACE = ContextVar("matu_workflow_trace", default=None)


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def begin_trace(*, round_number: int, prompt_id: str, run_index: int):
    trace = {
        "round": int(round_number),
        "prompt_id": str(prompt_id),
        "run": int(run_index),
        "events": [],
        "final_response": None,
        "error": None,
    }
    return _CURRENT_TRACE.set(trace)


def add_event(event_type: str, operator: str, **payload):
    trace = _CURRENT_TRACE.get()
    if trace is None:
        return

    trace["events"].append({
        "step": len(trace["events"]) + 1,
        "event_type": str(event_type),
        "operator": str(operator),
        **_json_safe(payload),
    })


def end_trace(token, *, final_response=None, error=None):
    trace = _CURRENT_TRACE.get()

    try:
        if trace is None:
            return None

        trace["final_response"] = _json_safe(final_response)
        trace["error"] = None if error is None else str(error)
        return deepcopy(trace)
    finally:
        _CURRENT_TRACE.reset(token)
