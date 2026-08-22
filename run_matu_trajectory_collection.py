#!/usr/bin/env python3
"""Collect complete workflow trajectories for the no-framing MATU pilot."""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "RedCode" / "evaluation"))
os.chdir(SRC)

from scripts.async_llm import LLMsConfig
from scripts.optimizer_utils.graph_utils import GraphUtils
from RedCode_Gen.prompts import CODE_GENERATION_PROMPT
from RedCode_Gen.utils import (
    extract_entry_point_from_signature,
    read_prompt,
)
from workspace_matu_noframe_trajectory.HumanEval.workflows.template.trace import (
    begin_trace,
    end_trace,
)

VALID_ROUNDS = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11)

EXPECTED = {
    1:  {"generation": 1, "ensemble": 0, "test": 0},
    3:  {"generation": 3, "ensemble": 1, "test": 0},
    4:  {"generation": 1, "ensemble": 0, "test": 1},
    5:  {"generation": 1, "ensemble": 0, "test": 1},
    6:  {"generation": 1, "ensemble": 0, "test": 1},
    7:  {"generation": 1, "ensemble": 1, "test": 1},
    8:  {"generation": 3, "ensemble": 1, "test": 1},
    9:  {"generation": 1, "ensemble": 0, "test": 1},
    10: {"generation": 1, "ensemble": 0, "test": 1},
    11: {"generation": 3, "ensemble": 1, "test": 1},
}

DATASET = SRC / "RedCode" / "dataset" / "RedCode-Gen"
WORKSPACE = "workspace_matu_noframe_trajectory/HumanEval"


def atomic_json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prompt_id(item):
    return f"{item['family']}__{item['file'].removesuffix('.py')}"


def trajectory_path(out_dir, pid, run_index):
    return out_dir / "trajectories" / pid / f"run_{run_index}.json"


def validate_trace(trace, round_number):
    errors = []
    counts = Counter(
        event["event_type"]
        for event in trace.get("events", [])
    )

    for event_type, expected_count in EXPECTED[round_number].items():
        actual = counts.get(event_type, 0)
        if actual != expected_count:
            errors.append(
                f"{event_type}: expected {expected_count}, found {actual}"
            )

    generation_events = [
        event for event in trace.get("events", [])
        if event["event_type"] == "generation"
    ]

    for index, event in enumerate(generation_events, 1):
        if event.get("instruction") != "":
            errors.append(
                f"generation {index} used non-empty instruction"
            )
        if not isinstance(event.get("response"), str):
            errors.append(
                f"generation {index} has no string response"
            )

    test_events = [
        event for event in trace.get("events", [])
        if event["event_type"] == "test"
    ]

    for event in test_events:
        if event.get("status") != "skipped_no_test_cases":
            errors.append(
                f"unexpected Test status: {event.get('status')!r}"
            )

    if round_number == 7:
        ensemble_events = [
            event for event in trace.get("events", [])
            if event["event_type"] == "ensemble"
        ]
        if ensemble_events:
            event = ensemble_events[0]
            if event.get("candidate_count") != 5:
                errors.append("round 7 did not pass five candidates")
            if event.get("unique_candidate_count") != 1:
                errors.append(
                    "round 7 candidates were expected to be identical"
                )

    return errors


def load_models(config_path):
    import yaml

    config = yaml.safe_load(config_path.read_text())
    if "models" in config:
        config = config["models"]
    return LLMsConfig(config)


def usable_existing_record(path):
    if not path.exists():
        return None

    try:
        record = json.loads(path.read_text())
    except Exception:
        return None

    if record.get("error") is not None:
        return None
    if record.get("validation_errors"):
        return None
    if not isinstance(record.get("final_response"), str):
        return None

    return record


async def collect(args):
    subset_path = Path(args.subset)
    if not subset_path.is_absolute():
        subset_path = ROOT / subset_path

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    subset = json.loads(subset_path.read_text())
    if len(subset) != args.expected_prompts:
        raise SystemExit(
            f"expected {args.expected_prompts} prompts, "
            f"found {len(subset)} in {subset_path}"
        )

    models = load_models(config_path)
    executor = models.get("executor")

    print("round:", args.round)
    print("prompts:", len(subset))
    print("runs per prompt:", args.n_runs)
    print("temperature:", executor.temperature)
    print("workspace:", WORKSPACE)
    print("output:", out_dir)

    workflow_class = GraphUtils(WORKSPACE).load_graph(
        args.round,
        f"{WORKSPACE}/workflows",
    )

    completed = 0
    skipped = 0
    failed = 0

    for prompt_number, item in enumerate(subset, 1):
        pid = prompt_id(item)
        source_path = DATASET / item["family"] / item["file"]
        signature = read_prompt(str(source_path))
        problem = CODE_GENERATION_PROMPT + signature
        entry_point = extract_entry_point_from_signature(signature)

        for run_index in range(args.n_runs):
            output_path = trajectory_path(
                out_dir,
                pid,
                run_index,
            )

            if not args.overwrite:
                existing = usable_existing_record(output_path)
                if existing is not None:
                    skipped += 1
                    print(
                        f"[{prompt_number}/{len(subset)}] "
                        f"{pid} run {run_index + 1}/{args.n_runs} "
                        "RESUME-SKIP"
                    )
                    continue

            workflow = workflow_class(
                name=f"round_{args.round}_{pid}_run_{run_index}",
                llm_config=executor,
                dataset="HumanEval",
            )

            token = begin_trace(
                round_number=args.round,
                prompt_id=pid,
                run_index=run_index,
            )

            response = None
            error = None

            try:
                result = await workflow(
                    problem=problem,
                    entry_point=entry_point,
                )
                response = (
                    result[0]
                    if isinstance(result, tuple)
                    else result
                )
                if not isinstance(response, str):
                    raise TypeError(
                        f"workflow returned {type(response).__name__}, "
                        "expected str"
                    )
            except Exception as exception:
                error = (
                    f"{type(exception).__name__}: {exception}"
                )

            trace = end_trace(
                token,
                final_response=response,
                error=error,
            )

            trace["request"] = problem
            trace["entry_point"] = entry_point
            trace["family"] = item["family"]
            trace["file"] = item["file"]
            trace["workspace"] = WORKSPACE
            trace["temperature"] = executor.temperature
            trace["top_p"] = executor.top_p
            trace["trace_version"] = 1
            trace["validation_errors"] = (
                [] if error else validate_trace(trace, args.round)
            )

            atomic_json_write(output_path, trace)

            if error or trace["validation_errors"]:
                failed += 1
                state = "FAILED"
            else:
                completed += 1
                state = "OK"

            print(
                f"[{prompt_number}/{len(subset)}] "
                f"{pid} run {run_index + 1}/{args.n_runs} "
                f"{state} events={len(trace['events'])}"
            )

            if state == "FAILED":
                print("  error:", error)
                print(
                    "  validation:",
                    trace["validation_errors"],
                )
                if args.abort_on_error:
                    raise SystemExit(
                        "aborting after first invalid trajectory"
                    )

    run_files = []

    for run_index in range(args.n_runs):
        run_data = {}

        for item in subset:
            pid = prompt_id(item)
            path = trajectory_path(out_dir, pid, run_index)

            if not path.exists():
                continue

            record = json.loads(path.read_text())
            final_response = record.get("final_response")

            if record.get("error"):
                final_response = (
                    f"[RUN ERROR: {record['error']}]"
                )
            elif record.get("validation_errors"):
                final_response = (
                    "[RUN ERROR: invalid trajectory trace]"
                )

            run_data[pid] = {
                "request": record["request"],
                "response": final_response,
                "trajectory_file": str(
                    path.relative_to(out_dir)
                ),
            }

        run_path = out_dir / f"run_{run_index}.json"
        atomic_json_write(run_path, run_data)
        run_files.append(str(run_path))

    expected_total = len(subset) * args.n_runs
    valid_total = 0
    error_total = 0
    missing_total = 0

    for item in subset:
        pid = prompt_id(item)

        for run_index in range(args.n_runs):
            path = trajectory_path(out_dir, pid, run_index)

            if not path.exists():
                missing_total += 1
                continue

            record = json.loads(path.read_text())
            if (
                record.get("error")
                or record.get("validation_errors")
            ):
                error_total += 1
            else:
                valid_total += 1

    summary = {
        "round": args.round,
        "workspace": WORKSPACE,
        "subset": str(subset_path),
        "n_prompts": len(subset),
        "n_runs": args.n_runs,
        "expected_trajectories": expected_total,
        "valid_trajectories": valid_total,
        "error_trajectories": error_total,
        "missing_trajectories": missing_total,
        "completed_this_invocation": completed,
        "resume_skipped": skipped,
        "run_files": run_files,
    }

    atomic_json_write(out_dir / "collection_summary.json", summary)

    print("\nCOLLECTION SUMMARY")
    print(json.dumps(summary, indent=2))

    if error_total or missing_total:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        choices=VALID_ROUNDS,
    )
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--subset",
        default="results/redcode_subset.json",
    )
    parser.add_argument(
        "--expected-prompts",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--config",
        default="src/config/config_matu.yaml",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun even valid existing trajectories",
    )
    parser.add_argument(
        "--abort-on-error",
        action="store_true",
    )
    args = parser.parse_args()

    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
