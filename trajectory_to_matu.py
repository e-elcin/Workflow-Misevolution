#!/usr/bin/env python3
"""Convert traced workflow runs into final-only and full-trajectory MATU logs."""

import argparse
import json
from pathlib import Path


def atomic_json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def response_text(value):
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        direct = value.get("response")
        if isinstance(direct, str):
            return direct
        return json.dumps(value, sort_keys=True)

    return str(value)


def load_records(traces_dir, expected_prompts, expected_runs):
    root = traces_dir / "trajectories"

    if not root.is_dir():
        raise SystemExit(f"missing trajectory directory: {root}")

    prompt_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir()
    )

    if len(prompt_dirs) != expected_prompts:
        raise SystemExit(
            f"expected {expected_prompts} prompt directories, "
            f"found {len(prompt_dirs)}"
        )

    records = {}

    for prompt_dir in prompt_dirs:
        prompt_id = prompt_dir.name
        prompt_records = []

        for run_index in range(expected_runs):
            path = prompt_dir / f"run_{run_index}.json"

            if not path.exists():
                raise SystemExit(f"missing trajectory: {path}")

            record = json.loads(path.read_text())

            if record.get("error"):
                raise SystemExit(
                    f"trajectory error in {path}: {record['error']}"
                )

            if record.get("validation_errors"):
                raise SystemExit(
                    f"trace validation failed in {path}: "
                    f"{record['validation_errors']}"
                )

            if record.get("run") != run_index:
                raise SystemExit(
                    f"run-index mismatch in {path}: "
                    f"{record.get('run')} != {run_index}"
                )

            if record.get("prompt_id") != prompt_id:
                raise SystemExit(
                    f"prompt-ID mismatch in {path}"
                )

            if not isinstance(record.get("request"), str):
                raise SystemExit(
                    f"missing request text in {path}"
                )

            if not isinstance(record.get("final_response"), str):
                raise SystemExit(
                    f"missing final response in {path}"
                )

            prompt_records.append((path, record))

        records[prompt_id] = prompt_records

    return records


def build_logs(records):
    final_log = {}
    trajectory_log = {}
    metadata = {}

    llm_call_counts = []
    event_counts = []

    for prompt_id, prompt_records in records.items():
        final_runs = []
        trajectory_runs = []
        metadata_runs = []

        for path, record in prompt_records:
            request = record["request"]
            final_response = record["final_response"]
            events = record.get("events", [])

            llm_events = [
                event
                for event in events
                if event.get("event_type") == "llm_call"
            ]

            if not llm_events:
                raise SystemExit(
                    f"no LLM calls recorded in {path}"
                )

            final_runs.append([
                {
                    "role": "user",
                    "output": request,
                },
                {
                    "role": "assistant",
                    "output": final_response,
                },
            ])

            full_run = [
                {
                    "role": "user",
                    "output": request,
                }
            ]

            llm_metadata = []

            for llm_index, event in enumerate(llm_events, 1):
                output = response_text(event.get("response"))

                if not output:
                    raise SystemExit(
                        f"empty LLM output in {path}, "
                        f"event step {event.get('step')}"
                    )

                full_run.append({
                    "role": "assistant",
                    "output": output,
                })

                llm_metadata.append({
                    "trajectory_turn": llm_index,
                    "trace_step": event.get("step"),
                    "operator": event.get("operator"),
                    "operation": event.get("operation"),
                    "mode": event.get("mode"),
                })

            semantic_events = []

            for event in events:
                event_type = event.get("event_type")

                if event_type == "generation":
                    semantic_events.append({
                        "trace_step": event.get("step"),
                        "event_type": event_type,
                        "operator": event.get("operator"),
                        "entry_point": event.get("entry_point"),
                        "instruction_empty": (
                            event.get("instruction") == ""
                        ),
                    })

                elif event_type == "ensemble":
                    semantic_events.append({
                        "trace_step": event.get("step"),
                        "event_type": event_type,
                        "operator": event.get("operator"),
                        "candidate_count": event.get(
                            "candidate_count"
                        ),
                        "unique_candidate_count": event.get(
                            "unique_candidate_count"
                        ),
                        "selected_index": event.get(
                            "selected_index"
                        ),
                        "selected_letter": event.get(
                            "selected_letter"
                        ),
                    })

                elif event_type == "test":
                    semantic_events.append({
                        "trace_step": event.get("step"),
                        "event_type": event_type,
                        "operator": event.get("operator"),
                        "status": event.get("status"),
                        "result": event.get("result"),
                    })

            trajectory_runs.append(full_run)
            metadata_runs.append({
                "run": record["run"],
                "round": record["round"],
                "trace_file": str(path),
                "llm_call_count": len(llm_events),
                "event_count": len(events),
                "llm_calls": llm_metadata,
                "semantic_events": semantic_events,
                "final_response": final_response,
            })

            llm_call_counts.append(len(llm_events))
            event_counts.append(len(events))

        final_log[prompt_id] = final_runs
        trajectory_log[prompt_id] = trajectory_runs
        metadata[prompt_id] = metadata_runs

    return (
        final_log,
        trajectory_log,
        metadata,
        llm_call_counts,
        event_counts,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--expected-prompts",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    records = load_records(
        traces_dir,
        args.expected_prompts,
        args.expected_runs,
    )

    (
        final_log,
        trajectory_log,
        metadata,
        llm_call_counts,
        event_counts,
    ) = build_logs(records)

    run_counts = {
        len(runs)
        for runs in final_log.values()
    }
    if run_counts != {args.expected_runs}:
        raise SystemExit(
            f"unexpected run counts: {sorted(run_counts)}"
        )

    final_path = out_dir / "final_log.json"
    trajectory_path = out_dir / "trajectory_log.json"
    metadata_path = out_dir / "trajectory_metadata.json"

    atomic_json_write(final_path, final_log)
    atomic_json_write(trajectory_path, trajectory_log)
    atomic_json_write(metadata_path, metadata)

    # The validated HarmBench grader expects these conventional names.
    atomic_json_write(out_dir / "log.json", final_log)
    atomic_json_write(
        out_dir / "conversation_logs_harmbench.json",
        final_log,
    )

    summary = {
        "traces_dir": str(traces_dir),
        "prompts": len(final_log),
        "runs_per_prompt": sorted(run_counts),
        "total_trajectories": sum(
            len(runs) for runs in final_log.values()
        ),
        "llm_calls_per_trajectory": sorted(
            set(llm_call_counts)
        ),
        "events_per_trajectory": sorted(
            set(event_counts)
        ),
        "final_log": str(final_path),
        "trajectory_log": str(trajectory_path),
        "metadata": str(metadata_path),
    }

    atomic_json_write(
        out_dir / "adapter_summary.json",
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
