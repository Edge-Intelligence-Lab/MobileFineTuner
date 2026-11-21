#!/usr/bin/env python3
"""Read-only, manifest-authoritative effective update exporter (stdlib only)."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys


class RunError(ValueError):
    """Invalid or unavailable run metadata."""


def _reject_constant(value):
    raise RunError("Non-finite JSON number: " + value)


def _parse_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise RunError("Non-finite JSON number")
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RunError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float,
                      object_pairs_hook=_object)


def read_object(path):
    try:
        value = parse_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise RunError("Cannot read {}: {}".format(path, error)) from error
    if not isinstance(value, dict):
        raise RunError("Expected JSON object: " + str(path))
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise RunError("Invalid " + name)
    return value


def component(value, name, allow_empty=False):
    if allow_empty and value == "":
        return value
    if (not isinstance(value, str) or not value or value in (".", "..")
            or any(char in value for char in ("/", "\\", "\0"))):
        raise RunError("Invalid " + name)
    return value


def schema_one(value):
    if type(value.get("schema")) is not int or value["schema"] != 1:
        raise RunError("Unsupported schema (expected 1)")


def read_lineage(run):
    """Never scan generations or require pruned ancestor payloads."""
    pointer = read_object(run / "LATEST")
    schema_one(pointer)
    current = component(pointer.get("current"), "LATEST.current")
    previous = component(pointer.get("previous"), "LATEST.previous", True)
    manifests, seen, cuts, last_cut = [], set(), {}, {}
    while current:
        if current in seen:
            raise RunError("Cycle in manifest lineage: " + current)
        seen.add(current)
        manifest = read_object(run / "training_state" / current / "manifest.json")
        schema_one(manifest)
        if manifest.get("generation") != current:
            raise RunError("Manifest generation mismatch: " + current)
        parent = component(manifest.get("previous"), "manifest.previous", True)
        session = component(manifest.get("session_id"), "manifest.session_id")
        seq = integer(manifest.get("event_seq"), "manifest.event_seq")
        step = integer(manifest.get("completed_updates"), "manifest.completed_updates")
        if not manifests and previous != parent:
            raise RunError("LATEST.previous disagrees with current manifest")
        if manifests and step > manifests[-1]["completed_updates"]:
            raise RunError("Progress decreases along committed lineage")
        if session in last_cut and seq > last_cut[session]:
            raise RunError("Event cutoff decreases along committed lineage")
        last_cut[session] = seq
        cuts[session] = max(cuts.get(session, 0), seq)
        manifests.append(manifest)
        current = parent
    return pointer, manifests, cuts


def _process_status(run):
    try:
        import fcntl
        with (run / "run.lock").open("rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"process_active": True, "liveness_error": None}
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return {"process_active": False, "liveness_error": None}
    except (ImportError, OSError) as error:
        return {"process_active": None, "liveness_error": str(error)}


def inspect_status(run):
    run = Path(run).absolute()
    result = {"run_dir": str(run), "pointer_status": "missing",
              "current": None, "previous": None, "completed_updates": None,
              "total_steps": None, "status": None, "state_status": "unavailable",
              "trainer": None,
              "suspend_requested": (run / "SUSPEND").exists(),
              "payload_validated": False, "errors": []}
    result.update(_process_status(run))
    result["runtime_status"] = "active" if result["process_active"] else "unknown"
    try:
        pointer, manifests, _ = read_lineage(run)
    except RunError as error:
        result["pointer_status"] = "invalid" if (run / "LATEST").exists() else "missing"
        result["errors"].append(str(error))
        return result
    current = manifests[0]
    result.update(pointer_status="valid", current=pointer["current"],
                  previous=pointer["previous"], completed_updates=current["completed_updates"])
    try:
        state = read_object(run / "training_state" / pointer["current"] / "state.json")
        schema_one(state)
        trainer = state.get("trainer")
        if not isinstance(trainer, dict):
            raise RunError("Invalid trainer state")
        step = integer(trainer.get("completed_updates"), "trainer.completed_updates")
        total = integer(trainer.get("total_steps"), "trainer.total_steps", 1)
        if step != current["completed_updates"] or step > total:
            raise RunError("State progress disagrees with manifest or budget")
        status = state.get("status")
        if status not in ("running", "suspended", "completed"):
            raise RunError("Invalid committed status")
        if status == "completed" and step != total:
            raise RunError("Completed state has unfinished updates")
        result.update(state_status="valid", total_steps=total, status=status, trainer=trainer)
    except RunError as error:
        result["state_status"] = "invalid"
        result["errors"].append(str(error))
    if result["process_active"] is False:
        result["runtime_status"] = ("running_dead" if result["status"] == "running"
                                    else result["status"] or "inactive")
    return result


def _finite_or_null(value):
    if type(value) in (float, int):
        try:
            if math.isfinite(value):
                return value
        except OverflowError:
            pass
    return None


def _metrics(event):
    result = dict(event)
    # Missing measurements are not zero energy/power measurements.
    for field in ("power_w", "energy_j"):
        result[field] = _finite_or_null(event.get(field))
    return result


def build_report(run):
    run = Path(run).absolute()
    errors, warnings, attempts, manifests, cuts = [], [], [], [], {}
    pointer = None
    try:
        pointer, manifests, cuts = read_lineage(run)
    except RunError as error:
        errors.append(str(error))

    committed = {session: [] for session in cuts}
    try:
        sessions = sorted((run / "sessions").iterdir())
    except OSError as error:
        sessions = []
        errors.append("Cannot list sessions: " + str(error))
    for session in sessions:
        if not session.is_dir():
            continue
        path = session / "events.jsonl"
        try:
            with path.open("rb") as stream:
                for number, raw in enumerate(stream, 1):
                    attempt = {"source": str(path.relative_to(run)), "line": number,
                               "session_id": session.name, "event": None,
                               "classification": "invalid", "error": None}
                    try:
                        if not raw.endswith(b"\n"):
                            raise RunError("Incomplete event line")
                        event = parse_json(raw.decode("utf-8"))
                        if not isinstance(event, dict):
                            raise RunError("Expected event object")
                        attempt["event"] = event
                        seq = integer(event.get("event_seq"), "event.event_seq", 1)
                        if event.get("session_id") != session.name:
                            raise RunError("Event session_id disagrees with directory")
                        if not isinstance(event.get("type"), str):
                            raise RunError("Missing event type")
                        if event["type"] == "update":
                            integer(event.get("completed_updates"), "update.completed_updates", 1)
                            attempt["event"] = _metrics(event)
                        if session.name not in cuts:
                            attempt["classification"] = "abandoned_session"
                        elif seq > cuts[session.name]:
                            attempt["classification"] = "after_cutoff"
                        else:
                            attempt["classification"] = "committed_history"
                            committed[session.name].append(attempt)
                    except (ValueError, UnicodeError) as error:
                        attempt["raw_line"] = raw.decode("utf-8", errors="replace")
                        attempt["error"] = str(error)
                        warnings.append("{}:{}: {}".format(path, number, error))
                    attempts.append(attempt)
        except OSError as error:
            message = "Cannot read {}: {}".format(path, error)
            (errors if session.name in cuts else warnings).append(message)

    candidates = []
    evaluations = []
    for session, cutoff in cuts.items():
        events = committed[session]
        # The writer starts at 1 and appends every event, not just updates.
        if len(events) != cutoff or any(
                item["event"]["event_seq"] != seq for seq, item in enumerate(events, 1)):
            errors.append("Missing, duplicate or out-of-order committed events: " + session)
        candidates.extend(item for item in events if item["event"]["type"] == "update")
        evaluations.extend(item for item in events if item["event"]["type"] == "evaluation")
    candidates.sort(key=lambda item: item["event"]["completed_updates"])
    target = manifests[0]["completed_updates"] if manifests else None
    if target is not None and (len(candidates) != target or any(
            item["event"]["completed_updates"] != step
            for step, item in enumerate(candidates, 1))):
        errors.append("Effective updates must be a unique contiguous prefix 1..{}".format(target))

    # Replay only newly committed records at each generation, in lineage order.
    # This validates every prefix without quadratic rescans of the whole history.
    prior_cuts = {}
    expected_step = 0
    for manifest in reversed(manifests):
        session = manifest["session_id"]
        previous_cut = prior_cuts.get(session, 0)
        new_cut = manifest["event_seq"]
        # Committed sequence validation above guarantees ordinal == event_seq.
        for item in committed[session][previous_cut:new_cut]:
            event = item["event"]
            if event["type"] == "update":
                expected_step += 1
                if event["completed_updates"] != expected_step:
                    errors.append("Out-of-order logical update at " + manifest["generation"])
            elif event["type"] == "evaluation":
                step = event.get("completed_updates")
                if type(step) is not int or not 1 <= step <= manifest["completed_updates"]:
                    errors.append("Evaluation exceeds committing manifest progress at " + manifest["generation"])
                elif step != expected_step + 1:
                    # Both supported native trainers emit evaluation before its update event.
                    errors.append("Evaluation is attached to the wrong update at " + manifest["generation"])
        prior_cuts[session] = new_cut
        if expected_step != manifest["completed_updates"]:
            errors.append("Invalid update prefix at " + manifest["generation"])
            break

    evaluations.sort(key=lambda item: item["event"].get("completed_updates")
                     if type(item["event"].get("completed_updates")) is int else -1)
    eval_steps = []
    for item in evaluations:
        event = item["event"]
        step = event.get("completed_updates")
        if type(step) is not int or step < 1 or target is None or step > target:
            errors.append("Invalid committed evaluation update")
        elif step in eval_steps:
            errors.append("Duplicate committed evaluation update: " + str(step))
        eval_steps.append(step)
    valid = not errors
    effective = [_metrics(item["event"]) for item in candidates] if valid else []
    effective_evaluations = [item["event"] for item in evaluations] if valid else []
    for item in candidates:
        item["classification"] = "effective" if valid else "invalid_effective_history"
    durations = [_finite_or_null(event.get("core_step_sec")) for event in effective]
    total_time = (sum(durations) if valid and all(
        value is not None and value >= 0 for value in durations) else None)
    total_time = _finite_or_null(total_time)
    return {"schema": 1, "run_dir": str(run), "valid": valid, "pointer": pointer,
            "committed_updates": target, "session_cutoffs": cuts,
            "lineage": [manifest["generation"] for manifest in manifests],
            "effective_updates": effective, "effective_evaluations": effective_evaluations,
            "physical_attempts": attempts,
            "summary": {"effective_update_count": len(effective) if valid else None,
                        "effective_evaluation_count": len(effective_evaluations) if valid else None,
                        "physical_update_count": sum(
                            isinstance(item["event"], dict) and item["event"].get("type") == "update"
                            for item in attempts),
                        "total_core_step_sec": total_time,
                        "mean_core_step_sec": total_time / len(effective)
                        if total_time is not None and effective else None},
            "errors": errors, "warnings": warnings}


def write_report(report, stream, output_format="json"):
    if output_format == "json":
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        return
    fields = ["record_type", "report_valid", "committed_updates", "classification",
              "session_id", "event_seq", "completed_updates", "type", "loss", "lr",
              "grad_norm", "total_tokens", "ema_loss", "core_step_sec", "rss_bytes",
              "process_peak_so_far_bytes", "power_w", "energy_j", "source", "line", "details_json"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    base = {"report_valid": report["valid"], "committed_updates": report["committed_updates"]}
    metadata = {key: value for key, value in report.items()
                if key not in ("effective_updates", "effective_evaluations", "physical_attempts")}
    writer.writerow(dict(base, record_type="summary", details_json=json.dumps(metadata, allow_nan=False)))
    for event in report["effective_updates"]:
        row = {key: event[key] for key in fields if key in event}
        row.update(base, record_type="effective_update", classification="effective",
                   details_json=json.dumps(event, allow_nan=False))
        writer.writerow(row)
    for event in report["effective_evaluations"]:
        row = {key: event[key] for key in fields if key in event}
        row.update(base, record_type="effective_evaluation", classification="committed_history",
                   details_json=json.dumps(event, allow_nan=False))
        writer.writerow(row)
    for attempt in report["physical_attempts"]:
        event = attempt["event"] or {}
        row = {key: event[key] for key in fields if key in event}
        row.update(base, record_type="physical_attempt", classification=attempt["classification"],
                   source=attempt["source"], line=attempt["line"],
                   details_json=json.dumps(attempt, allow_nan=False))
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", metavar="RUN", help="local training run directory")
    parser.add_argument("--format", choices=("json", "csv"), default="json",
                        help="stdout format (default: json); CSV includes both views")
    args = parser.parse_args(argv)
    report = build_report(args.run)
    write_report(report, sys.stdout, args.format)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
