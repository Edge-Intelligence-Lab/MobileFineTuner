#!/usr/bin/env python3
"""Local checkpoint controller. No daemon, network access, or shell execution."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__:
    from .effective_metrics import RunError, build_report, inspect_status, read_object, schema_one, write_report
else:
    from effective_metrics import RunError, build_report, inspect_status, read_object, schema_one, write_report


THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                      "MKL_NUM_THREADS", "MFT_NUM_THREADS")


def request_suspend(run):
    """Durably publish a request; only the trainer can acknowledge suspension."""
    run = Path(run).resolve()
    schema_one(read_object(run / "run_config.json"))
    fd, name = tempfile.mkstemp(prefix=".SUSPEND.", suffix=".tmp", dir=str(run))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write('{"schema":1,"request":"suspend"}\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, run / "SUSPEND")
        directory_fd = os.open(str(run), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {"run_dir": str(run), "suspend_requested": True, "acknowledged": False}


def _control_value(name, value):
    if name == "slice_seconds":
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RunError("slice_seconds must be finite and >= 0")
    elif name == "resume_microbatch":
        if type(value) is not int or not 1 <= value <= 2147483647:
            raise RunError("resume_microbatch must be a positive 32-bit integer")
    elif type(value) is not int or value < (1 if name == "checkpoint_every_updates" else 0):
        raise RunError(name + " has an invalid update count")
    return str(value)


def resume_run(run, checkpoint_every_updates=None, suspend_after_updates=None, slice_seconds=None, resume_microbatch=None):
    run = Path(run).resolve()
    config = read_object(run / "run_config.json")
    schema_one(config)
    saved = config.get("executable")
    if not isinstance(saved, dict) or not isinstance(saved.get("path"), str) or not saved["path"]:
        raise RunError("Missing saved executable; refusing PATH or arguments[0] fallback")
    executable = Path(saved["path"])
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise RunError("Saved executable is missing or not executable: " + str(executable))
    directory = config.get("working_directory")
    if not isinstance(directory, str) or not Path(directory).is_absolute() or not Path(directory).is_dir():
        raise RunError("Saved working_directory is missing or invalid")
    saved_env = config.get("thread_environment")
    if not isinstance(saved_env, dict) or set(saved_env) != set(THREAD_ENVIRONMENT):
        raise RunError("Missing or invalid saved thread_environment")
    environment = os.environ.copy()
    for key, value in saved_env.items():
        if value is None:
            environment.pop(key, None)
        elif isinstance(value, str) and "\0" not in value:
            environment[key] = value
        else:
            raise RunError("Invalid saved environment entry: " + key)
    if not (run / "LATEST").is_file():
        raise RunError("No committed LATEST to resume")
    arguments = [str(executable), "--resume_state", str(run)]
    for name, value in (("checkpoint_every_updates", checkpoint_every_updates),
                        ("suspend_after_updates", suspend_after_updates),
                        ("slice_seconds", slice_seconds), ("resume_microbatch", resume_microbatch)):
        if value is not None:
            arguments.extend(("--" + name, _control_value(name, value)))
    # Training arguments and executable file stamps are restored/checked by C++.
    return subprocess.run(arguments, cwd=directory, env=environment, shell=False, check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="read LATEST progress and probe existing run.lock with flock")
    status.add_argument("run", metavar="RUN")
    suspend = commands.add_parser("suspend", help="atomically request a clean stop at an update boundary")
    suspend.add_argument("run", metavar="RUN")
    resume = commands.add_parser("resume", help="run the saved executable in the foreground, without trainargs")
    resume.add_argument("run", metavar="RUN")
    resume.add_argument("--checkpoint_every_updates", type=int, metavar="N", help="checkpoint interval >= 1 (C++ default: 1)")
    resume.add_argument("--suspend_after_updates", type=int, metavar="K", help="absolute completed-update stop >= 0; 0 disables")
    resume.add_argument("--slice_seconds", type=float, metavar="S", help="session time budget >= 0; 0 disables")
    resume.add_argument("--resume_microbatch", type=int, metavar="B",
                        help="maximum rows per forward/backward (1..original --batch); default: last committed cap")
    report = commands.add_parser("report", help="export effective metrics and retained physical attempts to stdout")
    report.add_argument("run", metavar="RUN")
    report.add_argument("--format", choices=("json", "csv"), default="json", help="output format (default: json)")
    args = parser.parse_args(argv)
    try:
        if args.command == "resume":
            code = resume_run(args.run, args.checkpoint_every_updates, args.suspend_after_updates,
                              args.slice_seconds, args.resume_microbatch)
            return code if code >= 0 else 128 - code
        if args.command == "report":
            result = build_report(args.run)
            write_report(result, sys.stdout, args.format)
            return 0 if result["valid"] else 2
        result = inspect_status(args.run) if args.command == "status" else request_suspend(args.run)
        json.dump(result, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
        return 2 if result.get("errors") else 0
    except (OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
