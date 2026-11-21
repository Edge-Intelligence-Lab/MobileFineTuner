# Local checkpoint control

Python 3.9+; standard library only. These tools do not modify model code, launch a
service, use a network, or provide Android/WorkManager lifecycle guarantees.

From `examples/checkpoint_resume/` (replace the example executable and training arguments):

```sh
"/absolute/path/to/trainer" --run_dir "/tmp/new run" TRAINARGS --checkpoint_every_updates 5 --suspend_after_updates 20 --slice_seconds 60
python3 tools/resume_control.py status "/tmp/new run"
python3 tools/resume_control.py suspend "/tmp/new run"
python3 tools/resume_control.py resume "/tmp/new run" --checkpoint_every_updates 5 --slice_seconds 60
python3 tools/resume_control.py report "/tmp/new run" > effective.json
python3 tools/resume_control.py report "/tmp/new run" --format csv > effective.csv
python3 tools/effective_metrics.py "/tmp/new run" --format json
python3 tools/resume_control.py --help
python3 tools/resume_control.py resume --help
python3 -B -m unittest discover -s tests -p 'test_resume_control.py' -v
```

Initial training uses `--run_dir NEW` plus normal training arguments. Resume uses
`--resume_state RUN` **without training arguments**: the C++ entrypoint restores
them. The controller requires `run_config.json.executable.path` to exist and be
executable; it never guesses from PATH or falls back to saved `arguments[0]`.
It uses saved `working_directory`, restores all five saved thread-environment
variables (JSON null unsets a variable), and inherits other environment entries.
Execution uses an argv list and `shell=False`; it waits and returns the child exit
code. The C++ entrypoint still validates the executable file stamp and assets.

Resume accepts `--checkpoint_every_updates N` (N >= 1),
`--suspend_after_updates K` (absolute update count, K >= 0), and `--slice_seconds S`
(finite S >= 0). Zero disables either stop control; omitted options use C++ defaults
(1, 0, 0). A slice is a cooperative budget, not a hard deadline.

For an unfinished run, `--resume_microbatch B` sets the maximum rows processed
at once during training and validation. For example, `resume RUN --resume_microbatch 1`
on an original batch of eight splits each logical batch without performing eight
optimizer updates. Choose any B from 1 through the original logical batch. The last
chunk of each logical batch may be shorter; larger caps are rejected. Without an explicit
override, the trainer uses the selected checkpoint's value.

`suspend` durably writes a temporary file inside RUN, then atomically replaces
`RUN/SUSPEND` and syncs the directory. This requests a clean stop after an optimizer
update; it is **not** an acknowledgement or a process kill. The trainer removes
the request on resume. `status` reports the persisted pointer, manifest progress,
and current state's committed `status` separately from `process_active` and
`runtime_status`. It probes the existing `run.lock` with nonblocking `flock`, never
creates it, and immediately releases a successful probe. A held lock means
`active`; an idle lock with committed `running` means `running_dead`. Idle
`suspended`/`completed` remain distinct. A missing/unreadable lock or unsupported
platform gives unknown liveness, not a claim that the trainer is active. The probe
is a point-in-time advisory check and does not identify the lock holder. It never infers
completion from checkpoint directory counts. Missing/invalid metadata yields JSON
diagnostics and exit 2. Tensor contents, optimizer compatibility, and executable
file stamps are validated by C++, not by status or the exporter.
The `trainer` object preserves committed progress, timing counters and `best_path`
from `state.json`; these are not reconstructed from uncommitted physical attempts.

## Metrics and lineage

Only `LATEST.current` and its manifest `previous` chain are authoritative. All
manifests are needed, but pruned ancestor `state.json`/tensor payloads are not.
The exporter computes the maximum committed `event_seq` per session, selects
updates at or below those cutoffs, and requires exactly one update for every
`completed_updates` in `1..LATEST.completed_updates`. Each ancestor prefix is also
checked. A malformed lineage, missing committed event, duplicate, or gap fails
closed: `valid=false`, no effective rows, diagnostics and exit 2. The physical
history is still exported. A complete valid report exits 0.

Fallback selection belongs to C++. Once its next checkpoint publishes a lineage
through the recorded predecessor, the corrupt abandoned branch is excluded. The
exporter never searches for a newer generation, auto-selects a fallback, or treats
a `checkpoint_committed` log event as pointer authority. Before a new commit it
reports the existing LATEST lineage, not a speculative resumed history. Status
`pointer_status=valid` means the pointer/manifest chain is readable, not that tensor
payloads can be restored (`payload_validated=false`).

JSON contains `effective_updates`, `effective_evaluations`, `physical_attempts` (all event records, including
non-update events and malformed raw lines), session cutoffs, lineage, diagnostics,
and effective timing totals. Update fields are preserved. Unmeasured `power_w`
and `energy_j` are null, never invented zeros. Physical classifications distinguish
effective updates, committed history, after-cutoff events, abandoned sessions,
and invalid history. After-cutoff means abandoned relative to this snapshot; an
active trainer may commit those events later. Invalid/unreadable logs are explicit
diagnostics; on-disk events are never rewritten or erased.

Evaluations use the same committed lineage cutoffs; evaluations from an interrupted
uncommitted attempt remain in the physical view, not the effective validation curve.
Duplicate committed evaluation update numbers fail closed. This does not certify
the evaluation procedure or turn a missing/nonfinite PPL into a valid measurement.

CSV includes a `summary` row, `effective_update`, `effective_evaluation`, and `physical_attempt` rows.
Filter `record_type` before aggregation to avoid counting the same effective update
again in the physical view. Missing measurements are empty CSV cells (JSON null in
`details_json`); this column preserves complete records and diagnostics. Readers
take one LATEST snapshot; concurrent new commits can make it stale, so rerun to
refresh. A truncated tail beyond the committed cutoff is retained as a warning.

## Offline Android import folder

```sh
python3 tools/prepare_android_task.py --family qwen \
  --model /absolute/model --train /absolute/train.jsonl --valid /absolute/valid.jsonl \
  --output /absolute/new-import-folder --steps 5 --seq 64 --rank 8 --pause-grace 300
```

Copy the prepared folder to shared emulator storage, then select it with the App's
Import picker. Inputs must already use this model's tokenizer. The tool validates
single-file layout, token ranges and supervision masks; it does not prove tokenizer
identity, model compatibility or workload admission. No network calls are made.
It never overwrites an existing output. Model weights are copied once into the
import folder and again by Android SAF into private storage, so budget both copies.
This explicit 120+300 second window was exercised with full Qwen seq64 on the
API35 emulator. Five steps are a continuity pilot, not an accuracy benchmark or
200-step acceptance. The helper's default grace remains 30 seconds; do not omit
the explicit override for this tested profile. Other models/workloads require
their own pilot. See the [Android guide](../ANDROID_CONTINUITY.md) for the bounded watchdog semantics.
