# MobileFineTuner Checkpoint Resume

An opt-in C++ training entry point for Qwen/Gemma LoRA and a standalone Android
controller. Existing trainers, Android SDK APIs, and adapter warm-start commands
are unchanged. Run this guide's commands from `examples/checkpoint_resume/`.

The Android app offers **Pause** (finish the update), **Release resources now**
(exit and replay from the last committed checkpoint), and **Focus for one hour**
(block training across all tasks until the user starts it again after focus).
Memory pressure requests a safe pause; optional CPU contention detection monitors
the trainer, not the names of other apps. See [Android controls](ANDROID_CONTINUITY.md#choosing-how-to-stop).
See [user-controls verification and reproduction](USER_CONTROLS_VALIDATION.md) for this candidate's results and limitations.

Pretrained weights and datasets are supplied by the user and remain subject to
their own terms. For a local demo without downloading either, use
`python3 tools/make_android_fixture.py` to generate a tiny random fixture.

## What is saved

At a completed optimizer-update boundary: all trainable LoRA tensors, named Adam/
AdamW state (including AMSGrad), exact data permutation/cursor and shuffle RNG,
global update, original scheduler budget/configuration, EMA, token counters,
immutable best adapter reference, committed timing counters and session events.
Frozen backbone files are referenced, not copied into every checkpoint. Dropout
must be zero. Initial random LoRA values are saved; dataset seed alone does not
make two independently initialized runs identical.

`--resume_from` retains its old adapter warm-start meaning. Exact continuation
uses `--resume_state RUN`, with no training arguments. This reloads the original
configuration, not a shortened schedule or a fresh optimizer.

**Less memory after suspension:** on an unfinished task, choose **Resume physical
batch** in the Android app, or pass `--resume_microbatch` on the native resume
command. For an original `--batch 8 --grad_accum 1`, the executions below
all consume the same logical training batch per Adam update:

| Physical batch | Chunks per full update | Examples per full update |
| ---: | ---: | ---: |
| 8 | 1 | 8 |
| 5 | 2 (5+3) | 8 |
| 4 | 2 | 8 |
| 3 | 3 (3+3+2) | 8 |
| 1 | 8 | 8 |

If the original accumulation was 2, the effective batch stays 16 instead.
Each logical batch is split independently: cap 5 with accumulation 2 plans four
chunks (5+3, 5+3), not a single stream of 16. Resume accepts caps from 1 through
the original logical batch inclusive; larger values are rejected. Omit the option to
keep the last committed value. Sampling, target-token weighting, validation
coverage and the learning-rate schedule remain tied to the original run.
FP32 summation order changes, so results are not guaranteed bit-identical when
resizing. This changes activation size, not backbone/optimizer memory; the Android
memory guard still blocks a restart under pressure. Start a new run with this
build; a checkpoint pinned to an older executable is not migrated.

## Build

```sh
cmake -S . -B build-final -DCMAKE_BUILD_TYPE=Release -DUSE_BLAS=ON -DBUILD_TESTS=ON
cmake --build build-final -j 4
ctest --test-dir build-final --output-on-failure
python3 tests/test_resume_process.py --binary build-final/operators/test_resume_process
python3 -m unittest discover -s tests -p 'test_resume_control.py' -v
```

For a host check that actually trains, pauses, and resumes with a non-divisor
physical batch, run the following from the same directory. Use a fresh output
directory each time; the generated fixture is random test data, not a benchmark.

```sh
python3 tools/make_android_fixture.py
python3 tests/microbatch_admission.py build-final/mft_qwen_resume
python3 tests/test_elastic_microbatch.py --ragged --binary build-final/mft_qwen_resume \
  --model android/app/src/main/assets/tiny_qwen/model --output out/ragged-check
```

The isolated build provides `mft_qwen_resume` and `mft_gemma_resume`.
Use pre-tokenized JSONL (`ids` and matching `mask` arrays) or the existing
pretokenized-stream format. Raw-text tokenization needs the existing optional
native tokenizer build; no model or dataset is downloaded by these commands.

## Start, suspend, continue

```sh
./build-final/mft_qwen_resume \
  --model_dir /absolute/qwen-model \
  --jsonl_train /absolute/train.jsonl --jsonl_valid /absolute/valid.jsonl \
  --steps 200 --batch 8 --seq_len 64 --lora_r 8 --lora_dropout 0 \
  --lr 0.0002 --lr_scheduler cosine --warmup_steps 10 --shuffle \
  --run_dir /absolute/new-run --checkpoint_every_updates 5

python3 tools/resume_control.py suspend /absolute/new-run
python3 tools/resume_control.py status /absolute/new-run
python3 tools/resume_control.py resume /absolute/new-run
python3 tools/resume_control.py report /absolute/new-run --format csv > /absolute/report.csv
```

Alternatively, after a confirmed suspension, use
`python3 tools/resume_control.py resume /absolute/new-run --resume_microbatch 4`,
or directly `./build-final/mft_qwen_resume --resume_state /absolute/new-run --resume_microbatch 4`.
Use `mft_gemma_resume` for a Gemma session.

Run the suspend command in a second terminal. A request is not an acknowledgement:
wait for `SUSPENDED completed_updates=K` and process exit. SIGINT/SIGTERM also request
the next safe boundary; SIGSTOP is not used. Accumulated microbatches finish as one
update before suspension, so a large update can delay the response.

For finite execution slices add `--slice_seconds 60`; a later `resume` can use the
same option. This is cooperative, not a hard 60-second deadline. Model loading is
included in the slice clock; a very small slice may suspend at update zero.
`--suspend_after_updates 73` requests suspension at global update 73 while keeping
the original total budget. Zero disables that control. Reopening a completed run
does not execute additional updates.

## Recovery and storage

- A run lock prevents two writers. Each process gets a separate session directory.
- A generation is durable before atomic `LATEST` publication. Only current and
  the recorded predecessor are recovery candidates; directory order is never used.
- Retain two full-state payloads and all small lineage manifests/session logs.
  Failed/orphan payloads and disposable shard caches are reclaimed while locked.
- A corrupted current payload can fall back only to the recorded predecessor.
  Missing/invalid LATEST, changed model/data assets or a different executable fail.
- `SAVE_FAILED` is a nonzero failure, not a paused acknowledgement. This CLI does
  not wait indefinitely for disk repair. Fix storage, then explicitly resume the
  last committed state; uncommitted work may need replay. Post-publication I/O
  ambiguity requires restarting, not blindly retrying the same session.
- SIGKILL/power loss can lose work since the last commit. Tests cover process death
  and injected I/O failure, not physical power-cut durability on every filesystem.
- File size/mtime/path stamps detect ordinary asset replacement, not arbitrary
  same-size bit corruption or malicious edits. Keep input assets immutable.

Regular user-requested adapter exports and immutable best files remain available;
the two-generation policy applies to full resume states, not all exported models.

## Boundaries

Native CPU FP32 frozen-backbone LoRA only, same executable/assets/thread settings.
Qwen Q/V or full targets; Gemma Q/V. Resume is wired into the two new executables,
not every legacy trainer, SDK, or Flower client/server. The standalone Android
app is documented in [ANDROID_CONTINUITY.md](ANDROID_CONTINUITY.md). MMLU's dataset
state API exists and is unit tested, but the separate MMLU training CLI is not
automatically converted to exact resume.

Nonzero dropout, FP16 disk shards, Full-FT, GPU, multiple slices per adapter,
mid-microbatch saves, and changed training parameters are not accepted in this
session path. Android foreground service and bounded WorkManager slices use the
same native state protocol; there is no permission bypass or never-killed guarantee.
Physical-device acceptance is separate from emulator tests. Full-model cold loading can still require
the original backbone memory; small resume files do not solve base-model OOM.

Unset `QWEN_DEBUG` and `DISABLE_QK_NORM`: session mode rejects these computation
overrides even if their environment value is `0`. Do not rebuild or replace the
executable while a run is suspended; continuation requires its original build.

Metrics distinguish committed effective updates from physical attempts including
replayed work. Unsampled power/energy is null, not zero. See [tools](tools/README.md).
The native event exporter is not an automatic integration with the separate
EdgeFlowerTune submission logger or leaderboard.

Checkpoint save, state restoration and full native readiness now have distinct
timing, logical checkpoint I/O, sampled RSS and OS disk-counter fields. Export:
`python3 tools/phase_metrics.py /absolute/run --output /absolute/phases.csv`.
Sampled phase RSS is not an exact phase high-water mark; process HWM includes
earlier model loading. OS disk I/O can be zero due to cache and is not the logical
checkpoint byte count. See the Android guide for measurement boundaries.
The exporter rejects an inconsistent RSS endpoint/peak record by default. For
historical diagnostics only, `--include-invalid-rss` retains the original values
and adds explicit warning columns; it does not repair or silently discard data.

## Tests and evidence

Tests include optimizer/sampler state, transactional failures, new OS processes,
10=4+6, 200=73+127, crash replay, immutable best references, and actual local
pretrained-model runs. The acceptance report records which model/optimization
combinations really ran. Test scripts in `tests/` preserve raw logs and failed
attempts; none of these short checks are task-accuracy benchmark results.

The JSON parser is vendored nlohmann/json 3.12.0 under its MIT license in
`operators/third_party/nlohmann/`. No network or secret configuration is required
at training time.
