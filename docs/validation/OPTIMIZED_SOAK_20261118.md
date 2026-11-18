# Combined-optimization long runs — 2026-11-18

The execution profile enables streamed attention, activation checkpointing and
FP32 frozen-weight offload together. This is a long run of that combined
profile, not an exhaustive long run of all individual switch subsets. Native
C++ CPU/Accelerate training uses seq32, batch/accumulation1, rank4, alpha8,
dropout0, LR2e-4, max gradient norm1 and four requested runtime threads.

## Completed Qwen queue

The new pinned executable (SHA-256 below) completes the queue at
**2026-11-18 03:50:01 CST**. Unlike the earlier default-path queue, this one
actually enables all three memory optimizations throughout calibration,
training, adapter resume and stress jobs.

| Model | Long + adapter-resume updates | Sampled long-run RSS peak MiB | Long-run median step seconds | Median RSS growth MiB | Late/early median step time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 300 + 30 | 2787.34 | 3.808 | -735.93 | 1.147 |
| Qwen3-0.6B | 300 + 30 | 2715.13 | 2.165 | -9.45 | 0.995 |
| Qwen2.5-0.5B | 300 + 30 | 2250.14 | 1.391 | -5.57 | 1.007 |

All logged losses, adapter gradients and parameters are finite. Clipped gradient
norms meet the configured tolerance; all B matrices are nonzero; every adapter
tensor changes on resumed training. The retained checkpoint300 matches final
export. No RSS-growth or slowdown trend flag is raised under the documented
runner thresholds. Offload payload reads/writes and loads/evictions are checked
per job, with 320/310/290 frozen entries respectively. Successful jobs remove
only their own temporary shards after validation and preserve evidence/adapters.

Qwen3.5 also completes five sequence128 stress updates from its resumed adapter,
with sampled peak RSS3140.55 MiB. This is a short larger-context stress check.
Adapter resume starts a fresh optimizer and does not restore moments or RNG.
These are stability checks, not downstream accuracy, convergence or universal
speed/memory improvements. Different storage/dropout settings and machine state
preclude treating the older default-path timings as a controlled benchmark.

Code/controller revision at launch: `9ee7201`. Pinned native executable SHA-256:
`46d1c478d215a20007103d80df8dbe6060a1d27c414db2e625062aae85e6117a`.
Raw local plan/status, CSVs, adapter checks, summaries and resource samples are
under `runs/optimized_soak_20261118/qwen/`. Future source changes do not rewrite
these results or the executable used by that queue.

## Gemma interrupted run and data-reader correction

The original Gemma270M combined run is stopped for performance investigation at
125/300 updates; 1B has not started in that queue. It is **not a completed pass**.
Logs and checkpoint100 are retained under `runs/optimized_soak_20261118/gemma/`.
The controller uses the generic stop-request result `user_stop_requested`; the
separate operator annotation records that the agent requested this stop to
investigate redundant tokenization. No user cancellation or numerical failure
is inferred from that token.

A two-second native sample finds 1680/1688 main-thread samples under Gemma
encoding inside `WikiText2Dataset::load_window_from_file`. Source inspection
confirms that every shuffled cache miss reopens the file and re-tokenizes its
entire preceding prefix. This sampling snapshot does not measure the fraction
of an entire run spent there. Read-only system logs also record repeated
clamshell/maintenance sleep; wall duration includes suspension. No OS sleep or
swap setting is modified.

The reader now records each line's file position and cumulative token offset
in its existing prescan. Window misses seek directly to the containing line,
then tokenize only the requested window. Cached token capacity, sample order,
token/EOS framing, labels and masks retain the established behavior. A small
line index is additional metadata; this is not a fixed cap on total process RSS.
Raw files and the supplied encoder must remain stable while a split is loaded.

The new regression compares IDs/labels/masks/order to independently materialized
expected tokens across shuffled/sequential access, EOS on/off, fraction0.5/1,
end windows and split reload. Token lengths deliberately differ from file byte
lengths. A counted encoder bounds work per random miss and catches the old
prefix scan. The strict BLAS-on Release suite passes all 35 entries with all 12 HF fixtures
supplied, executing 96 numerical profiles. The BLAS-off run has 34 passes and
one optional HF target skipped without its external fixtures.

A separate full-WikiText Gemma270M three-update comparison uses the old pinned
binary and indexed reader with identical model/training options. Every value in
all 252 exported adapter tensors is exactly equal (`max_abs=0`). The three-update
training interval is 63.324 seconds before and 14.068 seconds after indexing;
whole-process elapsed times are 177.365 and 129.133 seconds, including the initial
prescan. These are observations from this short comparison, not a universal
speedup. The initial prescan still tokenizes the split once. Its logs, commands,
binary hashes and comparison status remain in
`runs/optimized_soak_20261118/loader_comparison/`.

A fresh pinned Gemma270M/1B combined queue is prepared under
`runs/optimized_soak_20261118_gemma_indexed/gemma/`, with 300 updates per model,
calibration first, the same settings and resource guards. Its status/report must
show completion before a long-run pass is assigned. The old interrupted queue
is never resumed or relabeled as a completed 300-step run.

Physical-device training remains outside these host results.
