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

## Completed indexed-reader Gemma long runs

The fresh pinned queue at source revision `3ffe407` completes at
**2026-11-18 15:58:32 CST**. Both models pass their three-step calibration and
300-step combined-optimization training on full WikiText with the same options
and resource guards. Raw evidence is under
`runs/optimized_soak_20261118_gemma_indexed/gemma/`.

| Model | Completed optimizer updates | Sampled peak RSS MiB | Median step seconds | Median RSS growth MiB | Late/early median step time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma3 270M | 300 | 3229.19 | 4.396 | +0.60 | 1.005 |
| Gemma3 1B-PT | 300 | 5780.56 | 7.559 | -401.26 | 1.135 |

All 300 logged losses per model are finite. All 252/364 final adapter tensors
are finite and change between checkpoint100 and checkpoint300; every B matrix
is nonzero, and the final export equals checkpoint300 exactly. Native training
uses the shared finite-gradient/clipping checks. The external runner does not
export a per-step Gemma gradient CSV, so none is claimed here. Neither run raises
the documented sustained RSS-growth or slowdown flag.

Both runs demonstrate positive offload payload I/O: 270M reports
643,434,854,400 bytes read, 1,743,481,344 bytes written, 141,300 loads and 141,303
evictions; 1B reports 2,399,724,902,400 bytes read, 5,207,503,360 bytes written,
203,700 loads and 203,846 evictions. These are logical tensor-payload counts;
OS file caching means they are not measurements of physical SSD traffic.
Temporary per-job shards are removed only after successful validation; all
logs, resource samples, summaries and adapter snapshots remain.

Pinned SHA-256:

- 270M: `1e03a112eb4489da74c842126891286416140adbba169db3d2caf028cd6ecbad`
- 1B: `0b79816f37e232aac3f9a1d2f5ddbdf338377a1480c37bb068f6cc76d64cd58e`

The completion audit re-reads all four native logs (calibration/long per model),
step counts and finite losses, every saved final adapter tensor, checkpoint100
changes and checkpoint300 equality, payload I/O and binary hashes. It checks
that successful temporary shard directories are gone. The original interrupted
125-step run remains an interruption, never relabeled as a pass. Gemma's current
CLI has no adapter-resume entrypoint; these results do not imply one was tested.

All planned host short checks and the selected combined-profile long runs are
now complete. Hosted Mac build/test and Linux dense/hybrid reference jobs for
`3ffe407` all passed. This results-only documentation commit may trigger new CI.
These checks do not qualify every model/configuration or every memory-switch
subset, and they make no downstream accuracy or universal performance claim.

Physical-device training remains outside these host results.
