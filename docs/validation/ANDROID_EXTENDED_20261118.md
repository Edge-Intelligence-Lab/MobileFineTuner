# Extended physical Android validation — 2026-11-18

This is an incremental record. Queued/running jobs are not completed passes.
The [initial Android record](ANDROID_20261118.md) remains valid, including its
failed strict post-Adam coordinate diagnostic. The additional checks below
separate fixed-point numerical correctness from long-run execution stability.

## Fixed-point numerical checks: passed

On the same vivo V2505A/SM8850, Android16/API36 arm64 CPU, the full-size
Qwen3.5-0.8B text model completes all eight profiles: ordinary/streamed attention
crossed with no memory switch, checkpointing, offload, and checkpointing+offload.
All profiles explicitly use FP32 frozen weights. The independent HF fixture
uses nonzero initial LoRA A/B, rank3, alpha6, batch2, sequence4, padding masks,
and an SGD update. It does not replay the earlier rank2 three-step Adam run.

Each profile passes the unchanged existing HF logits, loss, every adapter
gradient, SGD update, reload and post-update checks. The direct comparison
additionally checks byte-identical pre-update parameters, token IDs, labels and
masks. Each of the 372 adapter gradients is compared separately; there is no
aggregation across tensors that could conceal a broken module.

Criteria declared before running the phone check:

- Gradient L2 error <= 1e-3 * reference L2 norm + 2e-7, and maximum absolute
  error <= 2e-3 * reference maximum magnitude + 2e-6. These pairwise bounds
  account for two implementations, each still subject to the existing HF
  bounds of 5e-4/1e-7 and 1e-3/1e-6 respectively.
- Pairwise logits absolute/relative tolerance2e-3/2e-4; loss4e-4/6e-4.
- All compared float tensors finite; parameters and inputs byte-identical.

Observed direct comparison against ordinary attention without memory switches:

| Profiles | Maximum logit difference | Loss difference | Maximum gradient difference | Largest per-tensor gradient relative L2 error |
| --- | ---: | ---: | ---: | ---: |
| All four ordinary-attention configurations | 0 | 0 | 0 | 0 |
| All four streamed-attention configurations | 6.2942505e-5 | 4.7683716e-7 | 4.0054321e-5 | 2.0575618e-5 |

This supports the covered fixed-point forward/backward and SGD semantics. It
neither establishes bitwise identity between attention kernels nor changes the
earlier failed2e-5 final-adapter coordinate diagnostic. It does not prove full
Adam trajectory parity, downstream accuracy or every model/input size.

## Native training acceptance queue: launched, not complete

The controller is launched on 2026-11-18 evening CST, after the fixed-point gate.
Qwen3-0.6B and Qwen2.5-0.5B have already completed native-base three-step tests:
all392/336 adapter tensors finite and changed between snapshots1 and3, all B
matrices nonzero, and final exports equal to snapshot3. Their logged losses
are7.12114/7.7135/5.8519 and5.40853/6.0804/6.6937 respectively.

The serial queue contains27 further jobs, with explicit commands in the local
plan. It starts with short execution/artifact checks, then long runs:

| Path | Short preflight | Long run planned | Adapter reload planned |
| --- | --- | --- | --- |
| Qwen2.5-0.5B, Qwen3-0.6B, Qwen3.5-0.8B | Native baseline plus combined switches; Qwen3.5 also matched FP32 baseline | 300 updates each with all three memory switches | 30 updates each with fresh Adam |
| Gemma3 270M and 1B-PT | Baseline and combined switches, 3 updates each | 300 updates each with combined switches | CLI does not expose adapter resume |
| GPT-2 small and medium | Baseline and adapter reload, 3 updates each | 300 baseline updates each | 30 updates each with fresh Adam |
| GPT-2 small Q8_0 | Quantize/save and load-Q8/adapter-reload, 3 updates each | 300 Q8_0 updates | 30 updates from saved Q8 base and adapter |
| Qwen3.5 larger sequence | Separate short stress | 5 sequence32 updates | Not a long-context qualification |

Training queue protocol: real local weights, WikiText raw excerpt, sequence8
(except the separate stress), batch/accumulation1, rank2, alpha4, dropout0,
LR2e-4, clipping norm1. This excerpt-based seq8 device stability protocol is
smaller than the seq32/full-WikiText Mac protocol and is not a model-quality
benchmark. Combined mode means streamed attention + activation checkpointing +
FP32 frozen-weight offload, 1536 MiB resident budget. GPT CLI does not expose
those three switches. Q8_0 is only a GPT-2-small CLI path in this PR.

The host controller runs under caffeinate. Each device job has its own detached
supervisor, so the current job's resource guard survives a host disconnect.
Jobs are serial; loss/update coverage, finite adapters, nonzero B matrices,
checkpoint changes/final equality, supported reloads and actual offload I/O are
validated before proceeding. No optimizer moments, global step or RNG state
are restored by adapter reload. Snapshots are retained every100 long updates,
every10 reload updates and every short update.

Device guards stop only this queue's child on RSS above7 GiB, available memory
below1500 MiB, battery below10%, battery temperature above42 C, operator stop,
or per-job timeout. Each job waits for charging, battery >=15% and battery
temperature <=38 C before starting. Temperature is a battery reading, not chip
temperature. Resource samples every~3 seconds can miss transient peaks. Own
successful temporary offload shards are removed after evidence validation;
failed evidence is retained and jobs are not silently retried.

## Reproduction and evidence

The native training code is unchanged from the published `b44f492` source
snapshot (implementation includes `3ffe407`). This addition instruments only
`test_qwen_hf_parity`; training/reference thresholds are unchanged. Android
executables use NDK26.1.10909125, android29, arm64-v8a, static C++ runtime,
Release, BLAS disabled and warnings-as-errors.

To collect the fixed-point snapshots, use a fresh output directory:

```sh
MFT_QWEN_MEMORY_MATRIX=1 MFT_QWEN_PARITY_SNAPSHOT_DIR=/unique/output \
  test_qwen_hf_parity /fixture/root
python3 scripts/compare_qwen_profile_snapshots.py \
  --root /unique/output/qwen35_full_fp32 --output /unique/comparison.json
```

The same instrumentation and comparator pass12 tiny HF fixture configurations
and96 profiles on Mac. A deliberately corrupted gradient is correctly rejected.
Both Linux HF CI jobs now exercise the snapshot comparison in addition to
existing reference checks. Hosted CI status is separate from local validation.

Raw fixed-point tensors, native logs, hashes, supervisor commands, the27-job
plan, queue status and resource samples are under local ignored
`runs/device_extended_20261118/`. `REPORT.md` and `queue_status.json` identify
actual completed/running/pending jobs. Earlier native short tests remain under
`runs/device_validation_20261118/`. Device identifiers and private contents,
model weights and generated binaries are not committed. Long-run results
must be added only after completion; this launch record does not certify them.

Pinned executable SHA-256 (all six device copies verified against host):

| Executable | SHA-256 |
| --- | --- |
| phone_gemma270 | `64349b52cf4f68201967a3719ba6905bd68bf7ab5fa4e3f0954d720dfdaf111d` |
| phone_gemma1b | `ec2dad223a0b51a9cdcd0a73b2b36b434db4960a026c3f1e3c082be8de25fb90` |
| phone_gpt2small | `7d0ac147810c666e6c883929ff4e0810490ad10712ab7cf73e8a93e670350f56` |
| phone_gpt2medium | `259012eed559c4397542b849d81c972a05503dc01637948c71c7351d668e9745` |
| test_qwen_hf_parity | `644aea35e52065e36cf56a1537e69a90865f41cc2605dcaecaf618fb69cf0ae9` |
| train_wikitext | `05f65da1dce0f82dc174dbce66c9d8813c2b27fe70f4f4d8911f08cfcb610221` |
