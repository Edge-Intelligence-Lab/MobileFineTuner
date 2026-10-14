# Opt-in energy-aware throttling

`ops::energy::PowerMonitor` suggests idle time between optimizer updates. It is
model-independent, disabled by default, and available through the installed
`mobile_finetuner/mobile_finetuner.h` umbrella header and `MobileFineTuner::operators`.
No model weights, gradients, learning rates, or CPU/GPU clocks are changed.

This is a **throttling mechanism, not an energy meter or a demonstrated energy
saving**. Extra idle time can reduce average load but increase total run time or
energy. Measure joules per completed workload on the target device before making
energy-efficiency claims. There is no automatic sensor polling, privileged
command, background thread, or network access in this module.

## Explicit schedule in the Qwen example

Build the existing CLI with optional BLAS acceleration:

```bash
cmake -S examples/qwen_lora_finetune -B build-qwen -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-qwen -j4
python3 scripts/test_energy_scheduler.py --binary build-qwen/train_wikitext
```

For actual pretrained training, add `--pm_schedule` to the usual command:

```bash
build-qwen/train_wikitext \
  --model_dir "$QWEN_MODEL_DIR" \
  --jsonl_train "$PIQA_TRAIN_JSONL" \
  --batch_size 8 --seq_len 128 --max_steps 10 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0 --qv_only \
  --seed 42 --no_shuffle --base_weight_storage fp32 \
  --pm_schedule '0-3:100,4-7:0,8-:200' \
  --output_dir outputs/qwen_energy
```

Paths above must point to existing local model/data assets. The JSONL input uses
the existing pretokenized `ids`/`mask` dataset interface. There are no bundled assets
or automatic downloads. CLI schedule support is currently in Qwen
`train_wikitext` (including its synthetic test path), not `train_mmlu` or the
other model CLIs. Other applications can use the C++ module below.

Ranges are **zero-based optimizer update indices**, inclusive at both ends.
`0-3:100` sleeps an additional 100 ms after each of the first four updates;
`8-:200` continues from update index 8 indefinitely. An explicit zero overrides
the sensor policy. Schedule delays do not subtract compute time. Ranges are
sorted and must not overlap; malformed values fail before model loading.
Whitespace inside a schedule is not supported. Programmatic schedules use the
same validation. Empty schedules disable only the explicit schedule.

With gradient accumulation, scheduling happens once per completed optimizer
update, **not per micro-batch**. With no option, existing training and metrics
remain unchanged. When enabled, `step_metrics.csv` includes idle in
`step_time_s`; a separate `energy_metrics.csv` records:

| Column | Meaning |
| --- | --- |
| `step` | One-based completed optimizer update |
| `compute_time_s` | Update wall time before scheduling (including accumulation) |
| `requested_sleep_ms` | Scheduler's additional requested idle |
| `observed_sleep_s` | Measured scheduler/sleep wall time, including OS wakeup overhead |
| `total_step_time_s` | Update time through return from sleep |

These times exclude the later per-step memory cleanup; neither is a joule or
watt measurement. Synthetic tests log scheduling to stdout rather than CSV.

## Sensor-policy integration for any C++ training loop

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
#include <chrono>
#include <thread>

ops::energy::PowerConfig config;
config.check_interval_steps = 5;
config.battery_threshold = 20.0f;
config.temp_threshold = 42.0f;
ops::energy::PowerMonitor monitor(config);

// Inside the application's loop, on a single training thread:
auto start = std::chrono::steady_clock::now();
auto result = trainer.train_step(input_ids, attention_mask, labels);
// Supply application-owned platform readings; NaN means unavailable.
monitor.set_manual_readings(battery_percent, cpu_temp_c);
auto elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
int sleep_ms = monitor.suggest_sleep_ms(zero_based_update_index, elapsed_ms);
if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
```

Each enabled, available source chooses its high/low target steps per second.
Battery strictly below its threshold selects `freq_b_low`; temperature strictly
above its threshold selects `freq_t_low`. The slower available target wins.
Missing readings are initialized to NaN, not fake battery/temperature values;
missing and disabled sources impose no constraint. With no available source,
the policy suggests zero delay. Readings must be refreshed by the application;
the module cannot detect stale readings. Use NaN to invalidate a failed read.

The policy computes `max(0, min(5000, 1000 / target_hz) - compute_time_ms)`,
rounded up to whole milliseconds. A slow update therefore needs no extra idle.
The 5000 ms target-period cap prevents pathological policy delays, so requested
frequencies below 0.2 Hz are not enforced. Explicit schedules are user-specified
nonnegative integer milliseconds and do not use that policy cap.

The first call and every `index % check_interval_steps == 0` refresh the policy;
intervening calls use the cached target period with the current compute time.
Readings set between checks take effect at the next check. Checks also run under
a schedule override. `check_interval_steps <= 0` disables policy suggestions.
Frequency values must be finite and positive for enabled sources. Battery
percent/threshold must be in [0,100]; temperature values are Celsius.

The historical one-argument `suggest_sleep_ms(index)` remains available and
assumes zero compute time. New callers should supply elapsed time. The module
does not itself sleep or modify trainer defaults. Its schedule parser now rejects
malformed entries instead of silently accepting partial schedules.

## Tests

```bash
cmake -S operator -B build -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

`PowerMonitor` tests policy thresholds, disabled/missing inputs, refresh cadence,
compute-time subtraction, schedule boundaries/overlaps, invalid values, and
default-off behavior. The Python integration test above compares ten genuine
forward/backward/Adam updates on tiny synthetic Qwen weights, checking identical
losses and final adapters with and without requested idle. This test verifies
correctness and scheduling, not hardware power reduction.
