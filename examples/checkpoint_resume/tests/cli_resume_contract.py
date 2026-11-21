"""Source-level integration guardrails; real numerical coverage is in run_real_resume.py.

Run without a build: python3 tests/cli_resume_contract.py
These checks do not claim numerical equivalence or durable filesystem behavior.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "qwen": ROOT / "qwen_lora_finetune/src/main.cpp",
    "gemma": ROOT / "gemma_3_270m_lora_finetune/src/benchmark_memory.cpp",
}
HELPERS = ROOT / "qwen_lora_finetune/src/resume_helpers.h"


class ResumeContract(unittest.TestCase):
    def setUp(self):
        self.sources = {name: path.read_text() for name, path in ENTRYPOINTS.items()}
        self.helpers = HELPERS.read_text()

    def test_invocation_precedes_cli_and_model_loading(self):
        for name, source in self.sources.items():
            with self.subTest(entrypoint=name):
                main = source[source.index("int main("):]
                self.assertIn(f'prepare_invocation(argc, argv, "{name}", resume_cli::batch_validator(parse_cli))', main)
                self.assertLess(main.index("prepare_invocation("), main.index("parse_cli("))
                self.assertLess(main.index("resume_cli::configure("), main.index("from_pretrained("))
                self.assertIn("argument.data()", main)

    def test_restore_precedes_initial_eval_and_next_batch(self):
        for name, source in self.sources.items():
            with self.subTest(entrypoint=name):
                start = source.index("auto restored = session->restore(named, *adam,")
                restored = source[start:]
                self.assertLess(restored.index("prepare_restore("), restored.index("print_step0();"))
                self.assertIn("next.restore(state, total_steps, layout)", self.helpers)
                self.assertIn("train.prepare_import", self.helpers)
                self.assertIn("valid.prepare_import", self.helpers)
                self.assertIn("for (int step = trainer.completed_updates; step < run_steps; ++step)", restored)
                self.assertIn("make_scheduler(step, total_steps,", restored)
                self.assertIn("trainer.completed_updates == total_steps", restored)

    def test_update_boundary_follows_eval_cleanup_and_flush(self):
        for name, source in self.sources.items():
            with self.subTest(entrypoint=name):
                loop = source[source.index("for (int step = trainer.completed_updates;"):]
                boundary = loop.index("if (commit_boundary(false)) return 0;")
                for required in ("const float valid_ppl = evaluate_valid();",
                                 "microbatches.clear();", "grads.clear();",
                                 "gradient_accumulator = GradientAccumulator(0);",
                                 "Engine::instance().clear_graph();", "telemetry.flush();",
                                 "checkpoint_trace.flush();", 'session->event({{"event", "update"}'):
                    self.assertLess(loop.index(required), boundary)

    def test_authoritative_event_and_suspend_ack(self):
        for name, source in self.sources.items():
            with self.subTest(entrypoint=name):
                for field in ('{"event", "update"}', '{"type", "update"}',
                              '{"completed_updates", step + 1}', '{"update_id", step + 1}',
                              '{"loss", avg_loss}', '{"lr", cur_lr}', '{"grad_norm", grad_norm}',
                              '{"total_tokens", total_tokens}', '{"ema_loss", ema_loss}',
                              '{"core_step_sec", step_seconds}', '"rss_bytes"',
                              '"process_peak_so_far_bytes"'):
                    self.assertIn(field, source)
                commit = source.index("session->commit(named, *adam,")
                self.assertLess(commit, source.index('"SUSPENDED completed_updates="'))
                self.assertIn('finished ? "completed" : (suspend ? "suspended" : "running")', source)

    def test_best_and_session_outputs(self):
        for source in self.sources.values():
            self.assertIn('"/lora_best_step" + std::to_string(step + 1)', source)
            self.assertIn('resume_cli::finite_or_null(valid_ppl)', source)
            self.assertIn('resume_cli::validate_adapter_file(path, named)', source)
        self.assertIn('{"best_path", best_path}', self.helpers)
        self.assertIn('{"best_valid_ppl", finite_or_null(best_valid_ppl)}', self.helpers)
        self.assertIn('session.artifact_path("lora.safetensors")', self.helpers)
        self.assertNotIn('session.artifact_path("")', self.helpers)
        self.assertIn('cli.measurement_dir = session.session_dir()', self.helpers)
        self.assertIn('cli.eval_out = session.session_dir() + "/eval.jsonl"', self.helpers)

    def test_parameter_inventory_matches_model_order(self):
        for name, targets in (("qwen", ["attn.q", "attn.v", "attn.k", "attn.proj"]),
                              ("gemma", ["attn.q", "attn.k", "attn.v", "attn.proj",
                                         "mlp.gate", "mlp.up", "mlp.down"])):
            source = self.sources[name]
            positions = [source.index(f'add_linear(named, adapter_layout, i, "{target}"')
                         for target in targets]
            self.assertEqual(positions, sorted(positions))
            self.assertIn("resume_cli::validate_parameters(named, trainable)", source)

    def test_assets_and_explicit_restrictions(self):
        for required in ('session.track_assets(assets)', '"/config.json"', '"/model.safetensors"',
                         '"/tokenizer.json"', '"/wiki.train.raw"', '"/wiki.valid.raw"',
                         'canonical(cli.resume_from)', 'canonical(cli.pretokenized_meta)',
                         'cli.lora_dropout != 0', 'cli.shard_fp16_disk', 'cli.fixed_batch_steps != 0',
                         'cli.stop_after_steps != 0', 'slices.size() > 1'):
            self.assertIn(required, self.helpers)
        for name in ("QWEN_DEBUG", "DISABLE_QK_NORM"):
            self.assertIn(f'std::getenv("{name}")', self.helpers)
        self.assertNotIn('"/wiki.test.raw"', self.helpers)
        self.assertIn("session && cli.export_initial_only", self.sources["gemma"])

    def test_runtime_state_and_fail_closed_contract(self):
        for required in ('"schema_version"', '"completed_updates"', '"total_steps"',
                         '"ema_init"', '"ema_loss"', '"total_tokens"', '"timing"',
                         '"adapter_layout"', 'train.export_state()', 'valid.export_state()',
                         'train.prepare_import(', 'valid.prepare_import(', 'SAVE_FAILED: '):
            self.assertIn(required, self.helpers)
        self.assertIn('restart with --resume_state', self.helpers)


if __name__ == "__main__":
    unittest.main()
