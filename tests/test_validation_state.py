"""Regression for explicit fixed-prefix validation callbacks and state isolation."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mbt.runtime import TrainingLifecycle


class FakeTorch:
    distributed = SimpleNamespace(get_rank=lambda: 0)
    @staticmethod
    def is_tensor(value): return False
    @staticmethod
    def no_grad(): return nullcontext()


class ValidationTests(unittest.TestCase):
    def test_resumed_validation_sampler_is_rebuilt_at_zero(self):
        calls = []
        old = SimpleNamespace(dataset="same-validation-dataset", consumed=500)
        backend = SimpleNamespace(build_pretraining_data_loader=lambda dataset, consumed:
                                  calls.append((dataset, consumed)) or (dataset, consumed))
        owner = TrainingLifecycle({}, backend, iter)
        original = ("train", [old], None)
        result = owner.data_loaders_ready(original)
        self.assertIs(result, original)
        self.assertEqual(owner.fixed_loader, ("same-validation-dataset", 0))
        self.assertEqual(calls, [("same-validation-dataset", 0)])

    def test_eval_restores_state_and_rejects_router_or_train_cursor_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(consumed_train_samples=40, consumed_valid_samples=100,
                                   skipped_train_samples=0, eval_iters=1, eval_global_batch_size=2)
            module = SimpleNamespace(training=True, expert_bias=[.1, .2], qb_margin_chunks=[])
            model = SimpleNamespace(modules=lambda: [module])
            rerun = SimpleNamespace(get_mode=lambda: "train", set_mode=lambda value: None)
            backend = SimpleNamespace(get_args=lambda: args, torch=FakeTorch,
                                      get_rerun_state_machine=lambda: rerun)
            owner = TrainingLifecycle({"output_dir": tmp}, backend, iter)
            owner.fixed_loader = ["sample0", "sample1"]
            config = SimpleNamespace(timers="training-timers")
            with patch("mbt.runtime.preserve_rng", lambda _: nullcontext()):
                with self.assertRaises(ValueError):
                    with owner.validation_context("validation", [model], config, 5, None) as iterator:
                        self.assertEqual(list(iterator), ["sample0", "sample1"])
                        self.assertEqual(args.consumed_valid_samples, 0)
                        args.consumed_valid_samples = 2
                        module.training = False
                        module.expert_bias.append(99.)
                        args.consumed_train_samples += 2
                        owner.on_validation_result("validation", {"lm loss": 2.5}, False)
            self.assertEqual(args.consumed_train_samples, 40)
            self.assertEqual(args.consumed_valid_samples, 100)
            self.assertEqual(module.expert_bias, [.1, .2])
            self.assertTrue(module.training)
            self.assertFalse(owner.validation_active)
            self.assertFalse((Path(tmp) / "validation.jsonl").exists())

    def test_success_records_metrics_after_restoring_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(consumed_train_samples=8, consumed_valid_samples=17,
                                   skipped_train_samples=0, eval_iters=1, eval_global_batch_size=2)
            module = SimpleNamespace(training=True, expert_bias=[.1, .2], qb_margin_chunks=[])
            model = SimpleNamespace(modules=lambda: [module])
            rerun = SimpleNamespace(get_mode=lambda: "train", set_mode=lambda value: None)
            backend = SimpleNamespace(get_args=lambda: args, torch=FakeTorch,
                                      get_rerun_state_machine=lambda: rerun)
            owner = TrainingLifecycle({"output_dir": tmp}, backend, iter)
            owner.fixed_loader = ["sample0", "sample1"]
            with patch("mbt.runtime.preserve_rng", lambda _: nullcontext()):
                with owner.validation_context("validation", [model], None, 5, None):
                    args.consumed_valid_samples = 2
                    module.training = False
                    owner.on_validation_result("validation", {"lm loss": 2.5}, False)
            self.assertEqual(args.consumed_train_samples, 8)
            self.assertEqual(args.consumed_valid_samples, 17)
            self.assertTrue(module.training)
            self.assertIn('"validation_start_sample": 0', (Path(tmp) / "validation.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
