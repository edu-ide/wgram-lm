import importlib.util
import sys
import unittest
from pathlib import Path


def load_capacity_module():
    path = Path("scripts/540_plan_prefixlm_4090_capacity.py")
    spec = importlib.util.spec_from_file_location("prefixlm_4090_capacity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrefixLM4090CapacityPlannerTests(unittest.TestCase):
    def test_memory_floor_distinguishes_optimizer_choices(self):
        module = load_capacity_module()

        adamw = module.estimate_memory_floor(1_000_000, optimizer="adamw")
        galore = module.estimate_memory_floor(1_000_000, optimizer="galore_adamw8bit")

        self.assertGreater(adamw["floor_before_activations_gb"], galore["floor_before_activations_gb"])

    def test_meta_model_parameter_count_works_without_allocating_weights(self):
        module = load_capacity_module()
        trainer = module.load_trainer_module()

        count = module.model_parameter_count(
            trainer,
            vocab_size=1024,
            seq_len=32,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            train_think_steps=1,
        )

        self.assertGreater(count, 0)

    def test_launch_command_omits_activation_checkpointing_by_default(self):
        module = load_capacity_module()

        command = module.launch_command(
            {
                "name": "tiny",
                "d_model": 32,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_ff": 64,
            },
            sampled_data="/tmp/sample",
            out_dir="/tmp/out",
            steps=10,
            batch_size=1,
            seq_len=32,
            optimizer="galore_adamw8bit",
            lr=2.2e-4,
            seed=1,
        )

        self.assertIn("--length-bucketed-batches", command)
        self.assertNotIn("--activation-checkpointing", command)
        self.assertIn("--loss-kernel auto", command)
        self.assertIn("--optimizer galore_adamw8bit", command)

    def test_launch_command_can_opt_into_activation_checkpointing(self):
        module = load_capacity_module()

        command = module.launch_command(
            {
                "name": "tiny",
                "d_model": 32,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_ff": 64,
            },
            sampled_data="/tmp/sample",
            out_dir="/tmp/out",
            steps=10,
            batch_size=1,
            seq_len=32,
            optimizer="galore_adamw8bit",
            lr=2.2e-4,
            seed=1,
            activation_checkpointing=True,
        )

        self.assertIn("--activation-checkpointing", command)

    def test_stage86_launcher_keeps_activation_checkpointing_opt_in(self):
        launcher = Path("scripts/launch_stage86_local_913m_optimized_smoke.sh")

        text = launcher.read_text(encoding="utf-8")

        self.assertIn("ACTIVATION_CHECKPOINTING", text)
        self.assertIn("setsid", text)
        self.assertIn("CMD=(", text)
        self.assertIn('BATCH_SIZE="${BATCH_SIZE:-1}"', text)
        self.assertIn('--batch-size "${BATCH_SIZE}"', text)
        self.assertIn('LOG_EVERY="${LOG_EVERY:-50}"', text)
        self.assertIn('--log-every "${LOG_EVERY}"', text)
        self.assertNotIn("    --activation-checkpointing \\\n", text)

    def test_default_candidates_include_913m_smoke_target(self):
        module = load_capacity_module()

        known = [str(row["name"]) for row in module.DEFAULT_CANDIDATES]

        self.assertIn("risk_913m", known)


if __name__ == "__main__":
    unittest.main()
