import json
import unittest


class AsiCognitiveLoopContractTests(unittest.TestCase):
    def test_memory_write_requires_all_validation_gates(self):
        from wgram_lm.agentic.cognitive_loop import CandidateMemoryWrite

        write = CandidateMemoryWrite(
            content="The archive code is VX-913.",
            source_ids=("archive.md#0",),
            grounding_passed=True,
            contradiction_passed=False,
            novelty_passed=True,
            usefulness_passed=True,
            regression_passed=True,
        )

        self.assertEqual(write.commit_status, "quarantine")
        self.assertFalse(write.can_commit)
        self.assertEqual(write.failure_reasons, ("contradiction_failed",))

    def test_trace_transition_is_json_serializable_training_source(self):
        from wgram_lm.agentic.cognitive_loop import (
            Action,
            TraceTransition,
            VerifierResult,
        )

        transition = TraceTransition(
            task_id="memory-qa-001",
            step=0,
            state_summary="Question asks for the current archive code.",
            visible_prompt_hash="sha256:abc",
            workspace_evidence_ids=("archive.md#0",),
            action=Action.VERIFY_EVIDENCE,
            action_args={"claim": "archive code is VX-913"},
            observation="Evidence supports VX-913.",
            verifier=VerifierResult(
                status="SUPPORTED",
                reward=1.0,
                reason="Evidence directly states the code.",
            ),
            memory_writes=(),
            skill_writes=(),
            checkpoint="runs/qtrm/last.pt",
            mode="asi_cognitive_loop_v0",
        )

        payload = transition.to_json_dict()

        json.dumps(payload)
        self.assertEqual(payload["action"], "VERIFY_EVIDENCE")
        self.assertEqual(payload["verifier"]["status"], "SUPPORTED")
        self.assertEqual(payload["policy_role"], "residual_controller")

    def test_asi_contract_rejects_claims_without_causal_gates(self):
        from wgram_lm.agentic.cognitive_loop import asi_cognitive_loop_contract

        contract = asi_cognitive_loop_contract()
        report = contract.validate_gate_report(
            {
                "evidence_path": True,
                "latent_core": False,
                "world_model": True,
                "verifier": True,
                "self_improvement": False,
                "agent_memory": True,
            }
        )

        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failing_gates"], ("latent_core", "self_improvement"))
        self.assertIn("latent_core", contract.required_causal_gates)
        self.assertIn("self_improvement", contract.required_causal_gates)

    def test_scripted_harness_records_retrieve_verify_answer_trace(self):
        from wgram_lm.agentic.harness import ScriptedCognitiveHarness

        harness = ScriptedCognitiveHarness()
        run = harness.run_memory_qa(
            task_id="archive-code",
            question="What is the archive access code?",
            evidence=[
                {
                    "source": "archive.md",
                    "chunk_id": 0,
                    "text": "The archive access code is VX-913.",
                }
            ],
            candidate_answer="Answer: VX-913",
            expected_answers=("VX-913",),
        )

        payload = run.to_json_dict()

        json.dumps(payload)
        self.assertEqual(run.final_status, "SUPPORTED")
        self.assertEqual([step.action.value for step in run.transitions], [
            "RETRIEVE_MEMORY",
            "VERIFY_EVIDENCE",
            "ANSWER",
        ])
        self.assertEqual(payload["transitions"][1]["verifier"]["reward"], 1.0)
        self.assertEqual(payload["final_answer"], "Answer: VX-913")
        self.assertEqual(
            payload["transitions"][0]["visible_prompt_hash"],
            payload["context_tape"]["context_hash"],
        )
        self.assertEqual(
            payload["transitions"][1]["visible_prompt_hash"],
            payload["context_tape"]["context_hash"],
        )
        self.assertIn(
            "The archive access code is VX-913.",
            payload["context_tape"]["workspace_context"],
        )

    def test_typed_context_tape_renders_prompt_workspace_verifier_from_one_source(self):
        from wgram_lm.agentic.context_tape import ContextItem, ContextItemType, TypedContextTape

        tape = TypedContextTape(
            task_id="archive-code",
            items=(
                ContextItem(ContextItemType.SYSTEM, "Answer only from validated evidence."),
                ContextItem(ContextItemType.USER, "What is the archive access code?"),
                ContextItem(
                    ContextItemType.EVIDENCE,
                    "The archive access code is VX-913.",
                    item_id="archive.md#0",
                    metadata={"source": "archive.md", "trust": "validated"},
                ),
                ContextItem(
                    ContextItemType.VERIFIER_RESULT,
                    "SUPPORTED",
                    metadata={"reward": 1.0},
                ),
            ),
        )

        prompt = tape.render_chat_prompt()
        workspace = tape.render_workspace_context()
        verifier = tape.render_verifier_input(candidate_answer="Answer: VX-913")
        training = tape.to_training_record()

        json.dumps(training)
        self.assertIn("<evidence id=\"archive.md#0\">", prompt)
        self.assertIn("The archive access code is VX-913.", prompt)
        self.assertEqual(workspace, "The archive access code is VX-913.")
        self.assertEqual(verifier["claim"], "Answer: VX-913")
        self.assertEqual(verifier["evidence_ids"], ["archive.md#0"])
        self.assertEqual(training["context_hash"], tape.context_hash)

    def test_causal_loop_gate_requires_baseline_gain_and_component_drops(self):
        from wgram_lm.agentic.causal_gate import evaluate_causal_loop_gate

        rejected = evaluate_causal_loop_gate(
            {
                "scripted_harness": 0.70,
                "donor_harness": 0.72,
                "qtrm_harness": 0.75,
                "qtrm_latent_core_off": 0.75,
                "qtrm_world_model_off": 0.70,
                "qtrm_verifier_off": 0.60,
            },
            min_gain=0.02,
            min_drop=0.03,
        )

        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("latent_core_not_causal", rejected["failed_checks"])

        accepted = evaluate_causal_loop_gate(
            {
                "scripted_harness": 0.70,
                "donor_harness": 0.72,
                "qtrm_harness": 0.78,
                "qtrm_latent_core_off": 0.72,
                "qtrm_world_model_off": 0.73,
                "qtrm_verifier_off": 0.70,
            },
            min_gain=0.02,
            min_drop=0.03,
        )

        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["baseline"], "qtrm_harness")
        self.assertGreater(accepted["gain_over_donor_harness"], 0.02)

    def test_trace_replay_records_expose_action_targets_and_rewards(self):
        from wgram_lm.agentic.harness import ScriptedCognitiveHarness
        from wgram_lm.agentic.trace_replay import TraceReplayDataset

        run = ScriptedCognitiveHarness().run_memory_qa(
            task_id="archive-code",
            question="What is the archive access code?",
            evidence=[
                {
                    "source": "archive.md",
                    "chunk_id": 0,
                    "text": "The archive access code is VX-913.",
                }
            ],
            candidate_answer="Answer: VX-913",
            expected_answers=("VX-913",),
        )
        replay = TraceReplayDataset.from_runs((run,))

        self.assertEqual(len(replay), 3)
        first = replay[0]
        last = replay[2]
        json.dumps(first)
        self.assertEqual(first["action_target"], "RETRIEVE_MEMORY")
        self.assertEqual(last["action_target"], "ANSWER")
        self.assertEqual(last["reward"], 1.0)
        self.assertEqual(first["context_hash"], run.context_tape.context_hash)
        self.assertIn("<user>", first["chat_prompt"])
        self.assertIn("VX-913", first["workspace_context"])


if __name__ == "__main__":
    unittest.main()
