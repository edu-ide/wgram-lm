# Karpathy Cognitive Core And Data Quality

Sources:

- Dwarkesh Patel interview, `Andrej Karpathy - We're summoning ghosts, not
  building animals`, 2025-10-17: <https://glasp.co/youtube/lXUZvyajciY>
- Sonic AI summary of the same Dwarkesh episode:
  <https://usesonicai.com/episode/61>
- Teahose summary/transcript notes:
  <https://www.teahose.com/podcast/Dwarkesh/Andrej%20Karpathy%20%E2%80%94%20%E2%80%9CWe%E2%80%99re%20summoning%20ghosts%2C%20not%20building%20animals%E2%80%9D>
- No Priors Ep. 80 with Andrej Karpathy, 2024-09-05:
  <https://dexa.ai/nopriors/d/a89d5c3a-6ecd-11ef-8699-afe4daa88ba1>
- GPT-4 Technical Report:
  <https://arxiv.org/abs/2303.08774>
- Semafor discussion of GPT-4 scale reporting:
  <https://www.semafor.com/newsletter/07/28/2023/revealing-the-mysteries-of-chatgpt>
- Hugging Face Llama 3 release notes:
  <https://huggingface.co/blog/llama3>
- Chinchilla compute-optimal training:
  <https://arxiv.org/abs/2203.15556>
- TinyStories:
  <https://arxiv.org/abs/2305.07759>
- Textbooks Are All You Need / phi-1:
  <https://arxiv.org/abs/2306.11644>
- Textbooks Are All You Need II / phi-1.5:
  <https://arxiv.org/abs/2309.05463>

## Corrected Claim

Do not record the social-media version as a direct Karpathy claim:

> A clean 1B-parameter model can reach today's 1.8T frontier model intelligence.

The corrected version is:

> Karpathy argues that current LLMs mix two jobs: reasoning and memorizing a
> noisy internet-scale corpus. If memory/knowledge is moved out to retrieval,
> tools, or a separate memory system, the remaining cognitive core may be much
> smaller, plausibly around 1B parameters or less, especially if trained or
> distilled on high-quality reasoning traces.

## What The Sources Support

- The 2025 Dwarkesh discussion supports the idea of a small cognitive core, not
  a proof that a 1B dense model matches GPT-4-class systems.
- The 2024 No Priors discussion supports the need for inner-monologue or
  problem-solving trajectories. The "billion" there refers to many such traces,
  not a billion parameters.
- The 1.8T number is not official. GPT-4's technical report explicitly avoids
  disclosing model size, architecture, hardware, and dataset-construction
  details. Treat 1.8T as rumor or external reporting, not ground truth.
- Parameter ratios such as `1.8T / 1B = 1800x` are arithmetic, not an empirical
  efficiency law.
- Llama 3's public training note supports the broader point that much of recent
  progress comes from more and better data. It does not prove that all memory
  can be removed from model weights.
- Chinchilla supports balancing model size, data, and compute rather than
  blindly scaling parameters.
- TinyStories and Phi support the practical claim that smaller models can gain
  surprising capability from well-structured synthetic or textbook-quality data.

## QTRM Relevance

This source changes QTRM priorities:

- Treat data quality and trace shape as first-class architecture inputs.
- Keep QTRM as a cognitive/memory adapter around a stable donor before trying
  to replace the donor generator.
- Prefer external memory and retrieval over forcing every fact into QTRM
  weights.
- Use distillation and teacher-forced traces, but watch for synthetic-data
  collapse and low-entropy repetition.
- Do not justify long training only by "Karpathy says 1B is enough." Require
  local diagnostics, ablations, and generation gates.

## Guardrails

- Never cite this source as proof that QTRM can match a frontier model by
  scaling steps.
- Never compare dense 1B parameters directly to rumored MoE total parameters
  without noting active-vs-total uncertainty.
- Never treat clean data as a substitute for tokenizer alignment, target-token
  correctness, causal masking, and decoding tests.
