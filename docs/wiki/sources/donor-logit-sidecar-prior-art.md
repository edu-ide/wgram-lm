# Donor-Logit Sidecar Prior Art

Purpose: map the current QTRM donor-logit residual design to existing
paper-backed ideas without claiming that any one source already implements the
full QTRM architecture.

## Local References

Official or author code snapshots:

| Area | Local path | Upstream | Commit |
| --- | --- | --- | --- |
| DExperts | `references/official/dexperts` | `https://github.com/alisawuffles/DExperts` | `4ef198fe4cad` |
| FUDGE | `references/official/fudge-controlled-generation` | `https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation` | `32c60893d9e0` |
| GeDi | `references/official/gedi` | `https://github.com/salesforce/GeDi` | `2346c7ee99cd` |
| Proxy-Tuning | `references/official/proxy-tuning` | `https://github.com/alisawuffles/proxy-tuning` | `5f2da2c2783b` |
| Side-Tuning | `references/official/side-tuning` | `https://github.com/jozhang97/side-tuning` | `dea345691fb7` |
| Ladder Side-Tuning | `references/official/ladder-side-tuning` | `https://github.com/ylsung/Ladder-Side-Tuning` | `1798e82e52f2` |

Downloaded PDFs:

| Area | Local PDF | Source |
| --- | --- | --- |
| DExperts | `references/papers/logit_sidecar/dexperts_2021_acl_long_522.pdf` | `https://aclanthology.org/2021.acl-long.522/` |
| FUDGE | `references/papers/logit_sidecar/fudge_2104.05218.pdf` | `https://arxiv.org/abs/2104.05218` |
| GeDi | `references/papers/logit_sidecar/gedi_2009.06367.pdf` | `https://arxiv.org/abs/2009.06367` |
| Proxy-Tuning | `references/papers/logit_sidecar/proxy_tuning_2401.08565.pdf` | `https://arxiv.org/abs/2401.08565` |
| Side-Tuning | `references/papers/logit_sidecar/side_tuning_1912.13503.pdf` | `https://arxiv.org/abs/1912.13503` |
| Ladder Side-Tuning | `references/papers/logit_sidecar/ladder_side_tuning_2206.06522.pdf` | `https://arxiv.org/abs/2206.06522` |
| AdapterFusion | `references/papers/logit_sidecar/adapterfusion_2021_eacl_main_39.pdf` | `https://aclanthology.org/2021.eacl-main.39/` |

## What Carries Over To QTRM

Current QTRM is closest to a composition of four known ideas:

1. **Logit-level steering**
   DExperts, FUDGE, GeDi, and Proxy-Tuning all support the broad idea that a
   base language model can be steered by adding or reweighting another model's
   token-level signal at decoding time.

2. **Frozen backbone plus side network**
   Side-Tuning and Ladder Side-Tuning support the broad idea that a frozen
   backbone can remain intact while a trainable side path consumes backbone
   features and contributes task-specific behavior.

3. **Adapter isolation**
   LoRA, QLoRA, AdapterFusion, and PEFT support the engineering rule that new
   capability should first live in isolated trainable modules before touching
   the base donor weights.

4. **Latent workspace**
   Perceiver, Q-Former, and Flamingo support the learned-query/latent-token
   pattern that QTRM uses for in-context workspace computation.

## What Is QTRM-Specific

No downloaded prior-art source should be read as a direct implementation of
QTRM. The QTRM-specific composition is:

```text
Qwen frozen donor hidden states
+ Qwen donor logits as base language policy
+ trainable QTRM latent workspace
+ recursive QTRM core
+ small QTRM residual logits
+ optional MemoryOS retrieved evidence
```

The current generation formula is:

```text
final_logits = donor_logits_scale * donor_logits
             + qtrm_logits_scale  * qtrm_logits
```

With the current stable probe:

```text
donor_logits_scale = 1.0
qtrm_logits_scale  = 0.1
```

## Design Rules

- Do not call QTRM "LoRA"; LoRA modifies internal donor weights through
  low-rank deltas, while QTRM is a sidecar residual model.
- Do not call QTRM a standalone loop LM while donor logits remain the base
  language policy.
- Always compare donor-only versus donor-plus-QTRM residual on the same prompt,
  retrieval evidence, decoding settings, and score function.
- Treat the QTRM residual as useful only when it improves a target metric without
  regressing donor fluency or causing logit entropy/repetition collapse.

## Diagram Authoring Note

Mermaid is useful for wiki diagrams and diffs. Publication-style architecture
figures, like the original Transformer diagram, should be produced as vector art:

- **TikZ/LaTeX** for paper-native reproducibility.
- **draw.io/diagrams.net** for fast block diagrams.
- **Figma, Illustrator, or Inkscape** for polished vector figures.
- **SVG generated from a small script** when exact layout should stay versioned.

QTRM should keep Mermaid diagrams for implementation review and add a separate
SVG/PDF figure when preparing a paper-style overview.
