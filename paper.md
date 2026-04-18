---
title: 'Axis-HFE: A Python Library for Multi-Dimensional Hypothesis Space Exploration in LLM Reasoning'
tags:
  - Python
  - artificial intelligence
  - large language models
  - reasoning
  - hypothesis generation
  - multi-objective optimization
authors:
  - name: Tadashi Mazusaki
    orcid: 0009-0005-2463-9380
    affiliation: 1
affiliations:
  - name: Da-P-AIP, Independent Researcher
    index: 1
date: 19 April 2026
bibliography: paper.bib
---

# Summary

Axis-HFE (Hypothesis Field Engine) is a Python library that transforms Large Language Model (LLM) reasoning from single-answer selection into multi-dimensional hypothesis space exploration. Rather than prompting an LLM to return the most probable answer, Axis-HFE deploys multiple hypotheses simultaneously, evaluates each on a six-dimensional vector, and evolves them across iterative generations through nonlinear synthesis, leap exploration, and self-correction. The result is a reasoning process where the final answer can emerge as a novel synthesis that was not present in any of the initial hypotheses.

The library supports three LLM backends — Ollama (local, free), OpenAI, and Anthropic — through a unified asynchronous interface, and is installable via `pip install axis-hfe`.

# Statement of Need

Standard LLM inference selects a single response by maximizing the probability of the next token. While techniques such as Chain-of-Thought prompting [@wei2022chain] and Self-Consistency [@wang2022self] improve reasoning quality, they remain fundamentally selection-based: the final answer is chosen from a set of candidates, not synthesized from them.

More recent frameworks such as Tree of Thoughts [@yao2023tree] and Graph of Thoughts [@besta2024graph] extend reasoning into tree or graph structures, but their evaluation criteria remain implicit and their outputs are still chosen rather than constructed.

Axis-HFE addresses this gap by introducing an explicit, multi-dimensional evaluation space and a synthesis-first architecture. The key distinctions are:

| Conventional LLM Reasoning | Axis-HFE |
|---|---|
| Selects one answer by maximum probability | Deploys multiple hypotheses across a field |
| Evaluation criteria are implicit | Explicit 6-dimensional vector scoring |
| No constraint management | Hard constraints ensure safe operating region |
| Converges in one pass | Hypotheses evolve across multiple generations |
| Output is selected, not created | New answers emerge through vector synthesis |

This design is motivated by problems where no single perspective is sufficient — creative tasks, strategic decision-making, and multi-stakeholder scenarios where correctness, feasibility, novelty, and harmony must all be balanced simultaneously.

# Design and Implementation

## Six-Dimensional Evaluation Vector

Each hypothesis is represented as a point in a six-dimensional evaluation space:

- **accuracy**: validity and correctness relative to the problem
- **consistency**: internal logical coherence
- **risk**: probability and severity of failure or side effects (lower is better)
- **novelty**: originality and departure from conventional approaches
- **feasibility**: practicality of implementation
- **divergence**: breadth of perspective explored

The engine computes a weighted Euclidean distance between each hypothesis vector and a configurable ideal vector, yielding a scalar score used for ranking. Pre-defined ideal presets (`default`, `creative`, `safe`, `balanced`) allow users to tune the exploration toward different objectives without manual configuration.

## Evolution Loop

Each call to `engine.run()` executes the following pipeline for a configurable number of iterations:

1. **Generator**: prompts the LLM to produce N structurally diverse hypotheses, each pre-scored on the 6D vector, in a single call.
2. **Vectorizer**: re-evaluates hypotheses whose vector has not yet been assigned (e.g., fused candidates).
3. **Relational Layer**: augments each hypothesis score with a relational sub-score measuring mutual understanding, alignment, respect, and harmony — ensuring the highest-ranked hypothesis is also contextually appropriate.
4. **Constraint Engine**: hard-clips vectors that violate safety boundaries (e.g., `risk > 0.40`, `feasibility < 0.60`) by projecting them into a safe operating region.
5. **Evaluator**: ranks all candidates by weighted distance from the ideal vector.
6. **Fusion Engine**: constructs new hypotheses by linearly and nonlinearly blending the top-2 vectors — the blended result can exceed either parent on any axis.
7. **Jump Explorer**: generates a hypothesis displaced away from the worst candidate and toward the best (opposite-jump), plus random perturbations of the top-3.
8. **Self Corrector**: nudges each top-3 hypothesis closer to the ideal on every axis independently.

All generated candidates are pooled with the current generation, and the top-5 survive to the next iteration. This selection pressure ensures progressive convergence while maintaining diversity through the jump and perturbation operators.

## Security

Because the library is intended for embedding in applications that accept untrusted input, Axis-HFE implements several security measures: prompt injection mitigation via system/user role separation and input delimiters; SSRF protection by validating `ollama_base_url` against known cloud metadata endpoints; resource limits enforced at `EngineConfig` construction (`iterations ≤ 10`, `hypothesis_count ≤ 20`); API key masking in log output; and output sanitization on LLM-generated content.

## Emergent Synthesis

In benchmark tests using `gemma4:4b` via Ollama with `iterations=3` and the `creative` preset, the best-ranked hypothesis after evolution was consistently a synthesis that did not appear in the initial generation. In one business-domain test, the engine produced a pricing model innovation that was absent from all three seed hypotheses, combining stability properties from one, novelty from another, and emotional value from a third. This emergent behavior — answers constructed outside the initial hypothesis space — is the core design goal of Axis-HFE.

# Acknowledgements

The author thanks the open-source communities behind httpx [@httpx] and Pydantic [@pydantic] for the foundational libraries on which Axis-HFE is built.

# References
