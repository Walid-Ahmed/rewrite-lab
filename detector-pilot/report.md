# Detector pilot, 2026-11-18: result

**Verdict: negative.** The pilot fails two of the four conditions fixed in `plan.json` before any result existed
(plan sha256 `948c8637…`, unchanged). The model separates the training sources almost perfectly but does not detect
AI writing in general: it flags 91% of human-written arXiv abstracts.

One model, one seed (3407), no checkpoint selection, threshold fixed once from validation human texts, joint
holdout scored once. Model sha256 `37b59a9a…`. Nothing in production changed.

## What was built

| Split | Texts | Families | Human | AI |
|---|---|---|---|---|
| Train | 7,840 | 1,491 | 2,645 | 5,195 |
| Validation | 1,894 | 210 | 522 | 1,372 |
| Detector-test (held-out Wikipedia articles) | 342 | 35 | 205 | 137 |

- Human text: 3,119 paragraphs from 528 pre-2022 Wikipedia revisions (Level-4 vital articles) plus the master's
  Gutenberg sources.
- AI text: 1,578 topic- and length-matched paragraphs (526 each from `llama3`, `qwen2.5:14b`, Qwen3-1.7B base),
  521 Rewriter versions, and the master's AI sources and Humanizer B outputs under the plan's caps.
- Joint-holdout overlap check: 0 matches in 5 files.
- Training: 1,470 updates, 1 h 56 min. Validation AUC by epoch (diagnostic only): 0.9833, 0.9923, 0.9946.

## Threshold

0.97099, the score that 26 of 522 validation human texts (5%) exceed. Used unchanged everywhere below.

## Results against the frozen verdict

| # | Condition | Required | Observed | |
|---|---|---|---|---|
| 1a | Joint holdout, human flagged | ≤ 7 of 75 | 16 of 75 | fail |
| 1b | Joint holdout, AI caught | ≥ 60 of 75 | 56 of 75 | fail |
| 2a | Humanizer B rewrites caught | ≥ 15 of 30 | 30 of 30 | pass |
| 2b | Rewriter greedy rewrites caught | ≥ 20 of 30 | 30 of 30 | pass |
| 3 | Held-out Wikipedia human flagged | ≤ 10% | 11 of 205 (5.4%) | pass |
| 4 | arXiv human abstracts flagged | ≤ 15% | 171 of 188 (91.0%) | fail |

All four are required, so the verdict is negative.

### Joint holdout against Detector A

| | AUC | Human flagged | AI caught |
|---|---|---|---|
| Pilot detector | 0.8661 | 16 of 75 | 56 of 75 |
| Detector A (score > 0) | 0.9588 | 2 of 75 | 70 of 75 |

The pilot is worse than Detector A on both sides. It catches 3 of the 5 AI texts Detector A misses. Human false
flags are concentrated in computing (6) and survey methods (3).

### In-distribution

- Detector-test AUC 0.9988; raw matched AI 104 of 104 caught, Rewriter versions 32 of 33.
- Validation: Humanizer B rewrites of AI text 336 of 336, Rewriter 61 of 64, base-Qwen3 rewrites 139 of 148,
  Humanizer B rewrites of human text 79 of 109.

### Humanized set

| Kind | Pilot detector | Detector A (score > 0) |
|---|---|---|
| AI originals | 30 of 30 | 22 of 30 |
| Humanizer B rewrites | 30 of 30 | 4 of 30 |
| Rewriter greedy | 30 of 30 | 11 of 30 |
| Rewriter best-of-4 picks | 28 of 28 | not scored |

## Reading the result

**The condition-2 pass carries no evidence.** The humanized set is academic-register text, and the model flags
academic-register text whoever wrote it: human arXiv abstracts have a median score of 0.9996. Catching 30 of 30
rewrites there is what a model that flags everything academic would also do.

**What the model learned.** Human training text was almost entirely Wikipedia prose (2,422 of 2,645) with some
nineteenth-century books. The model learned "encyclopedia or old-book style is human, everything else is AI".
That explains the near-perfect held-out Wikipedia numbers and the collapse on arXiv. Topic and length matching
on the Wikipedia side was not enough, because the other 3,588 AI training texts came from different registers
with no human counterpart.

**The joint holdout confirms it is not only a register problem.** Its human side is also pre-2022 Wikipedia,
yet 16 of 75 are flagged, on applied topics the vital-article list does not cover. Its AI side comes from a
generator the model never saw (`claude-opus-5-5`), and 19 of 75 get through.

Scores are saturated: nearly every text scores below 0.01 or above 0.99, so the threshold choice barely matters
and no other threshold would rescue the result.

### Post-hoc diagnostics (not part of the verdict)

Computed from the saved score files after the verdict was known; none of these was planned.

- Only 14 of 188 arXiv abstracts score below 0.5.
- Joint holdout at 0.5 instead of the fixed threshold: 17 human flagged, 56 AI caught. No improvement.
- Ranking humanized-set texts against arXiv abstracts gives AUC 0.97, but the gap is in the fourth decimal place
  (0.9998 against 0.9996). There may be a weak residual signal inside one register; it is not usable as is and
  was not tested on held-out data.

## What this does and does not show

- It does show that a Detector trained on the data available today does not generalise to a new human source or
  a new generator, and is worse than Detector A on the joint holdout.
- It does not show that a general Detector is impossible. The failure has a specific cause, human text from too
  few registers, and one seed of one recipe was run.
- The humanized-text question that motivated the pilot (can a Detector catch what Detector A misses?) is
  **unanswered**, not answered yes: the test for it was confounded by register.

## If this is ever repeated

The change that matters is data, not model: human text from many registers (academic abstracts, news, essays,
forums, student writing), each with AI text generated in the same register on the same topic, and at least one
generator and one human source held out entirely. The arXiv set and the joint holdout have now been used once for
this model; a new attempt needs a fresh unseen human source to stay honest.

## Files

- `plan.json`: frozen plan and verdict rule
- `results.json`, `results-holdout.json`: the numbers above
- the scripts that fetched, generated, built, trained and evaluated, in pipeline order: `fetch_wiki_human.py`, `fetch_arxiv_human.py`, `gen_ai_matched.py` (driven by `run_generation.sh`), then `build_dataset.py`, `train_detector.py` and `evaluate.py` (driven by `run_pipeline.sh`)

The datasets, the model weights and the per-text scores are not published. Commercial service names are replaced by aliases
(Detector A, Humanizer B) in this public copy, so the sha256 values quoted above refer to the private originals.
