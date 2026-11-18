# Rewrite Lab

A solo research project on AI-generated text, run entirely on a laptop (Apple M4, 24 GB): a small model that rewrites
AI-written paragraphs, models that predict a commercial detector's score, and a pilot detector trained on who really
wrote a text. Each experiment had its success criteria written down before it ran, and the results are reported as
they came out, including the negative ones.

Commercial services are referred to by aliases: **Detector A** is a commercial AI-content detector and
**Humanizer B** is a commercial "AI humanizer" rewriting service.

## What's here

| Folder | What it is |
|---|---|
| [`rewriter/`](rewriter/) | The Rewriter: Qwen3-1.7B (4-bit) fine-tuned with LoRA in MLX to rewrite academic paragraphs, plus its evaluation tools |
| [`scorer/`](scorer/) | Encoders (BERT-base, DeBERTa-v3-base) fine-tuned to predict Detector A's 0–100% score from text alone |
| [`joint-holdout/`](joint-holdout/) | A frozen 150-paragraph test set (75 pre-2022 human, 75 AI) that no model trains or selects on |
| [`benchmark/`](benchmark/) | The Rewriter against Humanizer B on 30 matched paragraphs |
| [`detector-pilot/`](detector-pilot/) | A one-day pilot detector trained on true origin, with its frozen plan and result |
| [`docs/images/`](docs/images/) | Training-pipeline diagrams |

Datasets and model weights are not published. The code shows the method; running it needs your own data in the
layout the scripts expect, in the folder named by the `DATA_ROOT` environment variable.

## Results in brief

**Rewriter** ([pipeline](docs/images/model-1-training-pipeline.svg), [benchmark](benchmark/README.md)).
Qwen3-1.7B, LoRA rank 8 on 8 layers, trained on several hundred original-and-rewrite paragraph pairs with
family-grouped splits so near-duplicate texts never straddle train and validation. On 30 unseen AI paragraphs it cut
Detector A's mean score from 68.6% to 29.9% (Humanizer B: 5.1%). Rewrites that reached 0% with no meaning error:
10 of 30, against 18 of 30 for the commercial tool.

**Score predictor** ([pipeline](docs/images/model-2-training-pipeline.svg)).
A regression head on a fine-tuned encoder, trained on about 6,000 texts with Detector A labels. Three seeds per
encoder with identical budgets: mean absolute error 23.9 percentage points for BERT-base and 23.1 for DeBERTa-v3-base.
Neither passed the pre-set promotion gate against the existing model, so production stayed unchanged. Scores
cluster at 0% and 100%, which makes an average error of this size less useful than it sounds. On the joint holdout,
ranked by true origin, the best scorer reaches AUC 0.974, slightly above Detector A itself (0.959).

**Detector pilot** ([report](detector-pilot/report.md)). **Negative.** A DeBERTa classifier trained on true origin:
3,119 pre-2022 Wikipedia paragraphs as human text, topic- and length-matched paragraphs from three local LLMs, and
rewritten AI text. It scored AUC 0.999 on held-out Wikipedia articles, then flagged 91% of human-written arXiv
abstracts and did worse than Detector A on the joint holdout (AUC 0.866 against 0.959). It had learned
"encyclopedia style is human", not "AI wrote this". The fix is data from many registers, not a different model.

## Practices this project follows

- **Plans before results.** Each experiment writes a `plan.json` with its data rules, budget and pass/fail criteria
  before training starts, and the verdict is read against it unchanged. See
  [`detector-pilot/plan.json`](detector-pilot/plan.json).
- **No checkpoint picking on test data.** A fixed training budget, with validation used for diagnostics only.
- **Family-grouped splits.** A source text and every rewrite of it stay in one split.
- **A frozen holdout, used once per candidate.** `joint-holdout/verify.py` checks every training export for overlap.
- **Negative results are written up as carefully as positive ones.**

## Stack

Python, MLX and `mlx-lm` (LoRA fine-tuning on Apple Silicon), PyTorch on MPS, Hugging Face Transformers,
Weights & Biases, Ollama for local text generation, Wikipedia and arXiv APIs for human text.

## Tests

```bash
cd rewriter && python -m unittest test_ai_content_eval test_rewrite_quality_eval
```

```bash
cd joint-holdout && python -m unittest test_evaluate_joint_holdout
```

Tests that need the private data skip themselves when it is absent.

## Purpose

This is research into how AI-text detectors respond to rewriting, and whether a detector can be trained to see through
it. The detector pilot is the part aimed at the detection side.
