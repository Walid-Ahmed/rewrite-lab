# Rewriter against a commercial humanizer, November 2026

A matched comparison on 30 AI-written academic paragraphs, each from a different source family that neither model
trained on. Every paragraph was rewritten by Humanizer B (a commercial humanizer) and by the Rewriter (greedy decoding,
plus a best-of-4 variant that picks the lowest-scoring of four samples). All 90 originals and rewrites were scored by
Detector A (a commercial AI detector), and every rewrite was checked by hand against the original for material meaning
errors: a changed fact, number, claim or qualification.

## Results

| | Original | Humanizer B | Rewriter (greedy) | Rewriter (best of 4) |
|---|---|---|---|---|
| Mean Detector A score | 68.6% | 5.1% | 29.9% | – |
| Rewrites scoring 0% | 8 of 30 | 26 of 30 | 19 of 30 | – |
| Rewrites with a material meaning error | – | 10 of 30 | 12 of 30 | 21 of 30 |
| **0% and no material meaning error** | – | **18 of 30** | **10 of 30** | **9 of 30** |

## Reading it

- The Rewriter (Qwen3-1.7B with a LoRA adapter, running on a laptop) gets about halfway from the original to the
  commercial tool on detector score, and makes a similar number of meaning errors.
- On the measure that matters, text that both passes and still says the same thing, the commercial tool is ahead,
  18 to 10.
- Picking the lowest-scoring of four samples does not help. It mostly selects the rewrites that drifted furthest from
  the original, so meaning errors nearly double.
- Both tools get a low detector score partly by changing meaning. A detector score alone is not a quality measure.

## Limits

30 paragraphs, one domain (academic prose), one detector, one reviewer. The texts and per-paragraph scores are not
published.
