# Joint holdout, November 6, 2026

150 paragraphs for the final test of Model 1 and Model 2. **Keep them out of training and out of checkpoint selection for both models, and don't add them to `humanizer-master.json`.** Exports are built from the master, so anything added there could reach training.

Use this set once per finished candidate. Don't tune against it, or it stops being a fair test.

**This public copy has the code and the description only.** The paragraphs and scores are kept private; the scripts read them from the folder named by `DATA_ROOT`.

## Contents

- **150 paragraphs, 10 domains, 15 per domain:** survey methods, operations, labor economics, public economics, agriculture, public health, law and governance, history, psychology and education, computing.
- **75 human-written:** from Wikipedia revisions saved on or before December 31, 2021, before ChatGPT's public release. The origin label is `human_likely_pre_2022` because authorship isn't verified. The text is CC BY-SA 3.0; each row keeps its page, revision ID and URL, so keep that attribution if any of it is shared.
- **75 AI-written:** written by `claude-opus-5-5` for this set, with the prompt recorded in each row.
- **One paragraph per topic,** so all 150 count as separate source families.
- **Detector A scores:** all 150 scanned on November 10, 2026, in `detector-a-scores.json` (IDs, hashes and scores, no text). Human half: 73 at 0%, one at 36%, one at 100% (`jh-023`), mean 1.8%. AI half: 63 at 100%, 7 between 36% and 63%, 5 at 0%, mean 88.8%.

## Independence checks

- Screened against 10,119 project texts: every master source and output, all model exports including the October 29 Model 2 export, and the 36-paragraph benchmark with its outputs.
- A paragraph is rejected on an exact normalized match, any shared 10-word passage, or word-sequence similarity of 0.65 or more. None was rejected, and the highest similarity found was 0.2252.
- As a positive control, the screen rejected 40 of 40 sampled corpus texts, both unchanged and with 10% or 25% of their words replaced.

## Limits

- The screen compares wording. A heavy paraphrase of a project text, or a different paragraph on a similar topic, could still pass.
- The base Qwen3 model may have seen Wikipedia during its original training. That can't be checked from here.
- The AI half comes from one generator and has a narrower length range (84–98 words, against 64–150 for the human half). A model could tell the two groups apart by length, so report results by origin and don't rely only on the pooled number.
- The selection rule dropped human paragraphs that depend on earlier text, open with a quotation, or contain markup residue. Other imperfections were kept, because real input has them.
- The set has no claim-by-claim preservation checklists yet. Those are needed before it can measure meaning errors the way the 36-paragraph benchmark does.

## Files

- `holdout.json`: the frozen rows.
- `detector-a-scores.json`: official Detector A scores for each row, by `holdout_id` and `text_sha256`.
- `manifest.json`: counts, screen results and the hash of `holdout.json`.
- `verify.py`: run `python verify.py <export or file>` to confirm that an export shares no text with this set. Run it on every new training export.
- `overlap-rejected.json`: rejected candidates (empty).
- `build_holdout.py`, `claude_paragraphs.json`, `wiki_candidates.json`, `wiki_fetch.py`: how the set was built. `build_holdout.py` refuses to overwrite an existing folder.
