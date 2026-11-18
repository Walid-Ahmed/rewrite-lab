#!/bin/zsh
# Detector pilot, end to end, in the order fixed by plan.json: wait for the fetch and generation, build the dataset,
# train one seed, evaluate steps 1-4, then score the joint holdout once. Stops at the first failure.
cd "${0:A:h}"
ROOT=${DATA_ROOT:?set DATA_ROOT to the private project folder}
PY=$ROOT/.venv-score/bin/python
log() { echo "$(date +%FT%T) $*" >> pipeline.log }
fail() { log "PIPELINE FAILED at $1"; exit 1 }

log "waiting for fetch and generation"
until grep -q "GENERATION COMPLETE" generation.log 2>/dev/null; do sleep 30; done
grep -q "^DONE articles" fetch.log || fail "fetch did not finish"
ollama stop llama3 2>/dev/null; ollama stop qwen2.5:14b 2>/dev/null   # free memory for training; only the models this run loaded

log "building dataset"
python3 build_dataset.py > build.log 2>&1 || fail build
log "training"
$PY train_detector.py > training.log 2>&1 || fail training
log "evaluating steps 1-4"
$PY evaluate.py > evaluate.log 2>&1 || fail evaluate
log "scoring joint holdout once"
$PY evaluate.py --holdout > evaluate-holdout.log 2>&1 || fail holdout
log "PIPELINE COMPLETE"
