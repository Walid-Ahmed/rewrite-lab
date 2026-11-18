#!/bin/zsh
# Keep writing matched AI paragraphs while the Wikipedia fetch adds articles; stop once the fetch is done and nothing is pending.
# One generator at a time, so the models do not compete for memory.
cd "${0:A:h}"
ROOT=${DATA_ROOT:?set DATA_ROOT to the private project folder}
while true; do
  fetching=$(pgrep -f fetch_wiki_human.py | wc -l | tr -d ' ')
  total=0
  for g in llama3 qwen2.5:14b; do
    n=$($ROOT/.venv-score/bin/python gen_ai_matched.py --generator $g | tee -a generation.log | tail -1 | sed 's/[^0-9]//g')
    total=$((total + ${n:-0}))
  done
  n=$($ROOT/.venv-mlx/bin/python gen_ai_matched.py --generator qwen3-1.7b 2>>generation.err | tee -a generation.log | tail -1 | sed 's/[^0-9]//g')
  total=$((total + ${n:-0}))
  n=$($ROOT/.venv-mlx/bin/python gen_ai_matched.py --rewrite 2>>generation.err | tee -a generation.log | tail -1 | sed 's/[^0-9]//g')
  total=$((total + ${n:-0}))
  echo "$(date +%T) pass wrote $total; fetch running: $fetching" >> generation.log
  [[ $fetching -eq 0 && $total -eq 0 ]] && break
  [[ $total -eq 0 ]] && sleep 60
done
echo "GENERATION COMPLETE" >> generation.log
