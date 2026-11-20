# Stored evaluation outputs

Outputs of the runs reported in `report/main.pdf`, copied from the experiment machine. Read them with
`report/analysis.py` or the notebook `notebook/CE893_evaluation.ipynb`.

- `runs/test_*.jsonl.gz`: one JSON line per test question (332 BIRD dev questions on formula_1 and
  student_club) per arm: question, hint, gold SQL, predicted Cube query, `correct` (strict),
  `correct_relaxed`, steps, prompt/completion tokens, model latency and the full tool trace.
  `test_lora_on_lean_prompt_14b_partial` was stopped after 127 questions (0 correct).
- `runs/tune120_*`, `runs/tune363_*`: harness development runs on the tuning databases
  (superhero, toxicology, california_schools); no test question is in them.
- `training/*.json`: LoRA training configuration and Trainer log history (training loss every 10
  optimiser steps, validation loss every half epoch), plus `eval_loss_before_training`.
