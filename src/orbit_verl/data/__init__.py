"""Prompts, datasets, the rollout agent loop and the reward function for OrbitLM RL on verl.

- ``prompts``: the prompt-mode registry, prompt rendering and prompt token ids.
- ``prepare_data``: converts a jsonl or parquet source into verl's train/val parquet files.
- ``agent_loop``: ``OrbitLMAgentLoop``, which tokenizes raw prompts, records how each response ended
  and applies the thinking budget and the no-think cap.
- ``reward``: ``compute_score``, which routes to the MATH check (``math_answer``) and the code sandbox.
- ``distill_data``: builds the datasets of both recipes, with a prompt mode per row.
- ``code_data``: builds the code problems and their tests.

This file stays free of imports. Ray workers import ``orbit_verl.data.prompts`` and
``orbit_verl.data.reward`` once per process, and anything imported here would load with them.
"""
