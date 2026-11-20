"""Safe executor for the reward of code RL: a static AST policy plus a sandbox on the same machine.

    from orbit_verl.code_sandbox import execute, Limits, ExecutorPool
    result = execute(program, [{"input": "1 2\n", "output": "3\n"}], kind="stdin")
    reward = 1.0 if result["passed"] else 0.0

See README.md for the design and the threat model; rules.py holds the policy tables.
"""

from .executor import (
    REASONS, REWARD_INFO_KEYS, ExecutorPool, Limits, available_cpus, execute, extract_code, judge_prelude,
    outputs_match, reward_info, self_check,
)
from .policy import check_program

__all__ = ["execute", "Limits", "ExecutorPool", "self_check", "available_cpus", "check_program", "judge_prelude",
           "outputs_match", "extract_code", "reward_info", "REWARD_INFO_KEYS", "REASONS"]
