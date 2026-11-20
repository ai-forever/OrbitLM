"""On-policy distillation of OrbitLM from a Qwen3-family teacher, on verl 0.9.1's built-in distillation.

verl does the distillation itself: the student samples, a teacher served by vLLM on GPUs of its own
scores every token of each sample, and the per-token advantage is log p_teacher - log p_student
(``distillation.enabled``, ``distillation.distillation_loss.loss_mode=k1``,
``use_policy_gradient=true``). This package holds what our stack adds around it.

Modules:
- ``preflight``: the checks that the launcher runs before a distillation run starts. They refuse a
  teacher that is not a local directory, a teacher whose tokenizer gives our token ids other
  meanings, a sampling temperature other than 1.0, a policy loss that cannot run under
  distillation, and teacher context lengths that verl or vLLM would refuse later.
- ``teacher``: verl's agent-loop worker and manager with one change. The teacher scores the
  student's sequence without the leading ``<|endoftext|>`` of our prompts, and its rows are
  shifted back so that they line up with the student's tokens.

The launcher's side lives in ``orbit_verl.launch.config.finish_distillation``. It splits the job's
GPUs between the student and the teacher and fills the teacher's ``max_model_len``.

This file imports nothing, so importing one module of the package stays as cheap as that module.
"""
