"""The teacher's view of a distillation sample: the student's sequence without the leading ``<|endoftext|>``.

Why. verl's agent-loop worker sends the teacher exactly the student's prompt ids followed by the
response ids. Our SFT prompts start with ``<|endoftext|>`` (id 151643), because packing put that
token before most SFT dialogs, while a Qwen3 teacher's own chat template starts with
``<|im_start|>``. Measured on CPU, the extra token moves Qwen3-8B's per-token log-probs by 0.3-1.1
nats. That is as large as the student-teacher differences that distillation trains on, so the
teacher must score the sequence without it.

What changes. ``agent_loop_worker_class`` builds a subclass of verl's ``AgentLoopWorkerTQ`` that
overrides ``AgentLoopWorker._compute_teacher_logprobs``
(``verl/experimental/agent_loop/agent_loop.py``). The override strips the prefix from the prompt
and lets verl's own method send the shorter sequence to the teacher. It then puts dummy rows in
front of the teacher's answer, one for every stripped token. Routing, sampling parameters and the
teacher client stay verl's.

An answer whose realigned rows do not line up with the student's tokens fails its sample. verl
would only drop that sample and train on, so the worker also keeps the first such failure, and
its ``generate_sequences`` override raises it at the next dispatch. That stops the run within one
step.

Row convention. verl stores the teacher's answer as one row per token of the sequence. Row i holds
the teacher's log-prob of token i+1. The estimator losses such as k1 and k3 get one column
(``prompt_logprobs=0``), and ``forward_kl_topk`` gets the top-k ids and log-probs. The first token
has no row of its own, because nothing predicts it. verl appends a dummy row (id 0, log-prob 0.0)
at the end, so that the row count equals the token count (``extract_prompt_logprobs`` in
``verl/workers/rollout/vllm_rollout/utils.py``). The trainer reads rows prompt_len-1 to L-2 for the
L-prompt_len response tokens (``no_padding_2_padding`` in ``verl/workers/utils/padding.py``).

When the teacher sees a sequence that is n tokens shorter, its row j is the log-prob of student
token j+n+1. Putting n dummy rows of the same kind in front makes row i mean "token i+1 of the
student sequence" again, and the row count equals the student's token count again. The dummy rows
sit at prompt positions, which the loss never reads. A prefix is stripped only when at least one
prompt token is left, so the first response token always has a real teacher row.

How to switch it on. Name the manager in the config:
``actor_rollout_ref.rollout.agent.agent_loop_manager_class=orbit_verl.distill.teacher.OrbitLMAgentLoopManagerTQ``.
verl's ``TaskRunnerV1.init_agent_loop_manager`` loads that class by name. It is verl's
``AgentLoopManagerTQ`` with our worker class. verl's ``AgentLoopConfig`` has the key, but its
``rollout.yaml`` does not, so the key goes into the experiment YAML, or after a ``+`` on the command
line. ``orbit.distill.teacher_strip_prefix`` holds the prefix. A missing key or null means
``[151643]``, and ``[]`` turns stripping off. A prompt that does not start with exactly the prefix
goes to the teacher unchanged. ``teacher_prompt_problems`` is a pre-flight check for the launcher.

The student-only ids 151669-151680 are not masked. The teacher's output head has rows for those
ids but never trained them, so it gives them a very low log-prob. When the student samples one of
them, k1 = log p_student - log p_teacher is large and positive, and the advantage is its negative.
``distillation_loss.loss_max_clamp`` (10 in our config) bounds that penalty. The student is
therefore pushed away from emitting these ids. That is the intended effect, because chat, math and
code answers should not contain them.

verl never calls the teacher during validation, so validation is unchanged.

This module imports nothing from verl, torch or Ray when it is imported. The helpers can then be
tested on any python, and the verl classes are built on first use. ``OrbitLMAgentLoopManagerTQ``
is reached through the module's ``__getattr__`` for that reason.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# The launcher's config reader. It reads a DictConfig without importing omegaconf, and it returns
# the default for a key that a struct config does not have, instead of raising.
from orbit_verl.launch.env import _cfg_get

logger = logging.getLogger(__name__)

STRIP_PREFIX_KEY = "orbit.distill.teacher_strip_prefix"
# <|endoftext|> of the Qwen3 tokenizer, which the student and every Qwen3 teacher share.
DEFAULT_STRIP_PREFIX: tuple[int, ...] = (151643,)

MANAGER_CLASS_KEY = "actor_rollout_ref.rollout.agent.agent_loop_manager_class"
MANAGER_CLASS_NAME = "OrbitLMAgentLoopManagerTQ"
MANAGER_CLASS_FQN = f"orbit_verl.distill.teacher.{MANAGER_CLASS_NAME}"

# The values of verl's own dummy row at the end of the teacher's answer. The rows put in front of
# it use the same values, so every row the loss never reads looks the same.
DUMMY_TEACHER_ID = 0
DUMMY_TEACHER_LOGPROB = 0.0


def check_strip_prefix(value: Any) -> tuple[int, ...]:
    """Return ``value`` as a tuple of token ids, or raise ValueError when it is not a list of ids.

    A bare number or a string is refused instead of being read as a one-token prefix. Either one is
    more likely a typo than an intent, and a wrong prefix would quietly change what the teacher
    sees for every sample.
    """
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise ValueError(f"{STRIP_PREFIX_KEY} must be a list of token ids, such as [151643], got {value!r}.")
    ids = []
    for item in value:
        # A bool is an int in Python, but true or false in this list is a typo, not a token id.
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{STRIP_PREFIX_KEY} must hold non-negative token ids, got {item!r} in {value!r}.")
        ids.append(int(item))
    return tuple(ids)


def teacher_strip_prefix(config: Any) -> tuple[int, ...]:
    """The token ids to strip from the start of every teacher prompt, read from the whole run config.

    A missing key or null gives ``DEFAULT_STRIP_PREFIX``. An empty list turns stripping off.
    """
    value = _cfg_get(config, STRIP_PREFIX_KEY)
    return DEFAULT_STRIP_PREFIX if value is None else check_strip_prefix(value)


def stripped_prefix_length(prompt_ids: Sequence[int], prefix: Sequence[int]) -> int:
    """How many leading prompt tokens the teacher does not see: ``len(prefix)`` or 0.

    The prefix is stripped once, and only when the prompt starts with exactly these ids. A prompt
    that is nothing but the prefix keeps it, because the teacher needs at least one prompt token.
    Without one, the first response token would get a dummy row, and its log-prob of 0.0 would
    tell the loss that the teacher was certain of that token.
    """
    n = len(prefix)
    if n == 0 or len(prompt_ids) <= n:
        return 0
    return n if [int(t) for t in prompt_ids[:n]] == [int(t) for t in prefix] else 0


def realign_teacher_rows(teacher_ids: Any, teacher_logprobs: Any, n_stripped: int) -> tuple[Any, Any]:
    """Put ``n_stripped`` dummy rows in front of the teacher's ``[S, K]`` ids and log-probs.

    After this, row i is the teacher's view of token i+1 of the student sequence, and there are
    ``S + n_stripped`` rows, one per student token. The padding is the same ``F.pad`` call that
    verl's ``_pad_teacher_outputs`` uses, so the dtypes stay those of the teacher client (int32 ids,
    float32 log-probs). With nothing stripped, the tensors come back unchanged.
    """
    if n_stripped < 0:
        raise ValueError(f"n_stripped must be at least 0, got {n_stripped}.")
    if teacher_ids.dim() != 2 or tuple(teacher_ids.shape) != tuple(teacher_logprobs.shape):
        raise ValueError(
            f"Expected teacher ids and log-probs of the same shape [S, K], got {tuple(teacher_ids.shape)} "
            f"and {tuple(teacher_logprobs.shape)}."
        )
    if n_stripped == 0:
        return teacher_ids, teacher_logprobs
    from torch.nn import functional as F

    # F.pad lists the last dimension first: no columns are added, and n_stripped rows go in front.
    padding = (0, 0, n_stripped, 0)
    return (
        F.pad(teacher_ids, padding, value=DUMMY_TEACHER_ID),
        F.pad(teacher_logprobs, padding, value=DUMMY_TEACHER_LOGPROB),
    )


def check_teacher_rows(
    row_count: int,
    sequence_ids: Sequence[int],
    n_stripped: int,
    row_token_ids: Sequence[int] | None = None,
) -> None:
    """Raise RuntimeError unless the realigned teacher rows line up with the student sequence.

    There must be one row per student token. ``row_token_ids`` is the first id column of the rows.
    Pass it only when the teacher returned the log-prob of the token that is in the sequence
    (``prompt_logprobs=0``, every loss except top-k). The id in row i must then be token i+1 of the
    student sequence, for every row the teacher filled: rows n_stripped to L-2. A mismatch would
    train each token on another token's teacher log-prob, so the sample fails instead.
    """
    length = len(sequence_ids)
    if row_count != length:
        raise RuntimeError(
            f"The teacher returned {row_count} rows after realignment for a sequence of {length} tokens "
            f"({n_stripped} stripped). verl needs exactly one row per token."
        )
    if row_token_ids is None:
        return
    got = [int(t) for t in row_token_ids[n_stripped:length - 1]]
    want = [int(t) for t in sequence_ids[n_stripped + 1:]]
    if got != want:
        first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b) + n_stripped
        raise RuntimeError(
            f"Teacher row {first} holds token {got[first - n_stripped]}, but token {first + 1} of the student "
            f"sequence is {want[first - n_stripped]} ({n_stripped} prompt tokens stripped). The teacher's "
            "log-probs are not aligned with the student's tokens."
        )


def teacher_prompt_problems(config: Any) -> list[str]:
    """Pre-flight problems of the teacher prompt fix in a composed run config; empty when it is fine.

    Without our manager class, verl's stock worker sends the teacher the leading ``<|endoftext|>``,
    and every teacher log-prob is shifted. Nothing would fail, so the launcher has to ask.
    """
    if not bool(_cfg_get(config, "distillation.enabled", False)):
        return []
    problems = []
    manager = _cfg_get(config, MANAGER_CLASS_KEY)
    if manager != MANAGER_CLASS_FQN:
        problems.append(
            f"distillation.enabled=true needs {MANAGER_CLASS_KEY}={MANAGER_CLASS_FQN}, got {manager!r}. "
            "Without it the teacher scores the prompt with its leading <|endoftext|>."
        )
    try:
        teacher_strip_prefix(config)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


# The verl classes, built once per process by the two functions below.
_WORKER_CLASS: Any = None
_MANAGER_CLASS: Any = None


def worker_subclass(stock: type) -> type:
    """A subclass of the plain agent-loop worker class ``stock`` with the teacher prompt fix.

    ``agent_loop_worker_class`` passes verl's ``AgentLoopWorkerTQ``. The tests pass a stand-in
    class, so that the overrides can be checked without verl or Ray.
    """

    class OrbitLMAgentLoopWorkerTQ(stock):
        """verl's TransferQueue agent-loop worker, with the teacher prompt fix of this module.

        It also stops the run once a teacher answer fails the alignment check (``generate_sequences``).
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Read here, so that a bad value stops the run when the workers start.
            self.teacher_strip_prefix = teacher_strip_prefix(self.config)
            self._warned_prompt_without_prefix = False
            self._teacher_alignment_error: RuntimeError | None = None

        async def generate_sequences(self, batch: Any) -> None:
            """verl's dispatch of a batch, refused once a teacher answer has failed the alignment check.

            The check raises inside a sample's background task. verl catches that exception, marks
            the sample's prompt group as failed, logs it, and trains on without the group. Misaligned
            answers would then vanish from training, with only an error line in the log to show it.
            verl's manager waits for this method on every worker with ``ray.get``, so raising here
            stops the task runner at the next dispatch, within one step of the first misaligned answer.
            """
            if self._teacher_alignment_error is not None:
                raise RuntimeError(
                    "Stopping the run: a teacher answer failed the alignment check earlier in this worker. "
                    f"The first failure was: {self._teacher_alignment_error}"
                ) from self._teacher_alignment_error
            await super().generate_sequences(batch)

        async def _compute_teacher_logprobs(
            self,
            output: Any,
            prompt_ids: list[int],
            response_ids: list[int],
            validate: bool,
            sample_kwargs: dict[str, Any] | None = None,
        ) -> None:
            """verl's teacher call on the prompt without the prefix, with the rows realigned to the student."""
            if validate or not self.distillation_enabled:
                await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
                return

            n_stripped = stripped_prefix_length(prompt_ids, self.teacher_strip_prefix)
            if n_stripped == 0 and self.teacher_strip_prefix and not self._warned_prompt_without_prefix:
                # Chat-template prompts have no <|endoftext|>, so this is not an error. Saying it once
                # per worker still shows in the logs which prompts the teacher sees unchanged.
                self._warned_prompt_without_prefix = True
                logger.warning(
                    "A prompt does not start with %s=%s, so the teacher gets it unchanged. This is "
                    "logged once per agent-loop worker.",
                    STRIP_PREFIX_KEY,
                    list(self.teacher_strip_prefix),
                )
            await super()._compute_teacher_logprobs(
                output, list(prompt_ids[n_stripped:]), response_ids, validate, sample_kwargs
            )

            fields = output.extra_fields
            if "teacher_ids" not in fields or "teacher_logprobs" not in fields:
                return
            teacher_ids, teacher_logprobs = realign_teacher_rows(
                fields["teacher_ids"], fields["teacher_logprobs"], n_stripped
            )
            use_topk = self.teacher_server_manager.distillation_loss_config.loss_settings.use_topk
            # Top-k rows list the teacher's best tokens, not the token in the sequence, so only the
            # row count can be checked for them.
            try:
                check_teacher_rows(
                    teacher_ids.shape[0],
                    list(prompt_ids) + list(response_ids),
                    n_stripped,
                    row_token_ids=None if use_topk else teacher_ids[:, 0].tolist(),
                )
            except RuntimeError as exc:
                # The sample still fails here as before. The first failure is also kept, so that
                # generate_sequences can stop the whole run with it.
                if self._teacher_alignment_error is None:
                    self._teacher_alignment_error = exc
                raise
            fields["teacher_ids"], fields["teacher_logprobs"] = teacher_ids, teacher_logprobs

    return OrbitLMAgentLoopWorkerTQ


def agent_loop_worker_class() -> Any:
    """verl's ``AgentLoopWorkerTQ`` as a Ray actor class whose teacher sees the prompt without the prefix.

    verl defines ``AgentLoopWorkerTQ`` with ``@ray.remote``, so the module attribute is a Ray actor
    class, and Ray refuses to subclass one. The subclass is made from the plain class that Ray keeps
    in the actor class's metadata, as ``orbit_verl.launch.train.task_runner_class`` does for the
    task runner.
    """
    global _WORKER_CLASS
    if _WORKER_CLASS is not None:
        return _WORKER_CLASS

    import ray
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

    stock = AgentLoopWorkerTQ.__ray_metadata__.modified_class.__ray_actor_class__
    _WORKER_CLASS = ray.remote(worker_subclass(stock))
    return _WORKER_CLASS


def agent_loop_manager_class() -> type:
    """verl's ``AgentLoopManagerTQ`` with the worker class of ``agent_loop_worker_class``.

    The manager starts the agent-loop workers when verl's task runner calls ``create``.
    """
    global _MANAGER_CLASS
    if _MANAGER_CLASS is not None:
        return _MANAGER_CLASS

    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

    class OrbitLMAgentLoopManagerTQ(AgentLoopManagerTQ):
        """verl's TransferQueue agent-loop manager, whose workers apply the teacher prompt fix."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # AgentLoopManagerTQ sets verl's worker class before it calls the base __init__, and the
            # workers are started only later, in create(). Replacing the class here is therefore
            # both late enough to win and early enough to be used.
            self.agent_loop_workers_class = agent_loop_worker_class()

    # The class is reachable as a module attribute through __getattr__ below. Its qualified name
    # says so, which makes pickle and repr name it the way the config does.
    OrbitLMAgentLoopManagerTQ.__qualname__ = MANAGER_CLASS_NAME
    _MANAGER_CLASS = OrbitLMAgentLoopManagerTQ
    return _MANAGER_CLASS


def __getattr__(name: str) -> Any:
    """Build ``OrbitLMAgentLoopManagerTQ`` when verl loads it by name (PEP 562 module attribute).

    verl's ``load_class_from_fqn`` imports this module and reads the class as an attribute. Building
    it here instead of at import keeps verl out of the import of this module.
    """
    if name == MANAGER_CLASS_NAME:
        return agent_loop_manager_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
