"""CPU tests for the teacher prompt fix of on-policy distillation (``orbit_verl.distill.teacher``).

The helpers are checked on plain lists and on small synthetic tensors. A made-up teacher answers
the way verl's teacher client does: one row per token, row j for token j+1, and a dummy row at the
end. That fake is checked against verl's own conversion of vLLM's prompt log-probs where vLLM
imports. The worker's override runs verl's own ``_compute_teacher_logprobs`` with a fake teacher
client, without Ray. The stop on a misaligned answer is also checked on a stand-in for verl's
worker class, which needs only torch. The manager is loaded by name the way verl's task runner
loads it.

The tests that need neither torch nor verl import only this module and the standard library, so
they also run on a python without them.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbit_verl.distill import teacher

EOT = 151643           # <|endoftext|>, which our SFT prompts start with
IM_START, IM_END = 151644, 151645
THINK, END_THINK = 151667, 151668
CODE_TOKEN = 151669    # <file_sep>, one of the 12 ids that only the student has

# A short think prompt in the release chat format: <|endoftext|><|im_start|>user ... <|im_end|>\n
# <|im_start|>assistant\n<think>\n. The ids between the special tokens are arbitrary.
PROMPT = [EOT, IM_START, 872, 198, 3838, 374, 220, 17, 10, 17, 30, IM_END, 198, IM_START, 77091, 198, THINK, 198]
RESPONSE = [40, 1184, CODE_TOKEN, 220, 19, 13, END_THINK, 198, 19, IM_END]
STUDENT = PROMPT + RESPONSE


def _torch():
    return pytest.importorskip("torch")


def _teacher_logprob(context, token) -> float:
    """A made-up teacher: the log-prob of ``token`` after ``context``.

    It depends on every token of the context, so a context with or without the leading
    <|endoftext|> gives a different value. The values are multiples of 1/128, which float32 holds
    exactly, so tensors can be compared with ``torch.equal``. Python hashes tuples of ints the same
    way in every process.
    """
    return -0.25 - (hash((tuple(context), int(token))) % 997) / 128.0


def _teacher_answer(sequence, topk=0):
    """What verl's teacher client returns for ``sequence``: ``[S, max(topk, 1)]`` int32 ids and float32 log-probs.

    The first token has no row, row j holds token j+1, and a dummy row of zeros closes the list,
    as in verl's ``extract_prompt_logprobs``. With ``topk`` the row lists k ids, and the token in
    the sequence comes first.
    """
    torch = _torch()
    width = max(topk, 1)
    ids, logprobs = [], []
    for j in range(1, len(sequence)):
        row = [sequence[j]] if topk == 0 else [sequence[j] + r for r in range(topk)]
        ids.append(row)
        logprobs.append([_teacher_logprob(sequence[:j], t) for t in row])
    ids.append([0] * width)
    logprobs.append([0.0] * width)
    return torch.tensor(ids, dtype=torch.int32), torch.tensor(logprobs)


def _student_rows(n_stripped, length):
    """The ids that the realigned first column must hold for a student sequence of ``length`` tokens."""
    return [0] * n_stripped + STUDENT[n_stripped + 1:length] + [0]


# ---------------------------------------------------------------------------------------------
# Plain-python helpers
# ---------------------------------------------------------------------------------------------


def test_the_default_prefix_is_one_endoftext_and_an_empty_list_turns_it_off():
    assert teacher.STRIP_PREFIX_KEY == "orbit.distill.teacher_strip_prefix"
    assert teacher.DEFAULT_STRIP_PREFIX == (EOT,)
    assert teacher.teacher_strip_prefix({}) == (EOT,)
    assert teacher.teacher_strip_prefix({"orbit": {"run_name": "x"}}) == (EOT,)
    assert teacher.teacher_strip_prefix({"orbit": {"distill": {"teacher_strip_prefix": None}}}) == (EOT,)
    assert teacher.teacher_strip_prefix({"orbit": {"distill": {"teacher_strip_prefix": []}}}) == ()
    two = {"orbit": {"distill": {"teacher_strip_prefix": [EOT, IM_START]}}}
    assert teacher.teacher_strip_prefix(two) == (EOT, IM_START)


def test_bad_prefixes_are_refused_with_the_key_name():
    for bad in (EOT, str(EOT), f"[{EOT}]", {"id": EOT}, [True], [-1], [1.5], [str(EOT)], [EOT, None]):
        with pytest.raises(ValueError, match="orbit.distill.teacher_strip_prefix"):
            teacher.teacher_strip_prefix({"orbit": {"distill": {"teacher_strip_prefix": bad}}})


def test_only_the_exact_prefix_is_stripped_and_only_once():
    one = teacher.DEFAULT_STRIP_PREFIX
    assert teacher.stripped_prefix_length(PROMPT, one) == 1
    # A second <|endoftext|> stays: the prefix is stripped once.
    assert teacher.stripped_prefix_length([EOT, EOT, IM_START, 198], one) == 1
    # A prompt that does not start with the prefix, such as a chat-template prompt, is not touched.
    assert teacher.stripped_prefix_length(PROMPT[1:], one) == 0
    assert teacher.stripped_prefix_length([IM_START, EOT, 198], one) == 0
    # The teacher needs at least one prompt token, so a prompt that is only the prefix keeps it.
    assert teacher.stripped_prefix_length([EOT], one) == 0
    assert teacher.stripped_prefix_length([], one) == 0
    # An empty prefix strips nothing.
    assert teacher.stripped_prefix_length(PROMPT, ()) == 0
    # A longer prefix must match as a whole.
    assert teacher.stripped_prefix_length(PROMPT, (EOT, IM_START)) == 2
    assert teacher.stripped_prefix_length([EOT, IM_END, 198], (EOT, IM_START)) == 0
    assert teacher.stripped_prefix_length([EOT, IM_START], (EOT, IM_START)) == 0


def test_the_row_check_accepts_aligned_rows_and_refuses_shifted_ones():
    length = len(STUDENT)
    aligned = _student_rows(1, length)
    teacher.check_teacher_rows(length, STUDENT, 1, aligned)
    teacher.check_teacher_rows(length, STUDENT, 0, _student_rows(0, length))
    # Top-k rows carry no sequence ids, so only their count is checked.
    teacher.check_teacher_rows(length, STUDENT, 1)

    # The teacher's rows without the front dummy row: one row short.
    with pytest.raises(RuntimeError, match="one row per token"):
        teacher.check_teacher_rows(length - 1, STUDENT, 1, aligned[1:])
    # The right count, but every row moved down by one: row i would train token i+1 on the
    # teacher's log-prob of token i.
    shifted = [0, 0] + STUDENT[2:length - 1] + [0]
    with pytest.raises(RuntimeError, match="not aligned"):
        teacher.check_teacher_rows(length, STUDENT, 1, shifted)


def test_the_pre_flight_asks_for_the_manager_class_only_when_distillation_is_on():
    manager = {"agent_loop_manager_class": teacher.MANAGER_CLASS_FQN}
    on_without = {"distillation": {"enabled": True}}
    on_with = {"distillation": {"enabled": True}, "actor_rollout_ref": {"rollout": {"agent": manager}}}
    assert teacher.MANAGER_CLASS_KEY == "actor_rollout_ref.rollout.agent.agent_loop_manager_class"
    assert teacher.MANAGER_CLASS_FQN == "orbit_verl.distill.teacher.OrbitLMAgentLoopManagerTQ"
    assert teacher.teacher_prompt_problems({}) == []
    assert teacher.teacher_prompt_problems({"distillation": {"enabled": False}}) == []
    problems = teacher.teacher_prompt_problems(on_without)
    assert len(problems) == 1 and teacher.MANAGER_CLASS_KEY in problems[0] and "<|endoftext|>" in problems[0]
    assert teacher.teacher_prompt_problems(on_with) == []
    bad_prefix = dict(on_with, orbit={"distill": {"teacher_strip_prefix": EOT}})
    problems = teacher.teacher_prompt_problems(bad_prefix)
    assert len(problems) == 1 and teacher.STRIP_PREFIX_KEY in problems[0]


def test_the_module_imports_without_verl_torch_or_ray():
    # A None entry in sys.modules makes any import of that name fail, so the child proves that
    # importing the module and reading its constants needs none of them.
    code = (
        "import sys\n"
        "for name in ('verl', 'torch', 'ray', 'omegaconf', 'transfer_queue'):\n"
        "    sys.modules[name] = None\n"
        "import orbit_verl.distill.teacher as t\n"
        "print(t.MANAGER_CLASS_FQN, t.stripped_prefix_length([151643, 1], t.DEFAULT_STRIP_PREFIX))\n"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p))
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [teacher.MANAGER_CLASS_FQN, "1"]


def test_unknown_module_attributes_still_raise():
    with pytest.raises(AttributeError, match="NoSuchClass"):
        getattr(teacher, "NoSuchClass")


# ---------------------------------------------------------------------------------------------
# Tensor alignment on synthetic teacher answers (torch)
# ---------------------------------------------------------------------------------------------


def test_realigned_rows_score_each_student_token_in_the_teachers_context():
    torch = _torch()
    n = teacher.stripped_prefix_length(PROMPT, teacher.DEFAULT_STRIP_PREFIX)
    assert n == 1
    ids, logprobs = teacher.realign_teacher_rows(*_teacher_answer(STUDENT[n:]), n)

    length = len(STUDENT)
    assert tuple(ids.shape) == tuple(logprobs.shape) == (length, 1)
    assert ids.dtype == torch.int32 and logprobs.dtype == torch.float32
    # Row 0 is the dummy row put in front, and the last row is verl's own dummy row.
    for row in (0, length - 1):
        assert ids[row, 0].item() == teacher.DUMMY_TEACHER_ID and logprobs[row, 0].item() == 0.0
    # Every other row i is the teacher's log-prob of student token i+1, after the context that the
    # teacher saw, which starts after the stripped <|endoftext|>.
    for i in range(n, length - 1):
        assert ids[i, 0].item() == STUDENT[i + 1]
        assert logprobs[i, 0].item() == _teacher_logprob(STUDENT[n:i + 1], STUDENT[i + 1])
    teacher.check_teacher_rows(ids.shape[0], STUDENT, n, ids[:, 0].tolist())

    # verl's trainer reads rows prompt_len-1 .. L-2 for the response tokens (no_padding_2_padding).
    # Those rows are exactly the teacher's scores of the response.
    start = len(PROMPT) - 1
    response_rows = logprobs[start:start + len(RESPONSE), 0].tolist()
    expected = [_teacher_logprob(STUDENT[n:len(PROMPT) + k], token) for k, token in enumerate(RESPONSE)]
    assert response_rows == expected

    # The student-only code token is scored by the teacher like any other token, not masked.
    code_row = len(PROMPT) + RESPONSE.index(CODE_TOKEN) - 1
    assert ids[code_row, 0].item() == CODE_TOKEN and logprobs[code_row, 0].item() < 0

    # Scoring the sequence with the <|endoftext|> gives other response values, which is the
    # shift that the fix removes.
    _, unstripped = _teacher_answer(STUDENT)
    assert unstripped[start:start + len(RESPONSE), 0].tolist() != response_rows


def test_without_a_prefix_the_rows_are_verls_own():
    student = PROMPT[1:] + RESPONSE    # a chat-template prompt has no <|endoftext|>
    assert teacher.stripped_prefix_length(PROMPT[1:], teacher.DEFAULT_STRIP_PREFIX) == 0
    ids, logprobs = _teacher_answer(student)
    out_ids, out_logprobs = teacher.realign_teacher_rows(ids, logprobs, 0)
    assert out_ids is ids and out_logprobs is logprobs
    teacher.check_teacher_rows(out_ids.shape[0], student, 0, out_ids[:, 0].tolist())


def test_top_k_rows_get_dummy_rows_of_the_same_width_in_front():
    torch = _torch()
    for n in (1, 2):
        answer_ids, answer_logprobs = _teacher_answer(STUDENT[n:], topk=4)
        ids, logprobs = teacher.realign_teacher_rows(answer_ids, answer_logprobs, n)
        assert tuple(ids.shape) == tuple(logprobs.shape) == (len(STUDENT), 4)
        assert ids.dtype == torch.int32 and logprobs.dtype == torch.float32
        # The rows in front look exactly like verl's own dummy row at the end.
        for row in [*range(n), len(STUDENT) - 1]:
            assert ids[row].tolist() == [0, 0, 0, 0] and logprobs[row].tolist() == [0.0] * 4
        assert torch.equal(ids[n:], answer_ids) and torch.equal(logprobs[n:], answer_logprobs)
        teacher.check_teacher_rows(ids.shape[0], STUDENT, n)


def test_realign_refuses_rows_of_different_shapes():
    torch = _torch()
    ids, logprobs = _teacher_answer(STUDENT[1:])
    with pytest.raises(ValueError, match=r"\[S, K\]"):
        teacher.realign_teacher_rows(ids, torch.cat([logprobs, logprobs], dim=1), 1)
    with pytest.raises(ValueError, match=r"\[S, K\]"):
        teacher.realign_teacher_rows(ids[:, 0], logprobs[:, 0], 1)
    with pytest.raises(ValueError, match="at least 0"):
        teacher.realign_teacher_rows(ids, logprobs, -1)


def test_the_fake_teacher_matches_verls_conversion_of_vllm_prompt_logprobs():
    torch = _torch()
    try:
        from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs
    except Exception as exc:  # vLLM can refuse to import on a node without a GPU
        pytest.skip(f"verl's vLLM utils do not import here: {exc}")

    sequence = PROMPT[1:] + RESPONSE
    for topk in (0, 4):
        # vLLM gives no log-probs for the first prompt token, then one dict per position. With
        # prompt_logprobs=0 the dict holds only the token in the sequence, at its own rank.
        prompt_logprobs = [None]
        for j in range(1, len(sequence)):
            row = [sequence[j]] if topk == 0 else [sequence[j] + r for r in range(topk)]
            ranks = [7] if topk == 0 else list(range(1, topk + 1))
            prompt_logprobs.append({
                token: SimpleNamespace(logprob=_teacher_logprob(sequence[:j], token), rank=rank)
                for token, rank in zip(row, ranks)
            })
        result: dict = {}
        extract_prompt_logprobs(SimpleNamespace(prompt_logprobs=prompt_logprobs), topk, result)
        # The teacher client turns the lists into tensors this way (teacher_manager.py).
        ids = torch.tensor(result["prompt_ids"], dtype=torch.int32)
        logprobs = torch.tensor(result["prompt_logprobs"])
        want_ids, want_logprobs = _teacher_answer(sequence, topk)
        assert torch.equal(ids, want_ids) and torch.equal(logprobs, want_logprobs), topk


# ---------------------------------------------------------------------------------------------
# The worker override and the manager, on verl's real classes, without Ray
# ---------------------------------------------------------------------------------------------


class _FakeTeacherClient:
    """Stands in for verl's ``AsyncTeacherLLMServerManager``: it records what the teacher is sent.

    ``shift`` rolls the returned rows down, which is what a misaligned teacher answer looks like.
    """

    def __init__(self, topk=0, shift=0):
        self.topk = topk
        self.shift = shift
        self.sequences: list[list[int]] = []
        self.routing_keys: list = []
        self.distillation_loss_config = SimpleNamespace(loss_settings=SimpleNamespace(use_topk=topk > 0))

    async def compute_teacher_logprobs_single(self, sequence_ids, multi_modal_data=None, mm_processor_kwargs=None,
                                              mm_processor_output=None, routing_key=None):
        self.sequences.append(list(sequence_ids))
        self.routing_keys.append(routing_key)
        ids, logprobs = _teacher_answer(list(sequence_ids), self.topk)
        if self.shift:
            ids, logprobs = ids.roll(self.shift, dims=0), logprobs.roll(self.shift, dims=0)
        return ids, logprobs


def _worker(prefix=teacher.DEFAULT_STRIP_PREFIX, topk=0, shift=0, enabled=True):
    """Our worker with only the attributes the teacher call reads.

    ``__init__`` is skipped because it needs a model dir, Ray and TransferQueue. The attributes set
    here are the ones verl's ``AgentLoopWorker.__init__`` and ours set for the teacher call.
    """
    pytest.importorskip("verl.trainer.ppo.v1.agent_loop_tq")
    actor_cls = teacher.agent_loop_worker_class()
    plain = actor_cls.__ray_metadata__.modified_class.__ray_actor_class__
    worker = plain.__new__(plain)
    worker.distillation_enabled = enabled
    worker.teacher_key = "data_source"
    worker.teacher_server_manager = _FakeTeacherClient(topk, shift)
    worker.teacher_strip_prefix = tuple(prefix)
    worker._warned_prompt_without_prefix = False
    worker._teacher_alignment_error = None
    return worker


def _output(prompt, response):
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    return AgentLoopOutput(prompt_ids=list(prompt), response_ids=list(response),
                           response_mask=[1] * len(response), metrics=AgentLoopMetrics())


def _score(worker, output, validate=False):
    """Call the teacher the way verl's TransferQueue worker does in ``_agent_loop_postprocess``."""
    asyncio.run(worker._compute_teacher_logprobs(
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=validate,
        sample_kwargs={"data_source": "orbit/math", "uid": "u0", "session_id": 0},
    ))


def test_the_worker_sends_the_teacher_the_prompt_without_the_endoftext():
    torch = _torch()
    worker = _worker()
    output = _output(PROMPT, RESPONSE)
    _score(worker, output)

    client = worker.teacher_server_manager
    assert client.sequences == [PROMPT[1:] + RESPONSE]
    assert client.routing_keys == ["orbit/math"]   # verl's own routing, untouched
    want_ids, want_logprobs = teacher.realign_teacher_rows(*_teacher_answer(PROMPT[1:] + RESPONSE), 1)
    assert torch.equal(output.extra_fields["teacher_ids"], want_ids)
    assert torch.equal(output.extra_fields["teacher_logprobs"], want_logprobs)
    # The student's own prompt is what TransferQueue stores, so it keeps its <|endoftext|>.
    assert output.prompt_ids == PROMPT

    # as_dict is how the TransferQueue worker stores a sample: one teacher row per stored token.
    stored = output.as_dict()
    assert stored["prompts"].shape[0] + stored["responses"].shape[0] == stored["teacher_logprobs"].shape[0]
    assert torch.equal(stored["teacher_logprobs"], want_logprobs)


def test_a_prompt_without_the_prefix_and_an_empty_prefix_reach_the_teacher_unchanged():
    torch = _torch()
    worker = _worker()
    output = _output(PROMPT[1:], RESPONSE)
    _score(worker, output)
    assert worker.teacher_server_manager.sequences == [PROMPT[1:] + RESPONSE]
    assert torch.equal(output.extra_fields["teacher_logprobs"], _teacher_answer(PROMPT[1:] + RESPONSE)[1])
    assert worker._warned_prompt_without_prefix is True

    off = _worker(prefix=())
    output = _output(PROMPT, RESPONSE)
    _score(off, output)
    assert off.teacher_server_manager.sequences == [STUDENT]
    assert torch.equal(output.extra_fields["teacher_logprobs"], _teacher_answer(STUDENT)[1])
    assert off._warned_prompt_without_prefix is False


def test_validation_and_disabled_distillation_never_call_the_teacher():
    for worker, validate in ((_worker(), True), (_worker(enabled=False), False)):
        output = _output(PROMPT, RESPONSE)
        _score(worker, output, validate=validate)
        assert worker.teacher_server_manager.sequences == []
        assert "teacher_ids" not in output.extra_fields and "teacher_logprobs" not in output.extra_fields


def test_top_k_answers_are_realigned_too():
    worker = _worker(topk=4)
    output = _output(PROMPT, RESPONSE)
    _score(worker, output)
    ids = output.extra_fields["teacher_ids"]
    assert tuple(ids.shape) == (len(STUDENT), 4)
    assert ids[0].tolist() == [0, 0, 0, 0] and ids[1, 0].item() == STUDENT[2]


def test_a_misaligned_teacher_answer_fails_the_sample():
    worker = _worker(shift=1)
    with pytest.raises(RuntimeError, match="not aligned") as first:
        _score(worker, _output(PROMPT, RESPONSE))
    # The next dispatch of a batch to this worker stops the run with that failure. The override
    # raises before verl's own method runs, so no TransferQueue is needed here.
    with pytest.raises(RuntimeError, match="Stopping the run") as stop:
        asyncio.run(worker.generate_sequences(None))
    assert stop.value.__cause__ is first.value


class _StockWorker:
    """Stands in for verl's plain ``AgentLoopWorkerTQ`` class, so the overrides run without verl or Ray.

    It keeps only what the overrides rely on. ``generate_sequences`` records each batch it is
    handed, and ``_compute_teacher_logprobs`` asks the teacher and stores its rows the way verl's
    method does.
    """

    def __init__(self, config):
        self.config = config
        self.distillation_enabled = True
        self.teacher_server_manager = _FakeTeacherClient()
        self.dispatched = []

    async def generate_sequences(self, batch):
        self.dispatched.append(batch)

    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
        ids, logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=list(prompt_ids) + list(response_ids))
        output.extra_fields["teacher_ids"], output.extra_fields["teacher_logprobs"] = ids, logprobs


def test_a_misaligned_teacher_answer_stops_the_run_at_the_next_dispatch():
    """verl only drops a sample whose teacher call raised, so the worker keeps the first alignment
    failure and refuses the next batch with it. The task runner then stops within one step."""
    _torch()
    worker = teacher.worker_subclass(_StockWorker)({})
    asyncio.run(worker.generate_sequences("step 1"))
    output = SimpleNamespace(prompt_ids=PROMPT, response_ids=RESPONSE, extra_fields={})
    _score(worker, output)   # an aligned answer passes and is not recorded
    asyncio.run(worker.generate_sequences("step 2"))
    assert worker.dispatched == ["step 1", "step 2"]

    worker.teacher_server_manager.shift = 1
    with pytest.raises(RuntimeError, match="not aligned") as first:
        _score(worker, SimpleNamespace(prompt_ids=PROMPT, response_ids=RESPONSE, extra_fields={}))
    # A second failure does not replace the first one.
    worker.teacher_server_manager.shift = 2
    with pytest.raises(RuntimeError, match="not aligned"):
        _score(worker, SimpleNamespace(prompt_ids=PROMPT, response_ids=RESPONSE, extra_fields={}))

    with pytest.raises(RuntimeError, match="Stopping the run") as stop:
        asyncio.run(worker.generate_sequences("step 3"))
    assert stop.value.__cause__ is first.value and str(first.value) in str(stop.value)
    assert worker.dispatched == ["step 1", "step 2"]


def test_the_manager_loads_by_name_and_starts_our_worker():
    pytest.importorskip("verl.trainer.ppo.v1.agent_loop_tq")
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
    from verl.utils.import_utils import load_class_from_fqn

    # verl's task runner loads the class exactly this way (TaskRunnerV1.init_agent_loop_manager).
    manager_cls = load_class_from_fqn(teacher.MANAGER_CLASS_FQN, "AgentLoopManager")
    assert manager_cls is teacher.agent_loop_manager_class()
    assert manager_cls is getattr(teacher, teacher.MANAGER_CLASS_NAME)
    assert issubclass(manager_cls, AgentLoopManagerTQ) and manager_cls is not AgentLoopManagerTQ
    assert manager_cls.__qualname__ == teacher.MANAGER_CLASS_NAME

    # __init__ only records the worker class; create() would start the workers on Ray.
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {}, "model": {}}})
    manager = manager_cls(config=config, llm_client=None)
    worker_cls = manager.agent_loop_workers_class
    assert worker_cls is teacher.agent_loop_worker_class()
    assert isinstance(worker_cls, ray.actor.ActorClass)
    stock = AgentLoopWorkerTQ.__ray_metadata__.modified_class.__ray_actor_class__
    plain = worker_cls.__ray_metadata__.modified_class.__ray_actor_class__
    assert issubclass(plain, stock) and plain is not stock
