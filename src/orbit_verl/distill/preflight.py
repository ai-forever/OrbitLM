"""Pre-flight checks of a run with ``distillation.enabled``, before Ray or any model starts.

verl 0.9.1 checks very little about the teacher before the run is up. A teacher that is not on
disk fails only after Ray, the student and the teacher's vLLM servers have started. A teacher with
another tokenizer does not fail at all. verl hands the teacher the student's token ids as they are,
so such a teacher returns log-probs of other tokens, and the run trains on numbers that look
valid and mean nothing. The checks here catch both, plus the settings that make the per-token
advantage ``log p_teacher - log p_student`` wrong or crash the first update:

- (a) every teacher's ``model_path`` is an absolute path of a local directory;
- (b) the teacher's tokenizer gives every Qwen3 id the same token as the student's tokenizer,
  the special tokens sit at the Qwen3 ids, and the teacher's head has Qwen3's 151,936 rows;
- (c) ``actor_rollout_ref.rollout.temperature`` is 1.0;
- (d) ``actor_rollout_ref.actor.policy_loss.loss_mode`` is one that runs under distillation;
- (e) ``actor_rollout_ref.actor.use_kl_loss`` is refused while the task loss is off, because verl
  then drops the KL term; with the task loss on, it only gives a warning about its cost;
- the teacher's context lengths fit what verl and vLLM accept, and verl's own
  ``DistillationConfig`` accepts the block;
- a thinking budget (``orbit.env.ORBIT_THINK_BUDGET``) leaves room in the response length for the
  stop-thinking text and the answer after it;
- with several teachers and a distill dataset, the rows are routed by their think or no_think
  mode, and the teachers' keys are exactly those two modes;
- the run uses the agent-loop manager of ``orbit_verl.distill.teacher``, which keeps the leading
  ``<|endoftext|>`` of our prompts away from the teacher (``teacher_prompt_problems`` there).

The launcher runs them in ``run_checks``, also for ``--dry-run`` and ``--prepare-only``, and every
problem carries the override that fixes it.

The config may be a composed omegaconf config or a plain nested dict, so the checks can be tested
without verl or omegaconf. Only ``check_verl_config`` imports verl.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping

from orbit_verl.data.distill_data import TEACHER_ROUTE, THINKING_MODES
from orbit_verl.data.prompts import THINK_BUDGET_ENV, think_budget
from orbit_verl.launch.config import hydra_value
from orbit_verl.launch.env import _cfg_get
from orbit_verl.launch.prepare import DISTILL_BUILDER, MIXED_PROMPT_MODE, Problem

# Rows of the output head of every Qwen3 model, and of our release models.
QWEN3_VOCAB_SIZE = 151936
# Ids below this one are Qwen3's own tokens: 151,643 BPE tokens and 26 added tokens. Our tokenizer
# adds 12 code tokens from here on (151669-151680). The teacher never learned those, so only the
# ids below this bound have to mean the same token in both tokenizers.
QWEN3_TOKENS = 151669
# The tokens that our prompts, the think block and the stop tokens depend on, at their Qwen3 ids.
SPECIAL_TOKENS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<think>": 151667,
    "</think>": 151668,
}
# Policy loss modes that can train under distillation. With distillation on, verl's actor loss is
# distillation_ppo_loss, which calls verl's stock ppo_loss. OrbitLMRLTTEngine replaces only a plain
# ppo_loss by the RLTT loss, so an rltt_* mode reaches the raising stub in our loss registry on the
# first update. Add the rltt_* modes here once the engine also replaces the distillation loss.
DISTILL_LOSS_MODES = ("vanilla",)
# An upper bound on the ids of the stop-thinking text that the agent loop inserts when the thinking
# budget runs out (``response_info.STOP_THINKING_TEXT``). Qwen3's tokenizer writes it in about 25
# ids. The exact count needs the tokenizer, so the budget check counts this bound instead.
STOP_THINKING_MAX_IDS = 64
# The least room that the agent loop leaves for the answer after a forced close. This is a copy of
# ``orbit_verl.data.agent_loop.MIN_ANSWER_TOKENS``, because that module imports verl and these
# checks must run without it. A test keeps the two values equal.
MIN_ANSWER_TOKENS = 256

_TEACHERS = "distillation.teacher_models"
_TEACHER_KEY = "distillation.teacher_key"
_TEMPERATURE = "actor_rollout_ref.rollout.temperature"
_LOSS_MODE = "actor_rollout_ref.actor.policy_loss.loss_mode"
_KL_LOSS = "actor_rollout_ref.actor.use_kl_loss"
_TASK_REWARDS = "distillation.distillation_loss.use_task_rewards"
_THINK_BUDGET = f"orbit.env.{THINK_BUDGET_ENV}"
# The cap on how many differing ids one message lists; the count is always given in full.
_EXAMPLES = 5


def select(cfg: Any, dotted: str, default: Any = None) -> Any:
    """``cfg.<dotted>`` for an omegaconf config or a plain dict; ``default`` when absent or unset.

    The launcher's ``_cfg_get`` does the walk. On top of it, the mandatory marker ``???`` counts as
    unset: omegaconf raises for it, and a plain dict keeps it as a string. A value whose
    interpolation cannot be resolved also gives ``default``, so the check that needs it reports
    the missing value instead of crashing.
    """
    try:
        value = _cfg_get(cfg, dotted, default)
    except Exception:  # noqa: BLE001 - omegaconf raises several types for ??? and unresolvable values
        return default
    return default if isinstance(value, str) and value == "???" else value


def is_enabled(cfg: Any) -> bool:
    """True when the config turns verl's on-policy distillation on."""
    return bool(select(cfg, "distillation.enabled", False))


def teacher_entries(cfg: Any) -> dict[str, Any]:
    """The teacher entries that verl will run, keyed by their name under ``distillation.teacher_models``.

    verl runs the default entry ``teacher_model`` only while it is the only entry. Once another
    entry is added, verl drops ``teacher_model`` without a word (``DistillationConfig._resolve_teacher_models``).
    Checking it then would check a teacher that never runs, so it is left out.
    """
    models = select(cfg, _TEACHERS)
    if models is None:
        return {}
    names = list(models.keys())
    if len(names) > 1:
        names = [name for name in names if name != "teacher_model"]
    return {name: models[name] for name in names}


def student_lengths(cfg: Any) -> tuple[int, int]:
    """The student's rollout prompt and response lengths.

    verl's default teacher entry copies both through interpolations, and the teacher's context has
    to hold a whole student sequence. The rollout keys themselves default to
    ``data.max_prompt_length`` and ``data.max_response_length``.
    """
    prompt = select(cfg, "actor_rollout_ref.rollout.prompt_length", select(cfg, "data.max_prompt_length", 0))
    response = select(cfg, "actor_rollout_ref.rollout.response_length", select(cfg, "data.max_response_length", 0))
    return int(prompt), int(response)


def required_teacher_context(cfg: Any) -> int:
    """The smallest teacher ``max_model_len`` that verl accepts: a student sequence plus one token.

    The teacher scores a sequence by "generating" one token after it, so verl asks for room for
    the prompt, the whole response, and that one token (``validate_and_prepare_for_distillation``).
    """
    prompt, response = student_lengths(cfg)
    return prompt + response + 1


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ("", "???") else text


def _default_hub_dirs() -> list[str]:
    """Where the Hugging Face hub cache of the person who launches the run would be.

    The launcher moves ``XDG_CACHE_HOME``, and with it Hugging Face's default cache, under the runs
    root. Only the person's own ``HF_HUB_CACHE`` and the plain default are worth looking at for a
    model that they downloaded earlier.
    """
    dirs = [os.environ.get("HF_HUB_CACHE"), os.path.join("~", ".cache", "huggingface", "hub")]
    return [d for d in dirs if d]


def hub_snapshot(repo_id: str, hub_dirs: Iterable[str] | None = None) -> str | None:
    """The local snapshot dir of a hub id such as ``Qwen/Qwen3-8B``, or None when none is found.

    The hub cache keeps each repo under ``models--<org>--<name>``. Its ``refs/main`` names the
    snapshot that a plain download fetched. Without that file a single snapshot dir is taken, and
    several of them are ambiguous, so none is suggested.
    """
    if repo_id.count("/") != 1 or repo_id.startswith(("/", ".", "~")):
        return None
    folder = "models--" + repo_id.replace("/", "--")
    for hub in _default_hub_dirs() if hub_dirs is None else hub_dirs:
        root = os.path.join(os.path.expanduser(hub), folder)
        snapshots = os.path.join(root, "snapshots")
        try:
            with open(os.path.join(root, "refs", "main"), encoding="utf-8") as handle:
                ref = handle.read().strip()
        except OSError:
            ref = ""
        if ref and os.path.isdir(os.path.join(snapshots, ref)):
            return os.path.join(snapshots, ref)
        try:
            names = sorted(name for name in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, name)))
        except OSError:
            continue
        if len(names) == 1:
            return os.path.join(snapshots, names[0])
    return None


def _local_dir_hint(path: str, hub_dirs: Iterable[str] | None) -> str | None:
    """An absolute local directory that ``path`` most likely meant, or None."""
    expanded = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(expanded):
        return expanded
    return hub_snapshot(path, hub_dirs)


def check_teacher_paths(cfg: Any, hub_dirs: Iterable[str] | None = None) -> list[Problem]:
    """(a) Every teacher that verl will run has an absolute path of a local directory.

    The launcher moves the Hugging Face cache under the runs root, one cache per host. A hub id
    would therefore not be found in the person's own cache. Each node would download the teacher
    again, and on a machine without internet access the teacher's vLLM server would fail only after
    the run has started. A relative or ``~`` path would depend on the directory and the user that the
    teacher's Ray worker runs with, so only an absolute path is accepted.
    """
    entries = teacher_entries(cfg)
    if not entries:
        return [Problem(
            "distill", f"distillation.enabled is true, but {_TEACHERS} has no entry, so there is no teacher.",
            (f"{_TEACHERS}.teacher_model.model_path=/path/to/the/teacher",))]
    problems = []
    for name, entry in entries.items():
        key = f"{_TEACHERS}.{name}.model_path"
        path = _text(select(entry, "model_path"))
        if path is None:
            problems.append(Problem("distill", f"{key} is not set. Give the local directory of the teacher's "
                                    "weights, config.json and tokenizer.json.",
                                    (f"{key}=/path/to/the/teacher",)))
            continue
        if os.path.isabs(path) and os.path.isdir(path):
            continue
        hint = _local_dir_hint(path, hub_dirs)
        problems.append(Problem(
            "distill", f"{key} is {path!r}, which is not an absolute path of a local directory. The "
            "launcher moves the Hugging Face cache under the runs root, so the teacher's vLLM server would "
            "not find a hub id in ~/.cache and would have to download it on every node. A relative or ~ "
            "path would depend on the directory and the user that the teacher's Ray worker runs with.",
            (f"{key}={hydra_value(hint)}",) if hint else (f"{key}=/absolute/path/to/the/teacher",)))
    return problems


def read_tokenizer(path: str) -> tuple[dict[int, str], list[str]]:
    """Every id of a ``tokenizer.json`` with its token, and the BPE merges in one spelling.

    The ids come from the BPE vocabulary (``model.vocab``) and from ``added_tokens``, which hold
    the special tokens. The merges are compared only for a warning. Newer tokenizers write each
    merge as a pair and older ones as one string with a space, so both become the string form.
    Raises ValueError for a file that is not a BPE tokenizer.json.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    model = data.get("model") if isinstance(data, dict) else None
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if not isinstance(vocab, dict):
        raise ValueError(f"{path} has no model.vocab mapping, so it is not a BPE tokenizer.json")
    ids = {int(index): token for token, index in vocab.items()}
    for entry in data.get("added_tokens") or []:
        ids[int(entry["id"])] = entry["content"]
    merges = [" ".join(item) if isinstance(item, list) else str(item) for item in model.get("merges") or []]
    return ids, merges


def vocab_mismatches(student: Mapping[int, str], teacher: Mapping[int, str], below: int = QWEN3_TOKENS) -> list[int]:
    """Ids below ``below`` that the two tokenizers give different tokens, or that only one of them has."""
    return [index for index in range(below) if student.get(index) != teacher.get(index)]


def _describe_ids(ids: list[int], student: Mapping[int, str], teacher: Mapping[int, str]) -> str:
    shown = [f"id {index}: student {student.get(index)!r}, teacher {teacher.get(index)!r}" for index in ids[:_EXAMPLES]]
    more = f" and {len(ids) - _EXAMPLES} more" if len(ids) > _EXAMPLES else ""
    return "; ".join(shown) + more


def _special_problem(ids: Mapping[int, str], who: str, path: str) -> Problem | None:
    wrong = [f"{token} should be id {index}, which is {ids.get(index)!r}"
             for token, index in SPECIAL_TOKENS.items() if ids.get(index) != token]
    if not wrong:
        return None
    return Problem("distill", f"The {who} tokenizer {path} does not have Qwen3's special tokens at Qwen3's ids: "
                   f"{'; '.join(wrong)}. Our prompts, the think block and the stop tokens rely on those ids.")


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def _model_config_value(config: Mapping[str, Any], key: str) -> Any:
    """A model config value, also when it sits under ``text_config``, as in multimodal configs."""
    if config.get(key) is not None:
        return config[key]
    text = config.get("text_config")
    return text.get(key) if isinstance(text, dict) else None


def _student_tokenizer_dir(cfg: Any) -> str | None:
    # verl loads the tokenizer from tokenizer_path and falls back to the model path when it is null.
    return _text(select(cfg, "actor_rollout_ref.model.tokenizer_path")) or _text(
        select(cfg, "actor_rollout_ref.model.path"))


def check_teacher_tokenizers(cfg: Any) -> tuple[list[Problem], list[str]]:
    """(b) The teacher reads our token ids the way the student writes them. Returns problems and warnings.

    verl sends the teacher the student's prompt and response ids unchanged. The check therefore
    compares what each id means, read from both ``tokenizer.json`` files with the json module:

    - every id below 151,669 has the same token in both tokenizers. Our 12 extra code tokens start
      at 151,669, and the teacher simply scores them as untrained rows;
    - ``<|endoftext|>``, ``<|im_start|>``, ``<|im_end|>``, ``<think>`` and ``</think>`` sit at Qwen3's ids;
    - the teacher's config.json has ``vocab_size`` 151,936. Qwen3.5 and later models have 248,320
      rows with other ids for the same text, and they would return garbage without any error.

    Different BPE merges only give a warning. They do not change what an id means, but the
    teacher would split the same text differently, so the student's token sequences look unusual
    to it. A teacher dir that is missing is left to ``check_teacher_paths``. A student dir that is
    missing is left to the launcher's model checks, which also say how to build it.
    """
    problems: list[Problem] = []
    warnings: list[str] = []
    student: dict[int, str] | None = None
    student_merges: list[str] = []
    student_dir = _student_tokenizer_dir(cfg)
    student_file = os.path.join(student_dir, "tokenizer.json") if student_dir else ""
    if student_dir and os.path.isdir(student_dir):
        try:
            student, student_merges = read_tokenizer(student_file)
        except (OSError, ValueError) as exc:
            problems.append(Problem("distill", f"Cannot read the student's tokenizer {student_file}, so it cannot "
                                    f"be compared with the teacher's: {exc}"))
        else:
            special = _special_problem(student, "student's", student_file)
            if special is not None:
                problems.append(special)

    for name, entry in teacher_entries(cfg).items():
        path = _text(select(entry, "model_path"))
        if path is None or not os.path.isdir(path):
            continue
        config_file = os.path.join(path, "config.json")
        try:
            vocab_size = _model_config_value(_read_json(config_file), "vocab_size")
        except (OSError, ValueError) as exc:
            problems.append(Problem("distill", f"Cannot read the teacher {name}'s config {config_file}: {exc}"))
        else:
            if vocab_size != QWEN3_VOCAB_SIZE:
                problems.append(Problem(
                    "distill", f"The teacher {name} at {path} has vocab_size {vocab_size} in its config.json, not "
                    f"Qwen3's {QWEN3_VOCAB_SIZE}. verl hands the teacher the student's token ids unchanged, so a "
                    "teacher with another vocabulary (Qwen3.5 and later have 248,320 rows) would score other "
                    "tokens and return log-probs that look valid and mean nothing. Use a Qwen3-family teacher, "
                    "such as Qwen3-8B, Qwen3-14B, Qwen3-32B or Qwen3-30B-A3B-Thinking-2507."))
        teacher_file = os.path.join(path, "tokenizer.json")
        try:
            teacher, teacher_merges = read_tokenizer(teacher_file)
        except (OSError, ValueError) as exc:
            problems.append(Problem("distill", f"Cannot read the teacher {name}'s tokenizer {teacher_file}, so it "
                                    f"cannot be compared with the student's: {exc}"))
            continue
        special = _special_problem(teacher, f"teacher {name}'s", teacher_file)
        if special is not None:
            problems.append(special)
        if student is None:
            continue
        wrong = vocab_mismatches(student, teacher)
        if wrong:
            problems.append(Problem(
                "distill", f"The teacher {name}'s tokenizer ({teacher_file}) and the student's ({student_file}) "
                f"disagree on {len(wrong)} of the {QWEN3_TOKENS} Qwen3 ids: {_describe_ids(wrong, student, teacher)}. "
                "The teacher scores the student's ids as they are, so it would score other text."))
        elif teacher_merges != student_merges:
            warnings.append(
                f"The teacher {name}'s tokenizer ({teacher_file}) has other BPE merges than the student's "
                f"({student_file}). Every id still means the same token, but the teacher would split the same "
                "text differently, so the student's token sequences may look unusual to it.")
    return problems, warnings


def check_temperature(cfg: Any) -> list[Problem]:
    """(c) The student samples at temperature 1.0, the temperature at which the teacher scores.

    verl's teacher always scores at 1.0, whatever its config says (``_get_teacher_sampling_params``).
    The student's training log-probs, however, come from its logits divided by the rollout
    temperature. At any other temperature the advantage ``log p_teacher - log p_student``
    compares two different distributions, and every token is pushed by that difference alone.
    """
    value = select(cfg, _TEMPERATURE)
    try:
        ok = float(value) == 1.0
    except (TypeError, ValueError):
        ok = False
    if ok:
        return []
    return [Problem(
        "distill", f"{_TEMPERATURE} is {value}, but distillation needs 1.0. verl's teacher always scores at "
        "temperature 1.0, while the student's log-probs are taken from its logits divided by the rollout "
        "temperature. The per-token advantage log p_teacher - log p_student would then compare two different "
        "distributions and push every token by the gap between them.",
        (f"{_TEMPERATURE}=1.0",))]


def check_loss_mode(cfg: Any) -> list[Problem]:
    """(d) The actor's policy loss is one that can train under distillation (``DISTILL_LOSS_MODES``)."""
    mode = str(select(cfg, _LOSS_MODE, "vanilla"))
    if mode in DISTILL_LOSS_MODES:
        return []
    return [Problem(
        "distill", f"{_LOSS_MODE} is {mode}, which cannot train under distillation. With distillation on, verl's "
        "actor loss is distillation_ppo_loss, which calls verl's stock ppo_loss. OrbitLMRLTTEngine replaces only "
        "a plain ppo_loss by the RLTT loss, so the rltt_* stub in our loss registry would raise on the first "
        "update. vanilla trains the final loop, which is the loop vLLM samples from.",
        (f"{_LOSS_MODE}=vanilla",))]


def _kl_loss_and_task_rewards(cfg: Any) -> tuple[bool, bool]:
    # verl's distillation.yaml defaults use_task_rewards to true, so an absent key means true.
    return (bool(select(cfg, _KL_LOSS, False)), bool(select(cfg, _TASK_REWARDS, True)))


def check_kl_loss(cfg: Any) -> list[Problem]:
    """(e) The KL loss to the reference model is refused when the task loss is off.

    verl adds the KL term inside the task policy loss (``ppo_loss`` in verl/workers/utils/losses.py).
    With ``use_task_rewards=false``, ``distillation_ppo_loss`` sets that whole policy loss to 0, so
    the KL term goes with it (verl/trainer/distillation/losses.py). The run would still pay for the
    reference model and log an ``actor/kl_loss`` value, and so look as if the KL were active.

    A KL arm turns the task loss on as well. Rows without a gold answer (orbit/none) score 0 in
    every answer, so their GRPO advantage is exactly 0, and the task loss then adds only the KL.
    """
    kl_loss, task_rewards = _kl_loss_and_task_rewards(cfg)
    if not kl_loss or task_rewards:
        return []
    return [Problem(
        "distill", f"{_KL_LOSS} is true, but {_TASK_REWARDS} is false. verl adds the KL term inside the task "
        "policy loss and sets that whole loss to 0 when the task rewards are off, so the KL would never train "
        "while the run still pays for the reference model and logs a KL value. Turn the task loss on as well: "
        "rows without a gold answer (orbit/none) score 0 in every answer, so their GRPO advantage is 0 and "
        f"only the KL is added. Or set {_KL_LOSS}=false to run without the KL.",
        (f"{_TASK_REWARDS}=true",))]


def kl_loss_warnings(cfg: Any) -> list[str]:
    """(e) A warning when the actor keeps a KL loss to the reference model that actually trains.

    It is not refused: an A/B against the RL runs may want the same KL to the start model in both
    arms. It costs a reference model and a reference log-prob pass every step, though. The setting
    that would not train at all is refused by ``check_kl_loss`` instead.
    """
    kl_loss, task_rewards = _kl_loss_and_task_rewards(cfg)
    if not kl_loss or not task_rewards:
        return []
    return [f"{_KL_LOSS} is true. verl then keeps a reference model and runs a "
            "reference log-prob pass every step, although the teacher already keeps the student close to a "
            "good distribution. Keep it only when the run must match an RL arm that has the same KL; "
            f"otherwise set {_KL_LOSS}=false."]


def _builds_distill_dataset(cfg: Any) -> bool:
    """Whether the run's dataset comes from the distill builder, the rule of ``prepare.data_builder``.

    ``prepare.data_builder`` reads the config through omegaconf, and these checks also run on
    plain dicts, so its two-line rule is applied here with ``select``.
    """
    builder = select(cfg, "orbit.prepare.data.builder")
    if builder is None:
        return str(select(cfg, "orbit.prepare.data.prompt_mode")) == MIXED_PROMPT_MODE
    return str(builder) == DISTILL_BUILDER


def check_teacher_routing(cfg: Any) -> list[Problem]:
    """With several teachers, every row of a distill dataset reaches one of them.

    verl routes each row to the teacher whose ``key`` equals the row's value in the column that
    ``distillation.teacher_key`` names, and it fails on the first rollout when no teacher has that
    key (``_resolve_teacher_key`` in verl/experimental/teacher_loop/teacher_manager.py). The
    distill builder writes the row's thinking mode, think or no_think, into the column
    teacher_route. So that column must be the routing key, and the teachers' keys must be exactly
    those two modes. A single teacher gets every row whatever the key says, so it is not checked.
    """
    entries = teacher_entries(cfg)
    if len(entries) < 2 or not _builds_distill_dataset(cfg):
        return []
    problems = []
    key = select(cfg, _TEACHER_KEY)
    if key != TEACHER_ROUTE:
        problems.append(Problem(
            "distill", f"{_TEACHERS} has {len(entries)} teachers, so verl routes every row by the column that "
            f"{_TEACHER_KEY} names, which is {key!r}. The distill builder writes each row's thinking mode into "
            f"the column {TEACHER_ROUTE}, and verl would fail on the first rollout when a value of {key!r} matches "
            "no teacher's key.",
            (f"{_TEACHER_KEY}={TEACHER_ROUTE}",)))
    keys = {name: select(entry, "key") for name, entry in entries.items()}
    if sorted(map(str, keys.values())) != sorted(THINKING_MODES):
        # Entries named after a mode keep that mode; the others get the remaining modes in order.
        free = [mode for mode in THINKING_MODES if mode not in keys]
        fixes = [f"{_TEACHERS}.{name}.key={name if name in THINKING_MODES else free.pop(0)}"
                 for name in entries if name in THINKING_MODES or free]
        guessed = any(name not in THINKING_MODES for name in entries)
        problems.append(Problem(
            "distill", f"The teachers' keys are {keys}, but the rows of a distill dataset are routed to "
            f"{list(THINKING_MODES)} by their {TEACHER_ROUTE} column. Give exactly one teacher each of those keys: "
            "the think teacher think and the no-think teacher no_think."
            + (" Swap the values below if the entries are the other way round." if guessed else ""),
            tuple(fixes)))
    return problems


def check_teacher_lengths(cfg: Any) -> list[Problem]:
    """The teacher's context holds a whole student sequence, and the teacher model allows that context.

    - Every entry has an ``inference`` block whose prompt and response lengths are the student's
      rollout ones. verl's default ``teacher_model`` entry copies them through interpolations. An
      entry added for a second teacher without them gets RolloutConfig's 512 and 512, and verl
      would size that teacher's server for 1,024-token sequences.
    - An explicit ``inference.max_model_len`` holds a student sequence plus one token, as verl
      demands at start-up. A null one is filled by the launcher (``finish_distillation``).
    - The context does not exceed the teacher's ``max_position_embeddings``, which verl refuses
      when the teacher's server starts.
    """
    prompt, response = student_lengths(cfg)
    need = prompt + response + 1
    problems = []
    for name, entry in teacher_entries(cfg).items():
        prefix = f"{_TEACHERS}.{name}.inference"
        inference = select(entry, "inference")
        if inference is None:
            problems.append(Problem(
                "distill", f"{prefix} is missing. verl would run this teacher with RolloutConfig's defaults, "
                "including 512-token prompt and response lengths. Copy the inference block of verl's "
                f"{_TEACHERS}.teacher_model entry into this entry in the experiment YAML."))
            continue
        lengths = (select(inference, "prompt_length"), select(inference, "response_length"))
        if lengths != (prompt, response):
            problems.append(Problem(
                "distill", f"{prefix}.prompt_length and response_length are {lengths[0]} and {lengths[1]}, but "
                f"the student's rollout uses {prompt} and {response}. verl sizes the teacher's server from these "
                "values, so they must be the student's.",
                (f"{prefix}.prompt_length={prompt}", f"{prefix}.response_length={response}")))
        max_model_len = select(inference, "max_model_len")
        if max_model_len is not None and int(max_model_len) < need:
            problems.append(Problem(
                "distill", f"{prefix}.max_model_len is {max_model_len}, below the student's prompt {prompt} + "
                f"response {response} + 1 = {need}. verl refuses this at start-up, because the teacher has to "
                "read a whole student sequence.",
                (f"{prefix}.max_model_len={need}",)))
        path = _text(select(entry, "model_path"))
        if path is None or not os.path.isdir(path):
            continue
        try:
            positions = _model_config_value(_read_json(os.path.join(path, "config.json")), "max_position_embeddings")
        except (OSError, ValueError):
            continue   # check_teacher_tokenizers reports an unreadable config.json
        context = need if max_model_len is None else int(max_model_len)
        if positions is not None and context > int(positions):
            problems.append(Problem(
                "distill", f"The teacher {name} needs a context of {context} tokens, but its config.json has "
                f"max_position_embeddings {positions}, and verl refuses a longer max_model_len when the "
                "teacher's server starts. Shorten the student's responses, or use a teacher with a longer context.",
                (f"data.max_response_length={max(1, int(positions) - prompt - 1)}",)))
    return problems


def check_think_budget(cfg: Any) -> list[Problem]:
    """A thinking budget leaves room in the response length for the stop-thinking text and the answer.

    The budget is ``orbit.env.ORBIT_THINK_BUDGET``. The launcher copies that block into the env of
    every Ray worker, so the config is where the check reads it. A training response of a thinking
    mode first generates at most the budget. When its thinking block is still open then, the agent
    loop inserts the stop-thinking text and lets the model answer in what is left of the response
    length (``orbit_verl.data.agent_loop``).

    The agent loop refuses a budget that leaves the answer fewer than ``MIN_ANSWER_TOKENS`` tokens.
    It does so only when it builds the loop of the first rollout, after Ray and the rollout servers
    have started, so this check refuses such a budget before any of that. The launcher runs it for
    every config (``launch.cli.run_checks``), because RL runs can set the budget too. It counts the
    stop-thinking text as ``STOP_THINKING_MAX_IDS`` ids, because the exact count needs the
    tokenizer. A budget within about 40 tokens of the agent loop's limit is therefore refused here
    although the agent loop would accept it. A value that is not a positive whole number is refused
    here too, for the same reason as above.
    The response length is the rollout's, which the agent loop reads, and which defaults to
    ``data.max_response_length``.
    """
    raw = select(cfg, _THINK_BUDGET)
    try:
        # The launcher writes a null value as an empty string, which means no budget.
        budget = think_budget({THINK_BUDGET_ENV: "" if raw is None else str(raw)})
    except ValueError:
        return [Problem(
            "config", f"{_THINK_BUDGET} is {raw!r}, which is not a positive whole number of tokens. The agent loop "
            "would refuse it on the first rollout, after Ray and the rollout servers have started. Give the "
            "budget in tokens, or null for no budget.")]
    if budget is None:
        return []
    _, response = student_lengths(cfg)
    if budget + STOP_THINKING_MAX_IDS + MIN_ANSWER_TOKENS <= response:
        return []
    largest = response - STOP_THINKING_MAX_IDS - MIN_ANSWER_TOKENS
    return [Problem(
        "config", f"{_THINK_BUDGET} is {budget}, which leaves {response - budget} of the {response} tokens of the "
        f"response length after the thinking. The stop-thinking text takes up to {STOP_THINKING_MAX_IDS} of them, "
        f"and the answer after it needs at least {MIN_ANSWER_TOKENS}, or every forced answer would be cut. The agent "
        "loop checks the same with the exact count of those ids, but only on the first rollout, after Ray and the "
        "rollout servers have started. Lower the budget, or raise data.max_response_length.",
        (f"{_THINK_BUDGET}={largest}",) if largest >= 1 else (f"{_THINK_BUDGET}=null",))]


def check_verl_config(cfg: Any) -> list[Problem]:
    """Build verl's own ``DistillationConfig`` from the block, the way the teacher manager does at start-up.

    That catches what verl checks only after Ray is up: a teacher pool that the teachers' tensor
    parallel sizes do not divide, a k1 loss without the policy gradient, a top-k loss without k,
    and a policy loss mode other than vanilla inside the distillation loss.
    """
    try:
        from verl.utils.config import omega_conf_to_dataclass

        omega_conf_to_dataclass(cfg.distillation)
    except Exception as exc:  # noqa: BLE001 - verl raises several types for a bad block
        return [Problem("distill", f"verl refuses the distillation block: {type(exc).__name__}: {exc}")]
    return []


def run_preflight(cfg: Any, *, verl_check: bool = True,
                  hub_dirs: Iterable[str] | None = None) -> tuple[list[Problem], list[str]]:
    """All checks of a distillation run. Returns the problems that stop the run and the warnings.

    verl's own check runs last and only when nothing else was found. Its errors repeat the earlier
    findings in verl's words, without the fix, and an unset model path would make it fail on a
    detail. ``verl_check=False`` leaves it out, for tests without verl.
    """
    from orbit_verl.distill.teacher import teacher_prompt_problems

    problems = check_teacher_paths(cfg, hub_dirs)
    found, warnings = check_teacher_tokenizers(cfg)
    problems += found
    problems += check_temperature(cfg)
    problems += check_loss_mode(cfg)
    problems += check_kl_loss(cfg)
    problems += check_teacher_lengths(cfg)
    problems += check_teacher_routing(cfg)
    # The teacher module words its own findings; each already names the key and the value to set.
    problems += [Problem("distill", message) for message in teacher_prompt_problems(cfg)]
    warnings += kl_loss_warnings(cfg)
    if verl_check and not problems:
        problems += check_verl_config(cfg)
    return problems, warnings
