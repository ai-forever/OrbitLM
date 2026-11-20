"""The lm-eval model of the SFT (chat) evaluation: lm-eval's vLLM backend with the chat rules on top.

run_lm_eval.py imports this module only for "--model vllm_itr_chat", which the trainer puts on the
command line of every process that runs chat benchmarks. Importing it registers the model under that
name; nothing in lm-eval is changed. The subclass does what the stock backend of lm-eval 0.4.13 cannot
do safely for a hybrid think/no_think model:

* It renders every prompt in the process's mode with an explicit enable_thinking, applies the
  leading-token policy, and builds the no-prefill prompt (itr_render.py).
* It checks every prompt's ids against the token contract (README.md) and fails the
  process on the first mismatch, so a benchmark is never reported under the wrong mode.
* It writes a sidecar with the generated ids, the finish reason and the stop reason of every response,
  keyed by (task, document, repeat). lm-eval's saved samples keep only the text.
* It turns the suite's repetition_detection mapping into vLLM's parameter object.

After lm-eval has finished, write_records joins the sidecar with lm-eval's saved samples and writes one
compact record per response (itr_records.jsonl), which the trainer turns into the gates. README.md in
this folder describes the protocol.
"""
from __future__ import annotations

import collections
import functools
import gzip
import importlib.util
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Optional

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.utils import hash_string

import itr_render

try:
    from vllm.sampling_params import RepetitionDetectionParams
except ImportError:  # vLLM releases before repetition detection; a suite that asks for it then fails
    RepetitionDetectionParams = None

MODEL_NAME = "vllm_itr_chat"
HERE = Path(__file__).resolve().parent
# configure() copies lm-eval's --output_path here before lm-eval starts, so that the subclass writes its
# sidecar next to lm-eval's results. It copies --include_path to the second variable, so that the
# subclass and the records writer load the same chat_local.py as the task files.
OUT_ENV = "ORBIT_ITR_OUT"
TASKS_ENV = "ORBIT_ITR_TASKS"
SIDECAR_NAME = "itr_sidecar.jsonl"
RECORDS_NAME = "itr_records.jsonl"
RECORDS_ERROR_NAME = "itr_records_error.json"
MODES = ("think", "no_think", "no_prefill")


@functools.lru_cache(maxsize=None)
def load_chat_local(tasks_dir: str) -> Any:
    """tasks/chat_local.py, loaded by its path, as lm-eval loads the helpers of a task file."""
    path = Path(tasks_dir) / "chat_local.py"
    spec = importlib.util.spec_from_file_location("itr_chat_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chat_local() -> Any:
    return load_chat_local(os.environ.get(TASKS_ENV) or str(HERE / "tasks"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


@register_model(MODEL_NAME)
class VLLMITRChat(VLLM):
    """lm-eval's vLLM backend for SFT checkpoints; the module docstring lists what it adds."""

    def __init__(self, *args: Any, itr_mode: str = "think", itr_lead_eos: Any = True, **kwargs: Any) -> None:
        if itr_mode not in MODES:
            raise ValueError(f"itr_mode must be one of {MODES}, got {itr_mode!r}")
        self.itr_mode = itr_mode
        self.itr_lead_eos = _as_bool(itr_lead_eos)
        # chat_local's scorers read the mode, because the text of a think response that never closed
        # its block looks like a no_think response.
        os.environ[chat_local().MODE_ENV] = itr_mode
        super().__init__(*args, **kwargs)
        self._contract = itr_render.expected_ids(self.tokenizer)
        self._vocabulary = chat_local().vocabulary_ids(self.tokenizer)
        self._announced: dict[tuple[int, ...], collections.deque] = {}
        out = os.environ.get(OUT_ENV)
        self._sidecar = Path(out) / SIDECAR_NAME if out else None
        self._blank_ids: dict[int, bool] = {}

    # -- prompts --------------------------------------------------------------------------------------

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        if not add_generation_prompt:
            raise ValueError("the chat tasks of the SFT evaluation never continue an assistant message (gen_prefix)")
        return itr_render.render(
            self.tokenizer, chat_history, self.itr_mode, self.itr_lead_eos,
            chat_template=self.hf_chat_template, template_args=self.chat_template_args,
        )

    def _is_blank(self, token: int) -> bool:
        """Whether a token decodes to whitespace only; the reasoning length is counted without such tokens."""
        blank = self._blank_ids.get(token)
        if blank is None:
            blank = self.tokenizer.decode([int(token)]).strip() == ""
            self._blank_ids[token] = blank
        return blank

    # -- generation -----------------------------------------------------------------------------------

    def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
        """Announces every request's key before lm-eval's backend batches them.

        The backend sorts the requests by length before it calls _model_generate, which only sees token
        ids. Each request's ids are queued with its (task, document, repeat) key here, in the original
        order. The sort is stable, so identical prompts (the repeats of one document) reach
        _model_generate in that same order, and each one takes the next key of its queue.
        """
        contexts = [request.args[0] for request in requests]
        encoded = self.tok_encode(contexts)
        counts: collections.Counter = collections.Counter()
        self._announced = {}
        for request, ids in zip(requests, encoded):
            key = (request.task_name, request.doc_id)
            repeat = counts[key]
            counts[key] += 1
            self._announced.setdefault(tuple(ids), collections.deque()).append({
                "task": request.task_name, "doc_index": request.doc_id, "repeat": repeat,
                "prompt_hash": hash_string(request.args[0]),
            })
        return super().generate_until(requests, disable_tqdm=disable_tqdm)

    @staticmethod
    def modify_gen_kwargs(gen_kwargs: dict[str, Any], eos: Any = None, default_max_gen_toks: int = 256):
        """lm-eval's processing, plus the suite's repetition_detection mapping as vLLM's object."""
        gen_kwargs = dict(gen_kwargs)
        detection = gen_kwargs.pop("repetition_detection", None)
        kwargs, until, max_gen_toks = VLLM.modify_gen_kwargs(gen_kwargs, eos=eos, default_max_gen_toks=default_max_gen_toks)
        if detection:
            if RepetitionDetectionParams is None:
                raise RuntimeError("this vLLM has no repetition detection, but the suite asks for it")
            kwargs["repetition_detection"] = RepetitionDetectionParams(**detection)
        return kwargs, until, max_gen_toks

    def _model_generate(self, requests: list[list[int]], generate: bool = False, sampling_params: Any = None):
        if not generate:
            return super()._model_generate(requests, generate=generate, sampling_params=sampling_params)
        # vLLM takes one parameter object per request, so a single shared object is spread out here.
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(requests)
        keys = []
        for ids in requests:
            problem = itr_render.contract_problem(list(ids), self.itr_mode, self.itr_lead_eos, self._contract)
            if problem:
                raise itr_render.ContractError(
                    f"{problem}; the prompt was {self.tokenizer.decode(list(ids))[-400:]!r}"
                )
            queue = self._announced.get(tuple(ids))
            if not queue:
                # lm-eval shortens a prompt that does not fit next to the budget; the suite's window
                # check should make that impossible, and a cut prompt must not be scored.
                raise RuntimeError("a prompt reached the engine in another form than announced (cut to fit the window?)")
            keys.append(queue.popleft())
        outputs = super()._model_generate(requests, generate=True, sampling_params=params)
        self._write_sidecar(keys, requests, outputs)
        return outputs

    def _write_sidecar(self, keys: list[dict], requests: list[list[int]], outputs: list[Any]) -> None:
        if self._sidecar is None:
            return
        local = chat_local()
        self._sidecar.parent.mkdir(parents=True, exist_ok=True)
        with self._sidecar.open("a", encoding="utf-8") as handle:
            for key, ids, output in zip(keys, requests, outputs):
                completion = output.outputs[0]
                generated = [int(token) for token in completion.token_ids]
                finish = None if completion.finish_reason is None else str(completion.finish_reason)
                facts = local.token_facts(generated, self.itr_mode, finish, self._is_blank, self._vocabulary)
                stop_reason = completion.stop_reason
                line = {**key, "mode": self.itr_mode, "n_prompt": len(ids), "stop_reason": stop_reason, **facts,
                        "gen_ids": generated}
                handle.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# Records for the gates
# ---------------------------------------------------------------------------


def _argument(argv: list[str], name: str) -> Optional[str]:
    if name in argv:
        position = argv.index(name)
        if position + 1 < len(argv):
            return argv[position + 1]
    return None


def _task_of(samples_file: Path) -> str:
    """The task of lm-eval's samples_<task>_<date>.jsonl; the date part has no underscore."""
    stem = samples_file.name[len("samples_"):-len(".jsonl")]
    return stem[: stem.rfind("_")]


def write_records(out_dir: Path, samples_path: Optional[Path], mode: str) -> int:
    """Joins the sidecar with lm-eval's saved samples and writes one record per response.

    A record holds the response's key, its token facts (from the sidecar), its text facts (markers,
    opener, answer language) and its correctness, scored by the same chat_local functions as the
    task. Every sample must find its sidecar entry and the other way round, and lm-eval's prompt hash
    must match, or nothing is written: a partial record set would bias the gates. Returns the number
    of records.
    """
    local = chat_local()
    out_dir = Path(out_dir)
    samples_files = sorted(out_dir.glob("**/samples_*.jsonl"))
    if not samples_files:
        raise RuntimeError(f"no samples found under {out_dir}")
    documents = json.loads(Path(samples_path).read_text(encoding="utf-8")) if samples_path else {}
    sidecar: dict[tuple[str, int, int], dict] = {}
    for line in (out_dir / SIDECAR_NAME).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        index = int(entry["doc_index"])
        task = entry["task"]
        # With --samples, lm-eval numbers the documents of a shard 0..k-1 internally and saves them
        # under their indices in the whole test set.
        doc_id = documents[task][index] if task in documents else index
        sidecar[(task, doc_id, int(entry["repeat"]))] = entry
    records = []
    used = set()
    for samples_file in samples_files:
        task = _task_of(samples_file)
        for line in samples_file.read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            doc = sample["doc"]
            base = {"task": task, "doc_id": int(sample["doc_id"]), "mode": mode,
                    "lang_expected": doc.get("itr_lang") or None, "domain": doc.get("itr_domain") or None}
            texts = sample["resps"][0]
            for repeat, text in enumerate(texts):
                key = (task, int(sample["doc_id"]), repeat)
                entry = sidecar.get(key)
                if entry is None:
                    raise RuntimeError(f"no sidecar entry for {key}")
                if entry["prompt_hash"] != sample["prompt_hash"]:
                    raise RuntimeError(f"the sidecar entry of {key} belongs to another prompt")
                used.add(key)
                facts = {name: value for name, value in entry.items()
                         if name not in ("gen_ids", "task", "doc_index", "repeat", "prompt_hash", "mode")}
                # The no-prefill probe writes 64 tokens, which are never an answer to score.
                correct = None if mode == "no_prefill" else local.score_response(task, doc, text, mode)
                records.append({**base, "repeat": repeat, **facts, **local.text_facts(text, mode), "correct": correct})
    if len(used) != len(sidecar):
        missing = sorted(set(sidecar) - used)[:3]
        raise RuntimeError(f"{len(sidecar) - len(used)} generated responses have no saved sample, for example {missing}")
    if not records:
        raise RuntimeError(f"the samples under {out_dir} hold no responses")
    target = out_dir / RECORDS_NAME
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staging.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    os.replace(staging, target)
    return len(records)


def _gzip_in_place(path: Path) -> None:
    with path.open("rb") as source, gzip.open(path.with_name(path.name + ".gz"), "wb") as target:
        shutil.copyfileobj(source, target)
    path.unlink()


def finish(argv: list[str]) -> None:
    """Writes the records after lm-eval returned, and compresses the large files. Never raises.

    A failure leaves itr_records_error.json instead of the records. The process keeps lm-eval's exit
    code, so its accuracy is still reported; only the gates that need its records are withheld.
    """
    out = _argument(argv, "--output_path")
    if not out:
        return
    out_dir = Path(out)
    try:
        samples = _argument(argv, "--samples")
        count = write_records(out_dir, Path(samples) if samples else None, os.environ.get(chat_local().MODE_ENV, "think"))
        print(f"itr_chat: wrote {count} records to {out_dir / RECORDS_NAME}", flush=True)
    except BaseException:
        error = traceback.format_exc()
        print(f"itr_chat: cannot write the records:\n{error}", flush=True)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / RECORDS_ERROR_NAME).write_text(json.dumps({"error": error}), encoding="utf-8")
        except OSError:
            pass
        return
    # The samples and the sidecar hold every response in full (up to about 100 MB per pause); the
    # records hold what the gates need, so the rest is compressed for later calibration work.
    for path in [out_dir / SIDECAR_NAME, *out_dir.glob("**/samples_*.jsonl")]:
        try:
            if path.is_file():
                _gzip_in_place(path)
        except OSError as exc:
            print(f"itr_chat: cannot compress {path}: {exc}", flush=True)


def configure(argv: list[str]) -> None:
    """Tells the subclass where lm-eval writes its outputs and where the task helpers live."""
    out = _argument(argv, "--output_path")
    if out:
        os.environ[OUT_ENV] = str(Path(out).resolve())
    include = _argument(argv, "--include_path")
    if include:
        os.environ[TASKS_ENV] = str(Path(include).resolve())
    # A sidecar from an earlier run of this process folder would be joined with the new samples.
    if out and (Path(out) / SIDECAR_NAME).exists():
        (Path(out) / SIDECAR_NAME).unlink()


__all__ = ["VLLMITRChat", "configure", "finish", "write_records"]
