"""Runs the lm-eval command line for the in-training evaluation, and always exits.

The trainer (src/orbit/in_training_eval) starts this script with the evaluation
interpreter (eval_python), one process per group of work units on one GPU. The arguments are the
usual lm-eval ones. This script must not import the trainer package: it runs in a different
environment, with a different torch.

The script is written for the public lm-eval release that harness.LM_EVAL_VERSION pins, and it
calls that release's command line without patching anything in it. The trainer's startup check
refuses any other lm-eval. The same check imports lm-eval's vLLM backend, so a vLLM release whose
modules the backend cannot import stops the run at startup rather than at the first evaluation.
Four things happen around lm-eval:

* A watchdog on the trainer. The trainer starts this script in its own session, so that it can
  kill the whole process group, vLLM's EngineCore included. The flip side is that nothing kills
  the group when the trainer itself dies without cleaning up (SIGKILL, OOM killer, a segfault,
  NCCL's watchdog abort): the launcher only kills the trainer's own group. The EngineCore would
  then keep about 75% of the GPU and break the next attempt on that GPU. So a thread here
  checks every second that the trainer is still this process's parent, and kills the whole
  group when it is not.
* A guard against a broken optional package. vLLM uses flash_attn for its rotary kernel whenever
  the package is installed. An environment may have a flash_attn binary built for another torch,
  which fails to import and kills the engine. When flash_attn is installed but does not
  import, it is hidden, so vLLM falls back to its own kernel.
* The environment's own libstdc++, loaded before anything else. When the evaluation environment
  ships a newer libstdc++ than the system, the system's older copy would otherwise serve the whole
  process, and RULER's imports need the newer one.
  This is the second of two guards: the trainer already preloads the same file through
  LD_PRELOAD (environment.build_eval_env), and the load here covers a process started without
  the trainer's environment. Where the trainer decided against the preload, because the system
  has a newer copy, this load is skipped as well (see the comment where it is loaded).
* A forced exit. Once the vLLM engine is up, an exception anywhere in lm-eval can leave the
  process hanging in vLLM's shutdown with the GPU still held. Every outcome here ends in os._exit
  with a real exit code, so the trainer sees the process end, cleans up the engine's child
  processes, and gives the GPU back to training.

Processes of the SFT (chat) evaluation run "--model vllm_itr_chat". Only for them, the script imports
itr_chat.py next to it, which registers that model with lm-eval, and after lm-eval has finished it lets
itr_chat write the per-response records of the gates. The "--model vllm" path never imports it.
"""
import ctypes
import os
import signal
import sys
import threading
import time
import traceback

# The trainer sets this variable when it leaves this environment's libstdc++ out of LD_PRELOAD on
# purpose, and the value says why (environment.EVAL_NO_LIBSTDCXX_ENV).
NO_LIBSTDCXX_ENV = "ORBIT_EVAL_NO_ENV_LIBSTDCXX"

# When this environment ships a newer libstdc++ than the system, the system's older copy can get
# loaded ahead of the environment's newer one, and the copy a process loads first serves every
# library in it. vLLM and torch load it early. A later import that needs the newer copy then
# fails: nltk imports sqlite3, whose ICU library needs CXXABI_1.3.15, so every RULER task fails.
# Loading the environment's own libstdc++ first, by its full path, makes every later library use
# it, and the newer copy also serves the libraries built against the older one. vLLM's engine
# is a spawned child that runs this file's top-level code too, so it gets the same copy.
#
# The trainer normally preloads the same file through LD_PRELOAD, and then this load changes
# nothing. It is kept as a second guard for a process that did not get the trainer's
# environment. When the trainer left the copy out because the system has a newer one, the copy
# is not loaded here either. If the load fails, the process keeps the copy that the linker finds.
# Then only the evaluations that need the newer copy fail, rather than every evaluation.
_ENV_LIBSTDCXX = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
if os.path.exists(_ENV_LIBSTDCXX) and NO_LIBSTDCXX_ENV not in os.environ:
    try:
        ctypes.CDLL(_ENV_LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        print(f"run_lm_eval: cannot load {_ENV_LIBSTDCXX} ({exc}); the process keeps the libstdc++ "
              "that the linker finds", flush=True)

# Packages hidden in this process and in the processes it spawns. vLLM starts its engine with the
# multiprocessing spawn method, and a spawned child runs this file's top-level code again (as
# __mp_main__), so the hiding below reaches the engine through this variable.
HIDDEN_MODULES_ENV = "ORBIT_EVAL_HIDDEN_MODULES"
# A None entry in sys.modules is Python's own way to make a package unimportable: find_spec then
# returns None and import raises ImportError, which is exactly "not installed" for vLLM.
for _name in filter(None, os.environ.get(HIDDEN_MODULES_ENV, "").split(",")):
    sys.modules.setdefault(_name, None)


# The trainer puts its PID here (environment.EVAL_PARENT_ENV).
PARENT_PID_ENV = "ORBIT_EVAL_PARENT_PID"
# The lm-eval model of chat processes (benchmarks.CHAT_MODEL), registered by itr_chat.py.
CHAT_MODEL = "vllm_itr_chat"


def chat_requested(argv):
    """Whether the command line asks for the chat model, which itr_chat.py has to register first."""
    return "--model" in argv[:-1] and argv[argv.index("--model") + 1] == CHAT_MODEL


def start_parent_watchdog(interval=1.0):
    """Kills this process group as soon as the trainer that started this process is gone.

    When the trainer dies, this process is re-parented, so os.getppid() stops being the trainer's
    PID. The group is killed only when this process leads it, which is how the trainer starts it
    (start_new_session), so a manual run from a shell never kills the shell's group. Returns the
    thread, or None when no trainer PID was given.
    """
    expected = os.environ.get(PARENT_PID_ENV, "")
    if not expected.isdigit():
        return None
    parent = int(expected)

    def watch():
        while os.getppid() == parent:
            time.sleep(interval)
        print(f"run_lm_eval: the trainer (pid {parent}) is gone; stopping this evaluation", flush=True)
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(1)

    thread = threading.Thread(target=watch, name="trainer_watchdog", daemon=True)
    thread.start()
    return thread


def hide_if_broken(name):
    """Hides an installed package that fails to import, for this process and its children."""
    import importlib.util

    try:
        if importlib.util.find_spec(name) is None:
            return
        __import__(name)
    except ImportError as exc:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
        sys.modules[name] = None
        hidden = [x for x in os.environ.get(HIDDEN_MODULES_ENV, "").split(",") if x]
        os.environ[HIDDEN_MODULES_ENV] = ",".join(hidden + [name])
        print(f"run_lm_eval: {name} is installed but does not import ({exc}); it is hidden", flush=True)


def main():
    rc = 0
    chat = None
    try:
        start_parent_watchdog()
        hide_if_broken("flash_attn")
        if chat_requested(sys.argv[1:]):
            # Python puts this script's folder first on the path; the line covers a start without it.
            here = os.path.dirname(os.path.abspath(__file__))
            if here not in sys.path:
                sys.path.append(here)
            import itr_chat as chat

            chat.configure(sys.argv[1:])
        from lm_eval.__main__ import cli_evaluate

        cli_evaluate()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc()
        rc = 1
    if chat is not None and rc == 0:
        # The records are built from the files lm-eval has just written. finish never raises, and a
        # failure there leaves the exit code alone: the accuracy stays reportable, the gates do not.
        chat.finish(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
