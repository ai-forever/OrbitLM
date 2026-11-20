"""Runs the build of a stage on one machine: every task, split into parts, on all free CPUs.

Without --tasks it runs the stage's own task list (stage_tasks below): the stage's BUILD_TASKS,
then pack, merge and check for every stream in the stage's STREAMS. With --tasks it runs a task
list from a JSON file instead, for example to fan a builder out over many processes.

A task is one command, optionally split into parts. Each part is a separate process that gets
--part/--nparts, and a task may depend on other tasks. The runner keeps every CPU busy with
ready parts and starts a task only after the tasks it needs have fully finished.

It is safe to stop and restart. A part that finished leaves a marker in the stage's
RUN_LOG_DIR, and a restart skips it. When a task runs again, every task that needs it, directly
or through other tasks, runs again too, even if it has markers from an earlier run. For
example, a stream that is packed again is also merged and checked again. Those older markers
are removed as soon as a part they depend on finishes, so a restart in the middle still reruns
them. A part that failed is retried once; if it fails again, the tasks that depend on it are
skipped and everything else keeps going. A status file in RUN_LOG_DIR is rewritten every ten
seconds, and --status prints it.

Task format, as in a --tasks file ({"tasks": [...]}):
  {"name": "pack_x", "cmd": "{py} pack.py ... --part {part} --nparts {nparts}", "nparts": 4,
   "needs": ["build_x"], "clean": ["streams/x/part_{part2}"]}
In "cmd", {py} is this interpreter and {root} is the stage root. "clean" lists paths under the
stage root that are removed before a part starts, so a part that died halfway starts clean;
{part2} is the part number with two digits.

Usage:
  run_tasks.py --stage 32k [--tasks FILE.json] [--workers N] [--mem-gb 24] [--only a,b] [--dry-run]
  run_tasks.py --stage 32k --status
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import stages

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
# The packages that the stage tasks import. The preflight imports them once, so a missing package
# stops the build before any task starts instead of failing every part of it.
REQUIRED_MODULES = "numpy, pyarrow, tokenizers, streaming, zstandard"


def stage_tasks(stage):
    """Returns the stage's task list: its build tasks, then pack, merge and check for every stream."""
    tasks = [dict(t) for t in stage.BUILD_TASKS]
    for name, (nparts, args) in stage.STREAMS.items():
        pack = {"name": f"pack_{name}", "nparts": nparts, "clean": [f"streams/{name}/part_{{part2}}"],
                "cmd": f"{{py}} pack.py pack --stage {stage.NAME} --name {name} {args} --part {{part}} --nparts {{nparts}}"}
        if name in stage.STREAM_NEEDS:
            pack["needs"] = list(stage.STREAM_NEEDS[name])
        tasks += [
            pack,
            {"name": f"merge_{name}", "needs": [f"pack_{name}"],
             "cmd": f"{{py}} pack.py merge --stage {stage.NAME} --name {name} --nparts {nparts}"},
            {"name": f"check_{name}", "needs": [f"merge_{name}"],
             "cmd": f"{{py}} check_stream.py --stage {stage.NAME} {{root}}/streams/{name}"},
        ]
    return tasks


def dependents(tasks, names):
    """Returns the names of every task that needs one of these tasks, directly or through other tasks."""
    found, frontier = set(), set(names)
    while frontier:
        frontier = {t["name"] for t in tasks if frontier & set(t.get("needs", []))} - found
        found |= frontier
    return found


def cpu_count():
    """Returns the CPUs this container may really use; nproc can report the whole host instead."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except OSError:
        pass
    # sched_getaffinity exists only on Linux. Elsewhere, such as on macOS, the process may use
    # every CPU that the system reports.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def safe_clean(root, path):
    """Removes a leftover output before a part is (re)started; refuses anything outside the stage root."""
    full = os.path.realpath(os.path.join(root, path))
    if not full.startswith(os.path.realpath(root) + "/"):
        raise ValueError(f"refusing to clean outside the stage root: {full}")
    if os.path.isdir(full):
        shutil.rmtree(full)
    elif os.path.exists(full):
        os.remove(full)


def preflight(stage, tasks):
    """Returns the problems that would make the build fail, so it can stop before starting anything."""
    problems = [f"missing: {p}" for p in (PY, *stage.PREFLIGHT_PATHS) if not os.path.exists(p)]
    r = subprocess.run([PY, "-B", "-c", f"import {REQUIRED_MODULES}"], capture_output=True, text=True)
    if r.returncode:
        problems.append("the python env cannot import the needed packages: " + r.stderr.strip()[-300:])
    if os.path.exists(stage.ROOT):
        free_tb = shutil.disk_usage(stage.ROOT).free / 1e12
        if free_tb < stage.MIN_FREE_TB:
            problems.append(f"only {free_tb:.1f} TB free under {stage.ROOT}; the build wants {stage.MIN_FREE_TB} TB")
    for t in tasks:
        script = next((w for w in t["cmd"].split() if w.endswith(".py")), None)
        if script and not os.path.isabs(script) and not os.path.exists(os.path.join(CODE_DIR, script)):
            problems.append(f"task {t['name']}: script not found: {script}")
    return problems


def expand(stage, tasks):
    """Turns tasks into (task, part) units with their final commands."""
    units = []
    for t in tasks:
        n = int(t.get("nparts", 1))
        for p in range(n):
            cmd = t["cmd"].format(py=f"{PY} -B -u", part=p, nparts=n, root=stage.ROOT)
            units.append({"task": t["name"], "part": p, "nparts": n, "cmd": cmd,
                          "clean": [c.format(part=p, part2=f"{p:02d}") for c in t.get("clean", [])]})
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages.add_stage_argument(ap)
    ap.add_argument("--tasks", default="", help="a JSON task list to run instead of the stage's own tasks")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: CPUs minus two)")
    ap.add_argument("--mem-gb", type=int, default=24, help="address-space cap per process")
    ap.add_argument("--only", default="", help="comma-separated task names to run")
    ap.add_argument("--dry-run", action="store_true", help="print the parts that would run, and stop")
    ap.add_argument("--status", action="store_true", help="print the status file of the last run")
    a = ap.parse_args()
    stage = stages.get(a.stage)
    log_dir = stage.RUN_LOG_DIR
    status_name = os.path.splitext(os.path.basename(a.tasks))[0] if a.tasks else stage.NAME
    status_path = os.path.join(log_dir, f"status_{status_name}.json")
    if a.status:
        with open(status_path) as f:
            print(f.read())
        return

    if a.tasks:
        with open(a.tasks) as f:
            all_tasks = json.load(f)["tasks"]
    else:
        all_tasks = stage_tasks(stage)
    tasks = all_tasks
    if a.only:
        keep = set(a.only.split(","))
        tasks = [t for t in all_tasks if t["name"] in keep]
    by_name = {t["name"]: t for t in tasks}
    workers = a.workers or max(1, cpu_count() - 2)

    # The stage root is where this build writes, so the runner creates it. On a fresh machine
    # nothing else creates the 32K root before its first task runs. The free-space check of the
    # preflight also needs the directory to exist.
    os.makedirs(stage.ROOT, exist_ok=True)
    problems = preflight(stage, tasks)
    if problems:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(problems))
        sys.exit(2)
    os.makedirs(log_dir, exist_ok=True)

    def marker(task, part):
        return os.path.join(log_dir, f"{task}.p{part:03d}.done")

    def has_marker(unit):
        return os.path.exists(marker(unit["task"], unit["part"]))

    def task_done(name):
        """A task is done when none of its parts is waiting or running and every part has a marker."""
        t = by_name.get(name)
        if t is None:  # a dependency outside --only is assumed to be finished already
            return True
        if any(u["task"] == name for u in pending) or any(u["task"] == name for _, u, _, _ in running.values()):
            return False
        return all(os.path.exists(marker(name, p)) for p in range(int(t.get("nparts", 1))))

    nparts_of = {t["name"]: int(t.get("nparts", 1)) for t in all_tasks}

    def remove_markers(name):
        for p in range(nparts_of[name]):
            if os.path.exists(marker(name, p)):
                os.remove(marker(name, p))

    units = expand(stage, tasks)
    # A marker says that a part finished, given the outputs that its task needed at that time.
    # When a task has parts left to run, those outputs will change, so every task downstream of
    # it is out of date and runs again, even when it has markers from an earlier run.
    outdated = dependents(all_tasks, {u["task"] for u in units if not has_marker(u)})
    pending = [u for u in units if not has_marker(u) or u["task"] in outdated]
    running = {}
    print(f"{len(units)} parts in {len(tasks)} tasks, {len(units) - len(pending)} already done, {workers} workers, "
          f"{a.mem_gb} GB address-space cap per process", flush=True)
    rerun_with_markers = sorted({u["task"] for u in pending if has_marker(u)})
    if rerun_with_markers:
        print(f"these tasks have markers but run again, because a task they need runs again: "
              f"{', '.join(rerun_with_markers)}", flush=True)
    if a.dry_run:
        for u in pending:
            print(f"  {u['task']} p{u['part']}: {u['cmd']}")
        return

    failed_tasks, attempts = set(), {}
    env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=CODE_DIR)
    start = time.time()
    last_status = 0
    while pending or running:
        # Start ready parts while there are free workers.
        for u in list(pending):
            if len(running) >= workers:
                break
            needs = by_name[u["task"]].get("needs", [])
            if any(n in failed_tasks for n in needs):
                pending.remove(u)
                failed_tasks.add(u["task"])
                print(f"SKIP {u['task']} p{u['part']}: a dependency failed", flush=True)
                continue
            if not all(task_done(n) for n in needs):
                continue
            for c in u["clean"]:
                safe_clean(stage.ROOT, c)
            log = open(os.path.join(log_dir, f"{u['task']}.p{u['part']:03d}.log"), "a")
            log.write(f"\n==== {time.strftime('%F %T')} start: {u['cmd']}\n")
            log.flush()
            # The address-space cap makes a runaway process fail on its own instead of exhausting
            # the memory of the whole machine.
            capped = f"ulimit -v {a.mem_gb * 1000000}; exec nice -n 5 {u['cmd']}"
            proc = subprocess.Popen(["bash", "-c", capped], cwd=CODE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[proc.pid] = (proc, u, log, time.time())
            pending.remove(u)

        # Collect finished parts.
        for pid, (proc, u, log, started) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[pid]
            key = (u["task"], u["part"])
            if rc == 0:
                with open(marker(*key), "w") as f:
                    f.write(f"{time.strftime('%F %T')} {time.time() - started:.0f}s\n")
                # Any markers of the tasks downstream of this one describe runs on its older output.
                # They are removed now, so that a restart from this point still reruns those tasks.
                for name in dependents(all_tasks, {u["task"]}):
                    remove_markers(name)
                print(f"done {u['task']} p{u['part']} in {(time.time() - started) / 60:.1f} min", flush=True)
            else:
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] < 2:
                    print(f"RETRY {u['task']} p{u['part']} (exit {rc})", flush=True)
                    pending.insert(0, u)
                else:
                    failed_tasks.add(u["task"])
                    print(f"FAILED {u['task']} p{u['part']} (exit {rc}); see {log_dir}/{u['task']}.p{u['part']:03d}.log",
                          flush=True)

        if time.time() - last_status > 10:
            last_status = time.time()
            status = {}
            for t in tasks:
                n = int(t.get("nparts", 1))
                active = {u["part"] for (_, u, _, _) in running.values() if u["task"] == t["name"]}
                waiting = {u["part"] for u in pending if u["task"] == t["name"]}
                # A part that waits to run again is not counted as done, even while its old marker exists.
                done = sum(os.path.exists(marker(t["name"], p)) for p in range(n) if p not in active | waiting)
                status[t["name"]] = {"done": done, "running": len(active), "parts": n,
                                     "failed": t["name"] in failed_tasks}
            with open(status_path, "w") as f:
                json.dump({"time": time.strftime("%F %T"), "elapsed_min": round((time.time() - start) / 60, 1),
                           "tasks": status}, f, indent=1)
        time.sleep(1)

    with open(status_path, "w") as f:
        json.dump({"time": time.strftime("%F %T"), "finished": True, "failed": sorted(failed_tasks)}, f, indent=1)
    print(f"FINISHED in {(time.time() - start) / 60:.0f} min; failed tasks: {sorted(failed_tasks) or 'none'}", flush=True)
    sys.exit(1 if failed_tasks else 0)


if __name__ == "__main__":
    main()
