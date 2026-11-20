"""Downloads Hugging Face sources at pinned revisions into $ORBIT_RAW/hf.

Each repo goes to $ORBIT_RAW/hf/<org>__<name>/ at the revision that the plan pins. The default
plan, download_plan.json next to this script, holds the sources of the 16K stage;
data_prep/sft/download_plan.json holds those of the SFT mix. A plan entry may carry a "note"
for the reader, which this script does not use. The script can be re-run at any
time: files that are already complete are skipped. When it finishes,
$ORBIT_RAW/hf/downloads.<directory of the plan>.json (downloads.long_context.json or
downloads.sft.json; downloads.sft.v7_v9.json for data_prep/sft/download_plan_v7_v9.json)
records every file with its size and the commit it came from, so the data can be rebuilt from
exactly the same inputs.

Usage: download_hf.py [--plan PLAN.json] [--workers N]
"""
import argparse
import fnmatch
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import paths

os.environ.setdefault("HF_HOME", os.path.join(paths.RAW, ".hf_home"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402  (env must be set first)

# The token raises the rate limits for public repos, and gated repos such as Big-Math cannot be
# downloaded without it. It is read from the file that `huggingface-cli login` writes and is
# never printed. Without that file, huggingface_hub falls back to the HF_TOKEN variable.
TOKEN_FILE = os.path.expanduser("~/.cache/huggingface/token")
TOKEN = open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else None
api = HfApi(token=TOKEN)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def resolve(entry):
    """Turns one plan entry into (repo, commit sha, [(file, size)]), expanding patterns if needed."""
    repo, rev = entry["repo"], entry["revision"]
    info = api.dataset_info(repo, revision=rev, files_metadata=True)
    sha = info.sha
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    if "files" in entry:
        names = entry["files"]
    else:
        names = [f for f in sizes if any(fnmatch.fnmatch(f, p) for p in entry["patterns"]) and ".ipynb_checkpoints" not in f]
    return repo, sha, [(f, sizes.get(f, 0)) for f in sorted(names)]


def fetch(repo, sha, fname, size):
    """Downloads one file with retries; returns the local path and the size on disk."""
    local_dir = os.path.join(paths.RAW_HF, repo.replace("/", "__"))
    path = os.path.join(local_dir, fname)
    if size and os.path.exists(path) and os.path.getsize(path) == size:
        return path, size, "skip"
    for attempt in range(6):
        try:
            p = hf_hub_download(repo, fname, repo_type="dataset", revision=sha, local_dir=local_dir, token=TOKEN)
            return p, os.path.getsize(p), "ok"
        except Exception as e:  # rate limits and dropped connections are common on long runs
            wait = 30 * (attempt + 1)
            log(f"retry {attempt + 1} {repo}/{fname}: {type(e).__name__}: {str(e)[:200]} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed {repo}/{fname}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DL_WORKERS", "10")),
                    help="parallel downloads (default: $DL_WORKERS or 10)")
    ap.add_argument("--plan", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_plan.json"),
                    help="the plan to download (default: the 16K stage's download_plan.json)")
    a = ap.parse_args()
    with open(a.plan) as f:
        plan = json.load(f)
    jobs, pins, failed = [], [], []
    for entry in plan:
        repo, sha, files = resolve(entry)
        total = sum(s for _, s in files)
        log(f"plan {repo} @ {sha[:12]}: {len(files)} files, {total / 1e9:.2f} GB")
        pins.append({"repo": repo, "requested": entry["revision"], "sha": sha})
        jobs += [(repo, sha, f, s) for f, s in files]
    grand = sum(j[3] for j in jobs)
    log(f"total {len(jobs)} files, {grand / 1e9:.1f} GB")

    # Small files first, so the many tiny benchmark files do not wait behind the 500 MB shards.
    jobs.sort(key=lambda j: j[3])
    done_bytes, t0, records = 0, time.time(), []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, *j): j for j in jobs}
        for fut in as_completed(futs):
            repo, sha, fname, size = futs[fut]
            try:
                path, got, status = fut.result()
                done_bytes += got
                records.append({"repo": repo, "sha": sha, "file": fname, "bytes": got, "path": path})
                rate = done_bytes / max(time.time() - t0, 1) / 1e6
                log(f"{status} {repo}/{fname} {got / 1e6:.0f} MB | {done_bytes / 1e9:.1f}/{grand / 1e9:.1f} GB, {rate:.0f} MB/s")
            except Exception as e:
                failed.append({"repo": repo, "file": fname, "error": str(e)[:300]})
                log(f"FAILED {repo}/{fname}: {e}")

    # The record is named after the plan's directory. A second plan in the same directory, such as
    # sft/download_plan_v7_v9.json, adds the rest of its file name, so it does not overwrite the
    # record of the first plan.
    plan_name = os.path.basename(os.path.dirname(os.path.abspath(a.plan)))
    plan_extra = os.path.splitext(os.path.basename(a.plan))[0].removeprefix("download_plan").lstrip("_")
    if plan_extra:
        plan_name = f"{plan_name}.{plan_extra}"
    os.makedirs(paths.RAW_HF, exist_ok=True)
    with open(os.path.join(paths.RAW_HF, f"downloads.{plan_name}.json"), "w") as f:
        json.dump({"pins": pins, "files": sorted(records, key=lambda r: (r["repo"], r["file"])), "failed": failed},
                  f, indent=1)
    log(f"DONE {len(records)} files, {done_bytes / 1e9:.1f} GB, failed {len(failed)}, {(time.time() - t0) / 60:.0f} min")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
