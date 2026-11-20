"""Write the tool-calling dialogs of SFT v5 in the shared stored shape (tool_records.py).

SmolTalk2 hermes, xLAM and SmolAgents are converted in full. Dolci Instruct
keeps only the Tool Use domain. The two Nemotron sets are read from 100k-row
tool-calling samples of Nemotron-SFT-Agentic-v2 and
Nemotron-Post-Training-Dataset-v1, at the paths data_prep/README.md gives;
any 100k-row sample of their tool-calling rows can stand in. The output is
one jsonl file per set, which versions/v5_smoltalk2.yaml reads.
ToolMind (toolmind_open, v7 on) is read from the open_datasets folder of a
Nanbeige/ToolMind download in hf/Nanbeige__ToolMind under --raw-root.

  python data_prep/sft/prepare_tools.py
  python data_prep/sft/prepare_tools.py --only dolci_tool_use
"""
import argparse
import glob
import json
import os
from collections import Counter

import pyarrow.compute as pc
import pyarrow.parquet as pq

from render import unclosed_think
from tool_records import Skip, build_record

# Sources, relative to --raw-root.
SMOLTALK = "hf/HuggingFaceTB__smoltalk2/SFT"
DOLCI = "hf/allenai__Dolci-Instruct-SFT/data"
NEMOTRON_AGENTIC = "nemotron_tool_samples/Nemotron-SFT-Agentic-v2/tool_calling_100k.jsonl"
NEMOTRON_POST = "nemotron_tool_samples/Nemotron-Post-Training-Dataset-v1/tool_calling_100k.parquet"
TOOLMIND = "hf/Nanbeige__ToolMind/open_datasets"


def attempt(source, tools, messages):
    try:
        return build_record(source, tools, messages)
    except Skip as exc:
        return exc


def convert_smoltalk(path, source):
    table = pq.read_table(path, columns=["messages", "chat_template_kwargs"])
    for row in table.to_pylist():
        kwargs = row.get("chat_template_kwargs") or {}
        yield attempt(source, kwargs.get("xml_tools"), row.get("messages"))


def iter_dolci(root):
    for path in sorted(glob.glob(os.path.join(root, "train-*.parquet"))):
        domains = pq.read_table(path, columns=["domain"]).column("domain")
        if pc.sum(pc.equal(domains, "Tool Use")).as_py() == 0:
            continue
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["messages", "domain"]):
            for row in batch.to_pylist():
                if row["domain"] != "Tool Use":
                    continue
                tools = None
                for message in row["messages"]:
                    if isinstance(message, dict) and message.get("role") == "system" and message.get("functions"):
                        tools = message["functions"]
                        break
                yield attempt("dolci_tool_use", tools, row["messages"])


def iter_nemotron_agentic(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield Skip("bad_messages")
                continue
            yield attempt("nemotron_agentic_v2", row.get("tools"), row.get("messages"))


def iter_nemotron_post(path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            metadata = row.get("metadata")
            try:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                tools = metadata.get("tools") if isinstance(metadata, dict) else None
            except json.JSONDecodeError:
                yield Skip("bad_tools")
                continue
            yield attempt("nemotron_post_v1", tools, row.get("messages"))


def catalog(raw_root):
    smoltalk = os.path.join(raw_root, SMOLTALK)
    return {
        "smoltalk_hermes": lambda: convert_smoltalk(
            os.path.join(smoltalk, "hermes_function_calling_v1_no_think-00000-of-00001.parquet"),
            "smoltalk_hermes",
        ),
        "smoltalk_xlam": lambda: convert_smoltalk(
            os.path.join(smoltalk, "xlam_traces_no_think-00000-of-00001.parquet"),
            "smoltalk_xlam",
        ),
        "smoltalk_smolagents": lambda: convert_smoltalk(
            os.path.join(smoltalk, "smolagents_toolcalling_traces_think-00000-of-00001.parquet"),
            "smoltalk_smolagents",
        ),
        "dolci_tool_use": lambda: iter_dolci(os.path.join(raw_root, DOLCI)),
        "nemotron_agentic_v2": lambda: iter_nemotron_agentic(os.path.join(raw_root, NEMOTRON_AGENTIC)),
        "nemotron_post_v1": lambda: iter_nemotron_post(os.path.join(raw_root, NEMOTRON_POST)),
        "toolmind_open": lambda: iter_toolmind(os.path.join(raw_root, TOOLMIND)),
    }


def iter_toolmind(root):
    """Nanbeige ToolMind open_datasets. Loss is applied later, on the last turn only."""
    for path in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
        source = os.path.basename(path)[: -len("-query.jsonl")]
        kept = 0
        reasons = Counter()
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = _toolmind_record(source, json.loads(line))
                if isinstance(item, Skip):
                    reasons[item.reason] += 1
                else:
                    kept += 1
                yield item
        print(f"  {source}: kept {kept} skipped {dict(reasons)}", flush=True)


def _toolmind_record(source, row):
    item = attempt(source, row.get("tools"), row.get("conversations"))
    if isinstance(item, Skip):
        return item
    for message in item["messages"]:
        if message.get("role") != "assistant":
            continue
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if unclosed_think(text) or unclosed_think(reasoning):
            return Skip("unclosed_think")
    return item


SOURCES = ("smoltalk_hermes", "smoltalk_xlam", "smoltalk_smolagents", "dolci_tool_use",
           "nemotron_agentic_v2", "nemotron_post_v1", "toolmind_open")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default=os.environ.get("ORBIT_RAW"),
                        help="the raw sources root (default: $ORBIT_RAW)")
    parser.add_argument("--out", default=None, help="default: $ORBIT_DATA/sft/tools")
    parser.add_argument("--only", action="append", choices=sorted(SOURCES))
    args = parser.parse_args()
    if not args.raw_root:
        parser.error("set ORBIT_RAW or pass --raw-root")
    if args.out is None:
        if not os.environ.get("ORBIT_DATA"):
            parser.error("set ORBIT_DATA or pass --out")
        args.out = os.path.join(os.environ["ORBIT_DATA"], "sft", "tools")
    os.makedirs(args.out, exist_ok=True)
    chosen = catalog(args.raw_root)
    if args.only:
        chosen = {name: chosen[name] for name in args.only}
    summary = {}
    for name, factory in chosen.items():
        print(f"converting {name}", flush=True)
        path = os.path.join(args.out, name + ".jsonl")
        kept = 0
        reasons = Counter()
        with open(path, "w", encoding="utf-8") as handle:
            for item in factory():
                if isinstance(item, Skip):
                    reasons[item.reason] += 1
                    continue
                try:
                    line = json.dumps(item, ensure_ascii=False)
                except TypeError:
                    reasons["bad_call"] += 1
                    continue
                handle.write(line + "\n")
                kept += 1
                if kept % 20000 == 0:
                    print(f"  {kept}", flush=True)
        summary[name] = {"path": path, "kept": kept, "skipped": dict(reasons)}
        print(f"  kept {kept} skipped {dict(reasons)}", flush=True)
    stats_path = os.path.join(args.out, "stats.json")
    if args.only and os.path.exists(stats_path):
        previous = json.loads(open(stats_path, encoding="utf-8").read())
        previous.update(summary)
        summary = previous
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
