# Data

The scripts build all training data; only a few inputs are prepared by hand. Run the build from the
repository root in the data environment (`requirements/data_prep.txt` with the repository installed).
Raw sources go to `$ORBIT_RAW`, tokenized data to `$ORBIT_DATA`.

## Build order

1. Tokenizers: `data_prep/tokenizer/prepare_tokenizers.py`.
2. Stages 1 and 2 (4K): `data_prep/pretrain/tokenize_4k.py`, once per stream.
3. Stage 3 (16K): the builders `data_prep/long_context/build_*.py`, then `run_tasks.py --stage 16k`
   and `mix.py --stage 16k`.
4. Stage 4 (32K): `run_tasks.py --stage 32k` and `mix.py --stage 32k`.
5. Stage 5 (SFT): `data_prep/sft/prepare_tools.py`, then
   `data_prep/sft/build.py --version data_prep/sft/versions/v5_smoltalk2.yaml --build` and the same
   command with `--resolve`.
6. Stage 6 (RL) and distillation: the tools in `scripts/verl/data/` build their `--data` folder.

`data_prep/README.md` and `scripts/verl/data/README.md` have the full commands.

## Inputs you prepare

Any data of the kind below works. The configs keep the shares these streams had in the release. To
train without one, delete it from the config (for 16K and 32K also from the stage file); the trainer
spreads its share over the other streams.

| Input | Format | Used by |
|---|---|---|
| Russian text in five folders: `web`, `books`, `science`, `wikipedia`, `wikiquote` | `.jsonl`, `.jsonl.gz`, `.jsonl.zst`, `.jsonl.zstd` or `.parquet` with a `text` field; `tokenize_4k.py` writes each folder to `$ORBIT_DATA/pretrain_4k/ru/` | stages 1 to 4 |
| Russian chats: `$ORBIT_RAW/russian_sft/<kind>.jsonl` for `science`, `math`, `code`, `dialog`, `tasks`, `creative`, `education`, `qa_reasoning`, `summarization` | one chat per line, `{"messages": [...]}`: an optional system turn, then user and assistant turns, starting with the user and ending with the assistant | stages 3 to 5 and distillation |
| Math problems with short answers | jsonl rows with `instruction`, `response` as `\boxed{...}` and `source`; `scripts/verl/data/passrate/passrate.sh --kind math` keeps those the model solves only sometimes, as `math_keep.jsonl` (at least 6,454 rows) | stage 6 and distillation |
| Russian math problems | the same format; `passrate.sh --kind math_ru` writes `ru_math_keep.jsonl` | the `mix_ru` RL arm |

Measure the keeps on the export after SFT.