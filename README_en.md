<p align="center">
  <img src="assets/orbitlm_banner.webp" alt="OrbitLM" width="100%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.00673"><img alt="arXiv: 2610.00673" src="https://img.shields.io/badge/arXiv-2610.00673-b31b1b"></a>
  <a href="https://huggingface.co/ai-forever/OrbitLM-1.8B-Thinking"><img alt="Hugging Face: OrbitLM-1.8B-Thinking" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OrbitLM--1.8B--Thinking-ffc107"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-2f6fde"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-2f6fde">
</p>

<p align="center"><a href="README.md">Русский</a> · <b>English</b></p>

# OrbitLM-1.8B

OrbitLM-1.8B is a looped language model. It uses the Ouro-1.4B architecture with Qwen3's tokenizer:
one stack of 24 transformer layers (hidden_size 2048, 16 attn_heads) runs four times per token, and
after each pass an exit gate gives the token a probability of stopping. The model has 1.86B
parameters: 1.23B in the layers and 0.62B in the untied embedding and output matrices.

The paper is [Closing the Loop: Practical Training Recipes for Looped Language Models](https://arxiv.org/abs/2610.00673).

The repository holds the full training pipeline from scratch and the inference scripts with budget
forcing, an FP8 KV cache and speculative decoding. The packages are `orbit` (trainer) and
`orbit_verl` (distillation and RL), and the environment variables start with `ORBIT_`. Commands run
from the repository root.

## Model weights

The trained model is
[ai-forever/OrbitLM-1.8B-Thinking](https://huggingface.co/ai-forever/OrbitLM-1.8B-Thinking) on
Hugging Face. Its model card has the evaluation results and examples for transformers, vLLM and
self-speculative decoding. The model code ships with the weights, so loading needs
`trust_remote_code=True`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "ai-forever/OrbitLM-1.8B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, dtype=torch.bfloat16, trust_remote_code=True, early_exit_threshold=None
).to("cuda")
# PyTorch's cuDNN attention kernel fails on long contexts in some CUDA setups. Without it, SDPA uses
# its flash or memory-efficient kernel, which computes the same attention.
torch.backends.cuda.enable_cudnn_sdp(False)

messages = [{"role": "user", "content": "What is 17 times 23?"}]
prompt = "<|endoftext|>" + tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
output = model.generate(**inputs, max_new_tokens=8192)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Prompts start with `<|endoftext|>`, as in training. `enable_thinking=False` answers without a
reasoning trace.

## Pipeline

| # | Stage | Config | Starts from | Recipe |
|---|---|---|---|---|
| 1 | Pretraining 4K | `configs/train/pretrain/stage1_4k.yaml` | random init | 100,000 steps of 4.19M tokens, lr 3e-4, RoPE 4e4 |
| 2 | Pretraining 4K, continued | `configs/train/pretrain/stage2_4k.yaml` | stage 1, step 100000 | 200,000 steps, lr 1e-4 |
| 3 | Context 16K | `configs/train/pretrain/stage3_16k.yaml` | stage 2, step 200000 | 47,684 steps (200B tokens), RoPE 1e6, document masking |
| 4 | Context 32K | `configs/train/pretrain/stage4_32k.yaml` | stage 3, step 47684 | 4,769 steps (20B tokens), RoPE 4e6 |
| 5 | SFT | `configs/train/sft/sft_v5.yaml` | stage 4, step 4769 | 7,000 steps of 2.1M tokens, lr 1e-5 with a cosine to 1e-6 |
| 6 | RL | `configs/verl/rl_sft_mix.yaml` | stage 5, step 7000 | arm `mix` (math and code), loss on every loop weighted by the exit gate (`rltt_exit`), lr 1e-6, up to 150 steps |
| 7 | Inference | `inference/` | any chat export | vLLM, FP8 KV cache, speculative decoding, budget forcing |

Stages 1 to 5 run on the trainer (`src/orbit`), stage 6 on verl 0.9.1 (`src/orbit_verl`). The
release model is step 90 of stage 6. The code also has on-policy distillation, described under RL.

## Setup

| Environment | Install from | Used by |
|---|---|---|
| Training | `requirements/train.txt` (torch 2.9, CUDA 12.8) | stages 1 to 5, export, `inference/generate_hf.py` |
| Evaluation | `requirements/eval.txt` (vLLM 0.22) | the evaluation that stages 4 and 5 run during training |
| verl | `requirements/verl/build_env.sh` (torch 2.11, CUDA 13, vLLM 0.26) | stage 6, distillation, `inference/generate_vllm.py` |
| Data | `requirements/data_prep.txt` | the data build |

```bash
python -m pip install -r requirements/train.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . --no-deps
bash requirements/verl/build_env.sh /path/to/envs/orbit-verl /path/to/flash_attn-2.8.3-<torch2.11-cu13-cp312>.whl
```

Install the repository with `pip install -e . --no-deps` into every environment that imports `orbit`
or `orbit_verl`; `build_env.sh` does it for verl. Its second argument is a flash-attn 2.8.3 wheel that
you build once with nvcc; without it RL and distillation cannot load the model. Keep vLLM at 0.26 in
the verl environment: RL and distillation ran and were tested with it. For inference alone a newer vLLM works
too. vLLM dropped its Ouro model after 0.26, and the `orbit_vllm` plugin that the install above adds
brings it back (checked with vLLM 0.34). The evaluation environment is set up once, as
`eval/in_training/README.md` shows.

| Variable | Holds |
|---|---|
| `ORBIT_RAW` | raw sources and the data you prepare yourself |
| `ORBIT_DATA` | tokenized training data |
| `ORBIT_TOKENIZERS` | the three prepared tokenizers |
| `ORBIT_RUNS` | trainer runs and TensorBoard logs |
| `ORBIT_EXPORTS` | the HF exports between stages |
| `ORBIT_EVAL_PYTHON`, `ORBIT_LM_EVAL`, `ORBIT_EVAL_CACHE` | the evaluation interpreter, lm-eval folder and offline cache |
| `ORBIT_ATTN_KERNEL_PATH` | optional local build of the FA3 kernel |
| `ORBIT_VERL_RUNS` | prepared models, datasets and runs of verl |
| `VERL_PYTHON` | the python of the verl environment |

## Data

```bash
python data_prep/tokenizer/prepare_tokenizers.py --out-dir "$ORBIT_TOKENIZERS"
```

Then build the data of each stage as `docs/DATA.md` describes. A few inputs are prepared by hand:
Russian text, Russian chats and math problems.

## Training: stages 1 to 5

```bash
scripts/train/run_local.sh configs/train/pretrain/stage1_4k.yaml        # every GPU of one machine
torchrun --nnodes N --node_rank R --nproc_per_node 8 --rdzv_endpoint HOST:PORT \
    -m orbit.train --config configs/train/pretrain/stage1_4k.yaml       # several machines
```

`$ORBIT_RUNS` must be on a filesystem that all nodes share. Use a GPU count that divides 64, or the
steps grow and the step numbers below no longer exist; the release used 64 GPUs for stages 1 to 4 and
32 for stage 5. A restarted run resumes from its `latest.txt`. Stages 4 and 5 run the evaluation
during training, so set it up first.

Export the last checkpoint of each stage (on CPU). The SFT export needs `--chat-export`.

```bash
python -m orbit.export $ORBIT_RUNS/orbit_stage1_4k $ORBIT_EXPORTS/stage1_4k_step100000 \
    --model-config src/orbit/ouro_1.4b_qwen3_config.json --tokenizer $ORBIT_TOKENIZERS/tokenizer_qwen3_orbit
```

| Stage | Export step | As | Tokenizer |
|---|---|---|---|
| 1 | 100000 | `stage1_4k_step100000` | `tokenizer_qwen3_orbit` |
| 2 | 200000 | `stage2_4k_step200000` | `tokenizer_qwen3_orbit` |
| 3 | 47684 | `stage3_16k_step47684` | `tokenizer_qwen3_orbit` |
| 4 | 4769 | `stage4_32k_step4769` | `tokenizer_qwen3_orbit` |
| 5 | 7000 | `sft_v5_step7000` | `tokenizer_qwen3_4b_qwen35_tools`, with `--chat-export` |

`src/orbit/README.md` covers the trainer in more detail.

## RL: stage 6

```bash
RECIPE_TAG=lr1e6_fp8tis_rlttexit bash scripts/verl/run_rl.sh mix \
    --export $ORBIT_EXPORTS/sft_v5_step7000 --model-name sft_v5_step7000 --data /path/to/verl_data -- \
    actor_rollout_ref.actor.policy_loss.loss_mode=rltt_exit trainer.total_training_steps=150
$VERL_PYTHON scripts/verl/watch.py $ORBIT_VERL_RUNS/runs/rl_mix__lr1e6_fp8tis_rlttexit --rl --val-dir /path/to/verl_data --follow 600
$VERL_PYTHON scripts/verl/export_hf_fp32.py --ckpt $ORBIT_VERL_RUNS/runs/rl_mix__lr1e6_fp8tis_rlttexit/ckpt/global_step_90/actor \
    --base-export $ORBIT_EXPORTS/sft_v5_step7000 --out $ORBIT_EXPORTS/orbitlm_1.8b
```

The run uses the `mix` arm on one 8-GPU node, with the loss on every loop weighted by the exit gate
(`rltt_exit`) and a cap of 150 steps. The release model is its step 90, chosen by the validation
scores; no later step was better. RL can collapse late in a run, so keep the watcher running. On
80 GB GPUs add `--gpu-mem 0.7` before `--`. The arms `math`, `code` and `mix_ru` are alternatives
that the release model did not use.

`--data` holds the prompt pool, gold math and code rows and validation sets, built as
`scripts/verl/data/README.md` shows. The code reward runs programs in a sandbox that needs Linux on
x86_64 with user namespaces, Landlock and seccomp. For several nodes, run once with
`--prepare-only`, then with `--no-prepare` on every node with `RANK`, `WORLD_SIZE` and
`MASTER_ADDR` set; `docs/verl/README.md` has the details.

The code also has on-policy distillation from Qwen3-8B (`configs/verl/distill_rl_sft_16k.yaml`,
`scripts/verl/run_distill.sh`), and it works well too. It runs on one 8-GPU node, where six GPUs
train the student and two serve the teacher, and RL can start from its fp32 export instead of the SFT
export. `docs/verl/README.md` has the commands.

## Inference

```bash
$VERL_PYTHON inference/generate_vllm.py $MODEL --prompts prompts.jsonl --output answers.jsonl \
    --speculative --budget-forcing --think-budget 14336 --answer-budget 2048
```

The FP8 KV cache is on by default and holds 2.1 times as many tokens as bf16. `--speculative` drafts
four tokens with a one-loop pass; together with the FP8 cache it decodes 1.7 to 2.2 times as fast as
bf16 without speculation. `--budget-forcing` closes the
reasoning after `--think-budget` tokens, in Russian for Russian prompts. `inference/README.md` has the
rest.

## Tests

```bash
python -m pytest -q tests/train data_prep tests/inference                           # training env + dev.txt + data_prep.txt
CUDA_VISIBLE_DEVICES= $VERL_PYTHON -m pytest -q tests/verl tests/inference          # verl environment
```

Short GPU checks: the one-GPU configs in `configs/train/smoke/` and `bash scripts/verl/smoke.sh rl|distill`.

## Citation

```bibtex
@article{marchenko2026closing,
  title   = {Closing the Loop: Practical Training Recipes for Looped Language Models},
  author  = {Marchenko, Andrei and Bezrukov, Viacheslav and Kashurin, Oleg and Fedorova, Inessa and Bocharov, Dmitry and Shakhvalieva, Yuliana and Tikhonova, Maria and Ternovskii, Valerii},
  journal = {arXiv preprint arXiv:2610.00673},
  year    = {2026}
}
```

## License

Apache License 2.0 (`LICENSE`). Files with their own copyright notice keep it.
