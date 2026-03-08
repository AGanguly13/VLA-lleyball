# Cosmos-Guided Training: Two-Node Setup

Launch the vLLM inference server and VolleyBots training on **two separate GPU nodes**.

## Prerequisites

- SLURM cluster with GPU partitions
- `vllm` installed (`pip install vllm`)
- VolleyBots environment set up (see main README)
- Each node needs at least one GPU:
  - **vLLM node**: ~18 GB VRAM (Cosmos-Reason2-8B at FP8)
  - **Training node**: GPU for Isaac Sim + RL training

---

## Terminal 1 — vLLM Inference Server

Allocate a GPU node and start the model server:

```bash
# use the actual one with your project id
salloc -A [allocation number] -p gengpu -N 1 --gres=gpu:a100:1 --cpus-per-task=8 --mem=40G -t 4:00:00

# note the hostname — you'll need it for Terminal 2
hostname    # e.g. gpu-node-01

# activate your environment
conda activate volley

# launch vLLM serving Cosmos-Reason2-8B
vllm serve nvidia/Cosmos-Reason2-8B \
    --port 8000 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code
```

Wait until you see output like `Uvicorn running on http://0.0.0.0:8000`.
The server is now ready to accept requests.

---

## Terminal 2 — VolleyBots Training

In a **separate terminal**, allocate a second GPU node and launch training:

```bash
# grab another GPU node
salloc --gres=gpu:1 --mem=64G --time=8:00:00

conda activate volley
cd ~/VolleyBots/scripts

# point cosmos.server_url at the vLLM node from Terminal 1
# replace gpu-node-01 with the actual hostname from `hostname` above
python train.py \
    headless=true \
    cosmos.enabled=true \
    cosmos.server_url="http://gpu-node-01:8000" \
    task=SingleJuggleVolleyball \
    task.env.num_envs=16
```

All other Cosmos settings (`model_name`, `max_new_tokens`, `call_every_k`, etc.)
are read from `cfg/train.yaml`. Override any of them on the command line:

```bash
python train.py \
    headless=true \
    cosmos.enabled=true \
    cosmos.server_url="http://gpu-node-01:8000" \
    cosmos.call_every_k=100 \
    cosmos.max_frames=8 \
    task=SingleJuggleVolleyball
```

---

## Quick Verification

Before launching training, confirm the vLLM server is reachable from the
training node:

```bash
# from the training node
curl http://gpu-node-01:8000/v1/models
```

You should get a JSON response listing `nvidia/Cosmos-Reason2-8B`.

---

## Defaults (cfg/train.yaml)


| Setting          | Default                    |
| ---------------- | -------------------------- |
| `model_name`     | `nvidia/Cosmos-Reason2-8B` |
| `max_new_tokens` | `1024`                     |
| `call_every_k`   | `50`                       |
| `record_every`   | `5`                        |
| `max_frames`     | `8`                        |
| `num_commands`   | `6`                        |
| `timeout`        | `60.0`                     |


---

## Troubleshooting

- **Connection refused**: Make sure the vLLM server is fully started before
launching training. Check that the hostname and port are correct and that
no firewall blocks traffic between nodes.
- **OOM on vLLM node**: The 8B model needs ~18 GB VRAM at FP8. Use
`--dtype float16` if your GPU has more memory, or fall back to the 2B model
(`cosmos.model_name="nvidia/Cosmos-Reason2-2B"`).
- **Slow rollouts**: Cosmos is queried every `call_every_k` env steps. Increase
this value if inference latency is bottlenecking training throughput.

