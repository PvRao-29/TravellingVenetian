"""
Overnight Venice DQN training on Modal with Weights & Biases logging.

Prerequisites:
    uv add modal wandb
    uv run modal setup

    # W&B (recommended for overnight runs) — get key at https://wandb.ai/authorize
    uv run modal secret create wandb WANDB_API_KEY=<your-key>

    # Or skip W&B entirely:
    uv run modal run modal_train.py --no-wandb

Launch (defaults: 40k episodes, T4 GPU, ~8–12 h):
    uv run modal run modal_train.py

Custom run:
    modal run modal_train.py --episodes 50000 --gpu A10G --run-name venice-v1

Download artifacts after training:
    mkdir -p runs/dqn_venice
    modal volume get venice-dqn-runs dqn_venice/final_model.pt runs/dqn_venice/final_model.pt
    modal volume get venice-dqn-runs dqn_venice/best_model.pt runs/dqn_venice/best_model.pt
    modal volume get venice-dqn-runs dqn_venice/training_log.csv runs/dqn_venice/training_log.csv

Then re-run Part C in ``venice_experiment_pipeline.ipynb`` (it loads ``runs/dqn_venice/final_model.pt``).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import modal

APP_NAME = "venice-dqn-train"
VOLUME_NAME = "venice-dqn-runs"
WANDB_PROJECT = "venice-dqn"

# Overnight defaults: enough steps for epsilon decay + many gradient updates.
DEFAULT_EPISODES = 40_000
DEFAULT_EVAL_EVERY = 500
DEFAULT_CHECKPOINT_EVERY = 2_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_HIDDEN = (512, 512)
DEFAULT_GPU = "T4"  # env sim is CPU-bound; T4 is cost-effective for batch updates
TIMEOUT_SECONDS = 12 * 60 * 60  # 12 hours

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5.0",
        "wandb>=0.19.0",
        "gymnasium>=1.0.0",
        "networkx>=3.2",
        "numpy>=2.0.0",
    )
    .add_local_file("venice.py", "/root/venice.py")
    .add_local_file("dqn_agent.py", "/root/dqn_agent.py")
    .add_local_file("dqn_train.py", "/root/dqn_train.py")
)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/vol/runs": volume},
)
def train_remote(
    *,
    episodes: int = DEFAULT_EPISODES,
    deliveries: int = 15,
    docks: int = 6,
    seed: int = 7,
    eval_every: int = DEFAULT_EVAL_EVERY,
    eval_episodes: int = 20,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN,
    epsilon_decay_steps: Optional[int] = None,
    wandb_project: str = WANDB_PROJECT,
    wandb_run_name: Optional[str] = None,
    gpu_label: str = DEFAULT_GPU,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """Train on Modal GPU, persist checkpoints to the shared volume, optionally log to W&B."""

    sys.path.insert(0, "/root")
    os.chdir("/root")

    import torch

    from dqn_agent import DQNConfig
    from dqn_train import train_dqn

    if epsilon_decay_steps is None:
        # Decay epsilon over ~60% of expected gradient steps (avg ~50 steps/ep).
        epsilon_decay_steps = max(200_000, int(episodes * 50 * 0.6))

    run_name = wandb_run_name or (
        f"venice-{deliveries}d-{docks}k-seed{seed}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    )

    config = DQNConfig(
        num_deliveries=deliveries,
        num_docks=docks,
        max_episodes=episodes,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=batch_size,
        hidden_sizes=hidden_sizes,
        checkpoint_every=checkpoint_every,
        epsilon_decay_steps=epsilon_decay_steps,
        min_replay=5_000,
        replay_size=200_000,
    )

    print(
        f"Modal Venice DQN training\n"
        f"  gpu={gpu_label}  device={config.device}  cuda={torch.cuda.is_available()}\n"
        f"  episodes={episodes}  deliveries={deliveries}  docks={docks}  seed={seed}\n"
        f"  batch={batch_size}  hidden={hidden_sizes}  epsilon_decay={epsilon_decay_steps}\n"
        f"  eval_every={eval_every}  checkpoint_every={checkpoint_every}\n"
        f"  use_wandb={use_wandb}  wandb_project={wandb_project}  run={run_name}\n"
        f"  output=/vol/runs/runs/dqn_venice"
    )

    _agent, run_dir, summary = train_dqn(
        config,
        output_dir="/vol/runs",
        verbose=True,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=run_name,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
    )

    volume.commit()

    print("\nTraining complete.")
    print(f"  run_dir: {run_dir}")
    print(f"  final objective: {summary.get('eval_mean_objective_cost')}")
    print(f"  final success:   {summary.get('eval_success_rate')}")
    print(f"  best objective:  {summary.get('best_objective_cost')}")
    print("\nDownload with:")
    print("  modal volume get venice-dqn-runs dqn_venice/final_model.pt runs/dqn_venice/final_model.pt")

    return summary


@app.local_entrypoint()
def main(
    episodes: int = DEFAULT_EPISODES,
    deliveries: int = 15,
    docks: int = 6,
    seed: int = 7,
    eval_every: int = DEFAULT_EVAL_EVERY,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gpu: str = DEFAULT_GPU,
    run_name: str = "",
    wandb_project: str = WANDB_PROJECT,
    no_wandb: bool = False,
) -> None:
    """
    Local CLI entrypoint — dispatches training to Modal.

    Example:
        uv run modal run modal_train.py
        uv run modal run modal_train.py --episodes 50000 --gpu A10G --run-name overnight-v2
        uv run modal run modal_train.py --no-wandb   # skip W&B (no secret needed)
    """

    use_wandb = not no_wandb
    remote = train_remote.with_options(
        gpu=gpu,
        secrets=[modal.Secret.from_name("wandb")] if use_wandb else [],
    )

    summary = remote.remote(
        episodes=episodes,
        deliveries=deliveries,
        docks=docks,
        seed=seed,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        batch_size=batch_size,
        wandb_project=wandb_project,
        wandb_run_name=run_name or None,
        gpu_label=gpu,
        use_wandb=use_wandb,
    )

    print("\n=== Remote training finished ===")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
