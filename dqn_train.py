"""
Standalone training entry point for the Venice DQN agent.

This script owns the training loop (replay collection, optimization, logging, and
checkpointing). The agent, network, rollout, and evaluation helpers live in
``dqn_agent.py`` so the experiment notebook can load and run a trained policy
without importing any training machinery.

Artifacts are written under ``<output_dir>/runs/dqn_venice/``:
``training_log.csv``, periodic ``checkpoint_ep*.pt``, and ``final_model.pt``.

Examples:
    python dqn_train.py --episodes 2000 --deliveries 15 --docks 6 --seed 7
    python dqn_train.py --episodes 5000 --wandb --eval-every 250
    modal run modal_train.py
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from venice import Mode, VeniceTSPConfig, VeniceTSPEnv
from dqn_agent import (
    DQNAgent,
    DQNConfig,
    ReplayBuffer,
    evaluate_policy,
    greedy_route,
    observation_dim,
    run_episode,
)


def make_env(config: DQNConfig) -> VeniceTSPEnv:
    """Build the training environment described by a DQNConfig."""

    env_config = VeniceTSPConfig(
        num_delivery_nodes=int(config.num_deliveries),
        num_docks=int(config.num_docks),
        seed=int(config.seed),
    )
    return VeniceTSPEnv(config=env_config)


def evaluate_policy_objective(
    env: VeniceTSPEnv,
    agent: DQNAgent,
    *,
    episodes: int = 20,
    seed_base: int = 7,
) -> Dict[str, Optional[float]]:
    """
    Greedy evaluation scored through ``env.simulate_route`` (same path as baselines).

    Uses seeds ``seed_base, seed_base + 1, ...`` so eval aligns with the notebook's
    fixed-topology benchmark instances.
    """

    objectives: List[float] = []
    lateness_values: List[float] = []
    successes = 0

    for i in range(episodes):
        seed = int(seed_base + i)
        env.reset(seed=seed)
        route = greedy_route(env, agent, seed)
        sim = env.simulate_route(
            route,
            start_node=env.depot_node,
            start_mode=Mode.VAN,
            start_time_minutes=env.config.base_hour * 60.0,
            require_all_deliveries=True,
            require_return_to_depot=env.config.require_return_to_depot,
            penalize_revisits=True,
        )
        if not sim.get("feasible", False):
            continue
        successes += 1
        objective = sim.get("objective_cost")
        if objective is not None:
            objectives.append(float(objective))
        transitions = sim.get("transitions") or []
        lateness_values.append(
            sum(float(t["lateness_minutes"]) for t in transitions)
        )

    n = float(episodes)
    return {
        "eval_episodes": n,
        "eval_success_rate": successes / n if episodes else None,
        "eval_mean_objective_cost": (
            sum(objectives) / len(objectives) if objectives else None
        ),
        "eval_mean_lateness_minutes": (
            sum(lateness_values) / len(lateness_values) if lateness_values else None
        ),
    }


def train_dqn(
    config: DQNConfig,
    output_dir: str = ".",
    *,
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "venice-dqn",
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    eval_every: int = 0,
    eval_episodes: int = 20,
    eval_seed_base: Optional[int] = None,
    log_every: int = 1,
) -> Tuple[DQNAgent, str, Dict[str, Any]]:
    """
    Train a DQN agent and persist its log, checkpoints, and final model.

    Returns the trained agent, run directory, and a summary dict with final metrics.
    """

    env = make_env(config)
    run_dir = os.path.join(output_dir, "runs", "dqn_venice")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "training_log.csv")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    obs_dim = observation_dim(env)
    num_actions = env.total_nodes
    max_steps = config.max_steps_per_episode or env.max_episode_steps
    eval_seed_base = config.seed if eval_seed_base is None else int(eval_seed_base)

    agent = DQNAgent(obs_dim, num_actions, config)
    buffer = ReplayBuffer(config.replay_size, obs_dim, num_actions)

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            config={
                **{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()},
                "obs_dim": obs_dim,
                "num_actions": num_actions,
                "max_steps_per_episode": max_steps,
                "device": str(agent.device),
            },
        )

    reward_window: List[float] = []
    best_objective: Optional[float] = None
    best_eval: Dict[str, Optional[float]] = {}
    log_fields = [
        "episode",
        "reward",
        "length",
        "success",
        "lateness_minutes",
        "duration_minutes",
        "epsilon",
        "moving_avg_reward",
        "mean_loss",
    ]

    with open(log_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=log_fields)
        writer.writeheader()

        for episode in range(1, config.max_episodes + 1):
            stats = run_episode(
                env,
                agent,
                seed=config.seed + episode,
                greedy=False,
                buffer=buffer,
                max_steps=max_steps,
            )
            reward_window.append(stats["reward"])
            if len(reward_window) > 100:
                reward_window.pop(0)
            moving_avg = sum(reward_window) / len(reward_window)

            row = {
                "episode": episode,
                "reward": round(stats["reward"], 3),
                "length": stats["length"],
                "success": int(stats["success"]),
                "lateness_minutes": round(stats["lateness_minutes"], 3),
                "duration_minutes": round(stats["duration_minutes"], 3),
                "epsilon": round(agent.epsilon(), 4),
                "moving_avg_reward": round(moving_avg, 3),
                "mean_loss": (
                    None if stats["mean_loss"] is None else round(stats["mean_loss"], 6)
                ),
            }
            writer.writerow(row)

            if wandb_run is not None and episode % log_every == 0:
                log_payload: Dict[str, Any] = {
                    "episode": episode,
                    "train/reward": stats["reward"],
                    "train/length": stats["length"],
                    "train/success": int(stats["success"]),
                    "train/lateness_minutes": stats["lateness_minutes"],
                    "train/duration_minutes": stats["duration_minutes"],
                    "train/epsilon": agent.epsilon(),
                    "train/moving_avg_reward_100": moving_avg,
                    "train/replay_size": len(buffer),
                    "train/train_steps": agent.train_steps,
                }
                if stats["mean_loss"] is not None:
                    log_payload["train/mean_loss"] = stats["mean_loss"]
                wandb_run.log(log_payload, step=episode)

            if config.checkpoint_every and episode % config.checkpoint_every == 0:
                ckpt_path = os.path.join(run_dir, f"checkpoint_ep{episode}.pt")
                agent.save(ckpt_path)
                if wandb_run is not None:
                    import wandb

                    artifact = wandb.Artifact(
                        f"checkpoint-ep{episode}",
                        type="model",
                        metadata={"episode": episode},
                    )
                    artifact.add_file(ckpt_path)
                    wandb_run.log_artifact(artifact)

            if eval_every and episode % eval_every == 0:
                eval_metrics = evaluate_policy_objective(
                    env,
                    agent,
                    episodes=eval_episodes,
                    seed_base=eval_seed_base,
                )
                if verbose:
                    obj = eval_metrics["eval_mean_objective_cost"]
                    obj_str = "n/a" if obj is None else f"{obj:.1f}"
                    late = eval_metrics["eval_mean_lateness_minutes"]
                    late_str = "n/a" if late is None else f"{late:.1f}"
                    sr = eval_metrics["eval_success_rate"]
                    sr_str = "n/a" if sr is None else f"{sr:.0%}"
                    print(
                        f"  [eval ep {episode:>5}]  "
                        f"success={sr_str}  objective={obj_str}  lateness={late_str}"
                    )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"eval/{k.replace('eval_', '')}": v for k, v in eval_metrics.items()},
                        step=episode,
                    )

                obj_cost = eval_metrics.get("eval_mean_objective_cost")
                if obj_cost is not None and (best_objective is None or obj_cost < best_objective):
                    best_objective = obj_cost
                    best_eval = dict(eval_metrics)
                    best_path = os.path.join(run_dir, "best_model.pt")
                    agent.save(best_path)

            if verbose and (
                episode % max(1, config.max_episodes // 20) == 0 or episode == 1
            ):
                print(
                    f"  ep {episode:>5}/{config.max_episodes}  "
                    f"reward={stats['reward']:>10.1f}  len={stats['length']:>3}  "
                    f"success={int(stats['success'])}  eps={agent.epsilon():.3f}  "
                    f"avg100={moving_avg:>10.1f}"
                )

    final_path = os.path.join(run_dir, "final_model.pt")
    agent.save(final_path)

    final_eval = evaluate_policy_objective(
        env, agent, episodes=eval_episodes, seed_base=eval_seed_base
    )
    rollout_eval = evaluate_policy(make_env(config), agent, episodes=eval_episodes)

    summary = {
        "run_dir": run_dir,
        "final_model": final_path,
        "best_objective_cost": best_objective,
        **final_eval,
        **{f"rollout_{k}": v for k, v in rollout_eval.items()},
    }

    if wandb_run is not None:
        import wandb

        wandb_run.log({f"final/{k}": v for k, v in final_eval.items()})
        wandb_run.summary.update(summary)
        artifact = wandb.Artifact("final-model", type="model")
        artifact.add_file(final_path)
        wandb_run.log_artifact(artifact)
        if os.path.exists(os.path.join(run_dir, "best_model.pt")):
            best_artifact = wandb.Artifact("best-model", type="model")
            best_artifact.add_file(os.path.join(run_dir, "best_model.pt"))
            wandb_run.log_artifact(best_artifact)
        wandb_run.finish()

    if verbose:
        print(f"Training log: {log_path}")
        print(f"Final model:  {final_path}")
        if best_objective is not None:
            print(f"Best model:   {os.path.join(run_dir, 'best_model.pt')}  (objective={best_objective:.1f})")

    return agent, run_dir, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Venice DQN routing agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=2_000, help="Training episodes.")
    parser.add_argument("--deliveries", type=int, default=15, help="Number of delivery nodes.")
    parser.add_argument("--docks", type=int, default=6, help="Number of canal docks.")
    parser.add_argument("--seed", type=int, default=7, help="Seed (instance topology + torch).")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, or cuda.")
    parser.add_argument("--output-dir", default=".", help="Base directory for runs/.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="venice-dqn", help="W&B project name.")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name.")
    parser.add_argument("--eval-every", type=int, default=500, help="Greedy eval interval (0=off).")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per eval pass.")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint interval.")
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=None,
        help="Override epsilon decay steps (default scales with episodes).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    epsilon_decay = args.epsilon_decay_steps
    if epsilon_decay is None:
        # Roughly decay epsilon over the first half of training.
        epsilon_decay = max(50_000, args.episodes * 80)

    config = DQNConfig(
        num_deliveries=args.deliveries,
        num_docks=args.docks,
        max_episodes=args.episodes,
        device=args.device,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        epsilon_decay_steps=epsilon_decay,
    )
    print(
        f"Training DQN: episodes={args.episodes} deliveries={args.deliveries} "
        f"docks={args.docks} seed={args.seed} device={args.device} "
        f"epsilon_decay={epsilon_decay}"
    )
    agent, _run_dir, summary = train_dqn(
        config,
        args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )

    print("\nFinal greedy evaluation (simulate_route objective):")
    for key in (
        "eval_success_rate",
        "eval_mean_objective_cost",
        "eval_mean_lateness_minutes",
    ):
        print(f"  {key:<28} {summary.get(key)}")

    print("\nRollout evaluation (training env):")
    for key, value in summary.items():
        if key.startswith("rollout_"):
            print(f"  {key.removeprefix('rollout_'):<28} {value}")


if __name__ == "__main__":
    main()
