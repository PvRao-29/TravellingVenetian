"""
Supervised delivery-order model for the Venice TSP environment.

This is the project's "simple ML model": instead of learning routing from scratch
with reinforcement learning (see ``dqn_agent.py`` / ``dqn_train.py``, which did not
converge on this problem within a practical budget), we learn to *imitate* the
OR-Tools ``soft`` solver.

Idea (learning-to-rank / learned dispatch policy):
    * For a set of training instances, solve each with OR-Tools (soft time windows)
      to get a high-quality delivery visit order.
    * Replay that order step by step. At each step the model sees a feature vector
      for every still-undelivered delivery (leg cost/time from the current node,
      the delivery's time window, the implied arrival/slack/lateness, etc.) and the
      label is which delivery the solver visited next.
    * A small MLP scores candidates; we train it as a pointwise ranker (binary
      "is this the next stop?" with class weighting).

At inference time we run the learned scorer greedily: from the depot, repeatedly
pick the highest-scored undelivered delivery, advance an approximate clock, and
stop when all deliveries are ordered. The order is expanded into an executable
multimodal route with the same ``expand_required_order`` machinery the OR-Tools
bridge uses, then scored by ``env.simulate_route`` -- the exact metric path used
by every other method in the experiment notebook.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from venice import Mode, NodeKind, VeniceTSPConfig, VeniceTSPEnv
from ORtools import (
    build_multimodal_state_graph,
    build_required_node_matrices,
    expand_required_order,
    solve_with_ortools,
)

# Number of features produced by ``candidate_features``. Keep in sync.
FEATURE_DIM = 12

COST_SCALE = 100  # build_required_node_matrices stores objective_cost * 100


@dataclass
class RankerConfig:
    """Hyperparameters for the supervised delivery-order ranker."""

    hidden_sizes: Tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    epochs: int = 60
    batch_size: int = 256
    weight_decay: float = 1e-5
    seed: int = 0
    # Instance shape the labels were generated on (for reproducibility / metadata).
    num_deliveries: int = 15
    num_docks: int = 6
    time_window_mode: str = "soft"
    ortools_time_limit_seconds: int = 5
    train_seeds: Tuple[int, ...] = field(
        default_factory=lambda: tuple(range(100, 160))
    )


def required_node_setup(env: VeniceTSPEnv):
    """
    Build the depot+delivery required-node list and the OR-Tools cost/time/leg
    matrices for the *currently reset* env instance.

    Returns ``(required_nodes, cost_matrix, time_matrix, leg_paths, unreachable)``.
    """

    graph = build_multimodal_state_graph(env)
    delivery_nodes = [int(x) for x in env.delivery_nodes.tolist()]
    required_nodes = [int(env.depot_node)] + delivery_nodes
    cost_matrix, time_matrix, leg_paths, unreachable = build_required_node_matrices(
        env, graph, required_nodes
    )
    return required_nodes, cost_matrix, time_matrix, leg_paths, unreachable


def candidate_features(
    env: VeniceTSPEnv,
    cost_matrix: np.ndarray,
    time_matrix: np.ndarray,
    required_nodes: Sequence[int],
    i: int,
    j: int,
    current_time: float,
    frac_remaining: float,
) -> np.ndarray:
    """
    Feature vector for visiting required-node index ``j`` next, from index ``i``
    at clock ``current_time`` (minutes). All quantities are lightly normalized so
    the MLP sees roughly unit-scale inputs.
    """

    base = env.config.base_hour * 60.0
    node = env.nodes[int(required_nodes[j])]

    leg_cost = float(cost_matrix[i, j]) / COST_SCALE
    leg_time = float(time_matrix[i, j])
    arrival = current_time + leg_time

    tw_start = float(node.time_window_start)
    tw_end = float(node.time_window_end)
    slack = tw_end - arrival
    wait = max(0.0, tw_start - arrival)
    late = max(0.0, arrival - tw_end)
    flooded = float(env.tide_flooded[int(required_nodes[j])])

    return np.array(
        [
            leg_cost / 1000.0,
            leg_time / 120.0,
            (arrival - base) / 600.0,
            (tw_start - base) / 600.0,
            (tw_end - base) / 600.0,
            slack / 120.0,
            wait / 120.0,
            late / 120.0,
            (tw_end - tw_start) / 120.0,
            float(node.demand) / 4.0,
            float(frac_remaining),
            flooded,
        ],
        dtype=np.float32,
    )


def advance_time(
    env: VeniceTSPEnv,
    time_matrix: np.ndarray,
    required_nodes: Sequence[int],
    i: int,
    j: int,
    current_time: float,
) -> float:
    """
    Approximate clock advance after visiting ``j`` from ``i``.

    Uses the static leg time plus earliness wait and service time. This is an
    approximation of the env's true timing (it ignores in-leg crowd/flood/transfer
    effects), but it is identical between training and inference, so the learned
    features stay consistent.
    """

    node = env.nodes[int(required_nodes[j])]
    arrival = current_time + float(time_matrix[i, j])
    if env.config.earliness_wait:
        start_service = max(arrival, float(node.time_window_start))
    else:
        start_service = arrival
    service = node.service_time_minutes if node.kind == NodeKind.DELIVERY else 0.0
    return start_service + float(service)


class RankerNet(nn.Module):
    """Small MLP that maps a candidate feature vector to a scalar score (logit)."""

    def __init__(self, feature_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = feature_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _make_env(num_deliveries: int, num_docks: int, seed: int) -> VeniceTSPEnv:
    return VeniceTSPEnv(
        config=VeniceTSPConfig(
            num_delivery_nodes=int(num_deliveries),
            num_docks=int(num_docks),
            seed=int(seed),
        )
    )


def build_dataset(
    config: RankerConfig,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (features, labels, group_ids) by imitating OR-Tools on training seeds.

    ``features`` is ``(N, FEATURE_DIM)``, ``labels`` is ``(N,)`` in {0, 1} (1 = the
    delivery the solver visited next), ``group_ids`` marks the decision step each
    row belongs to (useful for diagnostics).
    """

    feats: List[np.ndarray] = []
    labels: List[float] = []
    groups: List[int] = []
    group_id = 0
    solved = 0

    for seed in config.train_seeds:
        env = _make_env(config.num_deliveries, config.num_docks, seed)
        env.reset(seed=seed)
        required_nodes, cost_matrix, time_matrix, _legs, unreachable = required_node_setup(env)
        if unreachable:
            if verbose:
                print(f"  seed {seed}: skipped (unreachable pairs)")
            continue

        result = solve_with_ortools(
            env,
            time_window_mode=config.time_window_mode,
            time_limit_seconds=config.ortools_time_limit_seconds,
            verbose=False,
        )
        if not result.success or not result.required_order:
            if verbose:
                print(f"  seed {seed}: skipped (no OR-Tools solution)")
            continue

        index_of = {int(node): idx for idx, node in enumerate(required_nodes)}
        # required_order is (depot, deliveries..., depot) as physical node ids.
        order_indices = [index_of[int(n)] for n in result.required_order]
        delivery_order = order_indices[1:-1]  # drop depot at both ends

        remaining = set(range(1, len(required_nodes)))  # all delivery indices
        current = 0
        current_time = env.config.base_hour * 60.0
        total = len(delivery_order)

        for chosen in delivery_order:
            if chosen not in remaining:
                break
            frac_remaining = len(remaining) / total if total else 0.0
            for cand in remaining:
                feats.append(
                    candidate_features(
                        env, cost_matrix, time_matrix, required_nodes,
                        current, cand, current_time, frac_remaining,
                    )
                )
                labels.append(1.0 if cand == chosen else 0.0)
                groups.append(group_id)
            group_id += 1
            current_time = advance_time(
                env, time_matrix, required_nodes, current, chosen, current_time
            )
            remaining.discard(chosen)
            current = chosen

        solved += 1

    if not feats:
        raise RuntimeError(
            "No training data generated. Check OR-Tools availability and train seeds."
        )

    if verbose:
        print(
            f"Built dataset from {solved}/{len(config.train_seeds)} solved instances: "
            f"{len(feats)} rows, {int(sum(labels))} positives."
        )

    return (
        np.asarray(feats, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
    )


def train_ranker(
    config: Optional[RankerConfig] = None,
    *,
    verbose: bool = True,
) -> Tuple[RankerNet, Dict[str, object]]:
    """Generate OR-Tools labels and train the pointwise ranker. Returns (model, info)."""

    config = config or RankerConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    X, y, _groups = build_dataset(config, verbose=verbose)

    device = torch.device("cpu")
    model = RankerNet(FEATURE_DIM, config.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Class imbalance: one positive per decision step, many negatives.
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_t = torch.as_tensor(X, device=device)
    y_t = torch.as_tensor(y, device=device)
    n = X_t.shape[0]

    history: List[float] = []
    for epoch in range(config.epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            logits = model(X_t[idx])
            loss = loss_fn(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        mean_loss = epoch_loss / max(1, batches)
        history.append(mean_loss)
        if verbose and (epoch % max(1, config.epochs // 10) == 0 or epoch == config.epochs - 1):
            print(f"  epoch {epoch + 1:>3}/{config.epochs}  loss={mean_loss:.4f}")

    info = {
        "config": dataclasses.asdict(config),
        "num_rows": int(n),
        "num_positives": int(n_pos),
        "final_loss": history[-1] if history else None,
        "loss_history": history,
    }
    return model, info


@torch.no_grad()
def ranker_order_indices(
    env: VeniceTSPEnv,
    model: RankerNet,
    required_nodes: Sequence[int],
    cost_matrix: np.ndarray,
    time_matrix: np.ndarray,
) -> List[int]:
    """Greedy learned ordering over required-node indices (depot at both ends)."""

    model.eval()
    n = len(required_nodes)
    remaining = set(range(1, n))
    total = len(remaining)
    current = 0
    current_time = env.config.base_hour * 60.0
    order = [0]

    while remaining:
        frac_remaining = len(remaining) / total if total else 0.0
        cands = sorted(remaining)
        feats = np.stack(
            [
                candidate_features(
                    env, cost_matrix, time_matrix, required_nodes,
                    current, j, current_time, frac_remaining,
                )
                for j in cands
            ]
        )
        scores = model(torch.as_tensor(feats)).cpu().numpy()
        chosen = cands[int(np.argmax(scores))]
        order.append(chosen)
        current_time = advance_time(
            env, time_matrix, required_nodes, current, chosen, current_time
        )
        remaining.discard(chosen)
        current = chosen

    order.append(0)  # return to depot
    return order


def ranker_route(env: VeniceTSPEnv, model: RankerNet) -> List[int]:
    """
    Full inference pipeline for a *reset* env: build matrices, run the learned
    greedy ordering, expand into an executable multimodal route.

    Returns an empty list if the instance has unreachable required pairs.
    """

    required_nodes, cost_matrix, time_matrix, leg_paths, unreachable = required_node_setup(env)
    if unreachable:
        return []

    order_idx = ranker_order_indices(env, model, required_nodes, cost_matrix, time_matrix)
    order_nodes = [int(required_nodes[i]) for i in order_idx]
    expanded, _legs = expand_required_order(order_nodes, required_nodes, leg_paths)
    return list(expanded)


def save_ranker(model: RankerNet, config: RankerConfig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(config),
            "feature_dim": FEATURE_DIM,
        },
        path,
    )
    return path


def load_ranker(path: str) -> Tuple[RankerNet, RankerConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_fields = {f.name for f in dataclasses.fields(RankerConfig)}
    config = RankerConfig(**{k: v for k, v in ckpt["config"].items() if k in cfg_fields})
    model = RankerNet(int(ckpt.get("feature_dim", FEATURE_DIM)), config.hidden_sizes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config


if __name__ == "__main__":
    # Quick standalone training + sanity evaluation.
    cfg = RankerConfig()
    trained, train_info = train_ranker(cfg)
    print(f"Trained ranker: {train_info['num_rows']} rows, final loss {train_info['final_loss']:.4f}")
    save_ranker(trained, cfg, os.path.join("runs", "ml_ranker", "ranker.pt"))
    print("Saved to runs/ml_ranker/ranker.pt")
