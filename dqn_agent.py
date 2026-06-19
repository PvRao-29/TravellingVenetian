"""
Double-DQN agent for ``VeniceTSPEnv`` with strict action masking.

This module is self-contained: it depends on numpy, PyTorch, and the Venice
environment. The agent never intentionally selects an invalid action - the
action mask is applied during epsilon-greedy exploration, greedy selection,
and the next-state target computation.

Training lives in ``dqn_train.py``; this module is the agent + inference surface
used by both the training script and the experiment notebook:
    DQNConfig, DQNAgent, ReplayBuffer, run_episode, evaluate_policy, greedy_route,
    encode_observation, observation_dim, get_action_mask
"""

from __future__ import annotations

import dataclasses
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from venice import Mode, VeniceTSPEnv


NEG_INF = -1e9


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean that ignores ``None`` and returns ``None`` for an empty sample."""

    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DQNConfig:
    """
    DQN hyperparameters plus the instance dimensions the agent was trained on.

    The env dimensions (``num_deliveries``/``num_docks``/``seed``) are stored here
    so a saved checkpoint is self-describing: a notebook can rebuild the exact
    environment - and therefore the matching observation/action sizes - without
    being told the training instance out of band.
    """

    # Instance the agent is trained/evaluated on.
    num_deliveries: int = 15
    num_docks: int = 6

    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    replay_size: int = 100_000
    min_replay: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_update_freq: int = 1_000
    grad_clip_norm: float = 10.0
    hidden_sizes: Tuple[int, ...] = (256, 256)
    max_episodes: int = 5_000
    max_steps_per_episode: Optional[int] = None  # default: env.max_episode_steps
    checkpoint_every: int = 500
    device: str = "auto"
    seed: int = 7


# =============================================================================
# Observation encoding and action masking
# =============================================================================


def observation_dim(env: VeniceTSPEnv) -> int:
    """Length of the encoded observation vector."""

    total = env.total_nodes
    n_modes = len(Mode)
    n_deliv = env.num_delivery_nodes
    # one-hot node + one-hot mode + delivered mask + time + tide mask + action mask
    return total + n_modes + n_deliv + 1 + total + total


def encode_observation(env: VeniceTSPEnv, obs: Mapping[str, Any]) -> np.ndarray:
    """
    Encode the dict observation into a flat float32 vector.

    Includes: one-hot current node, one-hot current mode, the visited/delivered
    mask, normalized time in [0, 1] (over a 24h day), the tide/flood mask, and the
    action mask itself (so the network can see which moves are currently legal).
    """

    total = env.total_nodes
    n_modes = len(Mode)

    node_oh = np.zeros(total, dtype=np.float32)
    node_oh[int(obs["current_node"])] = 1.0

    mode_oh = np.zeros(n_modes, dtype=np.float32)
    mode_oh[int(obs["current_mode"])] = 1.0

    delivered = np.asarray(obs["delivered"], dtype=np.float32)
    time_norm = np.array(
        [float(np.asarray(obs["current_time_minutes"]).reshape(-1)[0]) / 1440.0], dtype=np.float32
    )
    time_norm = np.clip(time_norm, 0.0, 1.0)
    tide = np.asarray(obs["tide_flooded"], dtype=np.float32)
    mask = np.asarray(obs["action_mask"], dtype=np.float32)

    return np.concatenate([node_oh, mode_oh, delivered, time_norm, tide, mask], axis=0)


def get_action_mask(env: VeniceTSPEnv, obs: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """
    Return a binary action mask, preferring the cheap observation field, then
    ``env.action_mask()``, then ``env.feasible_actions()``. Raises if none exist.
    """

    if obs is not None and "action_mask" in obs:
        return np.asarray(obs["action_mask"], dtype=np.int8)
    if hasattr(env, "action_mask"):
        return np.asarray(env.action_mask(), dtype=np.int8)
    if hasattr(env, "feasible_actions"):
        mask = np.zeros(env.total_nodes, dtype=np.int8)
        for a in env.feasible_actions():
            mask[int(a)] = 1
        return mask
    raise RuntimeError("Environment exposes neither action_mask() nor feasible_actions().")


# =============================================================================
# Replay buffer and Q-network
# =============================================================================


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions including the next-state mask."""

    def __init__(self, capacity: int, obs_dim: int, num_actions: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.zeros((self.capacity, num_actions), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        i = self.pos
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.next_masks[i] = next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.as_tensor(self.states[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_states": torch.as_tensor(self.next_states[idx]),
            "dones": torch.as_tensor(self.dones[idx]),
            "next_masks": torch.as_tensor(self.next_masks[idx]),
        }

    def __len__(self) -> int:
        return self.size


class QNetwork(nn.Module):
    """Simple MLP mapping an encoded observation to per-action Q-values."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Agent
# =============================================================================


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class DQNAgent:
    """Double-DQN agent with strict action masking everywhere a choice is made."""

    def __init__(self, obs_dim: int, num_actions: int, config: DQNConfig) -> None:
        self.config = config
        self.num_actions = int(num_actions)
        self.device = _resolve_device(config.device)

        self.online = QNetwork(obs_dim, num_actions, config.hidden_sizes).to(self.device)
        self.target = QNetwork(obs_dim, num_actions, config.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.train_steps = 0

    # -- action selection ---------------------------------------------------

    def epsilon(self) -> float:
        c = self.config
        frac = min(1.0, self.train_steps / max(1, c.epsilon_decay_steps))
        return c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start)

    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float) -> Optional[int]:
        """Epsilon-greedy with masking; returns ``None`` only if no action is legal."""

        valid = np.flatnonzero(np.asarray(mask) > 0)
        if valid.size == 0:
            return None
        if random.random() < epsilon:
            return int(random.choice(valid.tolist()))
        return self.greedy_action(state, mask)

    def greedy_action(self, state: np.ndarray, mask: np.ndarray) -> Optional[int]:
        valid = np.flatnonzero(np.asarray(mask) > 0)
        if valid.size == 0:
            return None
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s).squeeze(0).cpu().numpy()
        masked_q = np.where(np.asarray(mask) > 0, q, NEG_INF)
        return int(np.argmax(masked_q))

    # -- learning -----------------------------------------------------------

    def update(self, buffer: ReplayBuffer) -> Optional[float]:
        c = self.config
        if len(buffer) < max(c.min_replay, c.batch_size):
            return None

        batch = buffer.sample(c.batch_size)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        next_masks = batch["next_masks"].to(self.device)

        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net picks the next action, target net values it.
            next_q_online = self.online(next_states)
            next_q_online = next_q_online.masked_fill(next_masks <= 0, NEG_INF)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + c.gamma * next_q_target * (1.0 - dones)

        loss = F.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % c.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "train_steps": self.train_steps,
                "config": dataclasses.asdict(self.config),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.train_steps = int(ckpt.get("train_steps", 0))


# =============================================================================
# Rollouts and evaluation
#
# Training itself lives in dqn_train.py; this module keeps the agent, the shared
# episode rollout, and the (inference-time) evaluation helpers used by both the
# training script and the experiment notebook.
# =============================================================================


def run_episode(
    env: VeniceTSPEnv,
    agent: DQNAgent,
    *,
    seed: int,
    greedy: bool,
    buffer: Optional[ReplayBuffer],
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run one episode. If ``buffer`` is provided, transitions are stored and the
    agent is updated each step (training); otherwise the policy acts greedily.
    """

    obs, _info = env.reset(seed=seed)
    state = encode_observation(env, obs)
    mask = get_action_mask(env, obs)

    total_reward = 0.0
    length = 0
    total_lateness = 0.0
    step_losses: List[float] = []
    terminated = False
    truncated = False

    for _ in range(max_steps):
        epsilon = 0.0 if greedy else agent.epsilon()
        action = agent.select_action(state, mask, epsilon)
        if action is None:  # genuinely stuck: no legal action
            break

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = encode_observation(env, next_obs)
        next_mask = get_action_mask(env, next_obs)
        done = bool(terminated)

        if buffer is not None:
            buffer.push(state, action, reward, next_state, done, next_mask.astype(np.float32))
            loss = agent.update(buffer)
            if loss is not None:
                step_losses.append(loss)

        transition = info.get("transition")
        if transition is not None:
            total_lateness += float(transition.get("lateness_minutes", 0.0))

        total_reward += float(reward)
        length += 1
        state = next_state
        mask = next_mask
        if terminated or truncated:
            break

    duration = float(env.current_time_minutes - env.config.base_hour * 60.0)
    return {
        "reward": total_reward,
        "length": length,
        "success": bool(terminated),
        "lateness_minutes": total_lateness,
        "duration_minutes": duration,
        "mean_loss": _mean(step_losses),
    }


def evaluate_policy(
    env: VeniceTSPEnv,
    agent: DQNAgent,
    episodes: int = 20,
    *,
    seed_base: int = 10_000,
) -> Dict[str, Optional[float]]:
    """
    Greedy, masked evaluation of a trained agent over fresh instances.

    Reports success rate, average reward, average route length, average lateness,
    and average duration (lateness/duration are ``None`` if never observed).
    """

    max_steps = agent.config.max_steps_per_episode or env.max_episode_steps
    rewards: List[float] = []
    lengths: List[int] = []
    successes: List[bool] = []
    lateness: List[float] = []
    durations: List[float] = []

    for i in range(episodes):
        stats = run_episode(
            env, agent, seed=seed_base + i, greedy=True, buffer=None, max_steps=max_steps
        )
        rewards.append(stats["reward"])
        lengths.append(stats["length"])
        successes.append(stats["success"])
        lateness.append(stats["lateness_minutes"])
        durations.append(stats["duration_minutes"])

    return {
        "episodes": float(episodes),
        "success_rate": sum(1 for s in successes if s) / episodes if episodes else None,
        "avg_reward": _mean(rewards),
        "avg_route_length": _mean([float(x) for x in lengths]),
        "avg_lateness_minutes": _mean(lateness),
        "avg_duration_minutes": _mean(durations),
    }


def greedy_route(env: VeniceTSPEnv, agent: DQNAgent, seed: int) -> List[int]:
    """
    Greedy masked rollout that records the executed node sequence.

    Returned so the caller can score the policy through the same metric path as
    the OR-Tools / baseline routes (via ``env.simulate_route``).
    """

    obs, _ = env.reset(seed=seed)
    state = encode_observation(env, obs)
    mask = get_action_mask(env, obs)
    route: List[int] = []
    max_steps = agent.config.max_steps_per_episode or env.max_episode_steps
    for _ in range(max_steps):
        action = agent.greedy_action(state, mask)
        if action is None:
            break
        next_obs, _r, terminated, truncated, _info = env.step(action)
        route.append(int(action))
        state = encode_observation(env, next_obs)
        mask = get_action_mask(env, next_obs)
        if terminated or truncated:
            break
    return route
