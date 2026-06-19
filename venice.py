
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from math import hypot
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces


class Mode(IntEnum):
    VAN = 0
    BOAT = 1
    CART = 2


class NodeKind(IntEnum):
    HUB = 0
    DOCK = 1
    DELIVERY = 2


@dataclass(frozen=True)
class VeniceTSPConfig:
    """Static instance and objective configuration."""

    num_delivery_nodes: int = 15
    num_docks: int = 6
    base_hour: float = 8.0
    seed: Optional[int] = 7
    require_return_to_depot: bool = True

    # Objective weights. Keep these explicit so the later optimizer can tune them.
    time_cost_per_minute: float = 1.5
    monetary_cost_weight: float = 1.0
    lateness_cost_per_minute: float = 5.0
    earliness_wait: bool = True
    invalid_action_penalty: float = 10_000.0
    revisit_delivery_penalty: float = 1_000.0

    # Scenario stochasticity.
    flood_probability_delivery: float = 0.15

    # Operational assumptions.
    van_speed_m_per_min: float = 650.0      # about 39 km/h
    boat_speed_m_per_min: float = 260.0     # about 15.6 km/h; canal/lagoon average
    cart_speed_m_per_min: float = 70.0      # about 4.2 km/h with hand cart
    bridge_delay_minutes: float = 3.0
    service_time_minutes: float = 4.0

    # Transfer handling times/costs.
    van_boat_transfer_minutes: float = 30.0
    boat_cart_transfer_minutes: float = 15.0
    van_cart_transfer_minutes: float = 10.0
    van_boat_transfer_cost: float = 40.0
    boat_cart_transfer_cost: float = 10.0
    van_cart_transfer_cost: float = 8.0

    # Vehicle operating costs per move.
    van_fixed_cost: float = 5.0
    boat_fixed_cost: float = 25.0
    boat_cost_per_meter: float = 0.02
    cart_fixed_cost: float = 0.0

    # Time-dependent penalties.
    peak_start_hour: float = 10.0
    peak_end_hour: float = 16.0
    tourist_crowd_multiplier: float = 2.5
    flood_multiplier: float = 3.0

    # Episode safety cap.
    max_episode_steps_factor: int = 6


@dataclass(frozen=True)
class NodeSpec:
    node_id: int
    name: str
    kind: NodeKind
    sestiere: str
    x: float
    y: float
    road_access: bool
    water_access: bool
    cart_access: bool
    time_window_start: float
    time_window_end: float
    demand: float = 1.0
    service_time_minutes: float = 0.0
    nearest_dock: Optional[int] = None


@dataclass(frozen=True)
class ArcSpec:
    u: int
    v: int
    mode: Mode
    distance_m: float
    base_time_minutes: float
    monetary_cost: float
    bridges: int = 0


@dataclass(frozen=True)
class TransitionResult:
    feasible: bool
    reason: str
    u: int
    v: int
    start_mode: Mode
    travel_mode: Mode
    end_mode: Mode
    depart_time_minutes: float
    arrival_time_minutes: float
    completion_time_minutes: float
    duration_minutes: float
    travel_time_minutes: float
    handling_time_minutes: float
    service_time_minutes: float
    wait_time_minutes: float
    lateness_minutes: float
    monetary_cost: float
    objective_cost: float
    reward: float
    bridges: int
    crowd_multiplier: float
    flood_multiplier: float


class VeniceTSPEnv(gym.Env):
    """
    Multi-modal Venice TSP/VRP environment.

    Action = next physical node id.

    The mode is carried in the state. If the selected edge requires a different
    mode and the current node is a valid transfer point, the mode switch happens
    implicitly and the configured handling time/cost is charged.

    Hubs and docks are reusable. Delivery nodes are one-shot unless you decide
    to allow revisits in your optimizer by using simulate_route(...,
    penalize_revisits=True).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    MODE_VAN = Mode.VAN
    MODE_BOAT = Mode.BOAT
    MODE_CART = Mode.CART

    HUB_TRONCHETTO = 0
    HUB_PIAZZALE_ROMA = 1

    SESTIERI = (
        "San Marco",
        "Rialto/San Polo",
        "Cannaregio",
        "Dorsoduro",
        "Castello",
        "Santa Croce",
        "Giudecca",
    )

    # Hand-cart connectivity is not complete across all sestieri. These links
    # approximate bridge/corridor adjacency in the pedestrian layer.
    CART_ADJACENCY: Mapping[str, Tuple[str, ...]] = {
        "San Marco": ("Rialto/San Polo", "Castello", "Dorsoduro"),
        "Rialto/San Polo": ("San Marco", "Santa Croce", "Dorsoduro", "Cannaregio"),
        "Cannaregio": ("Rialto/San Polo", "Santa Croce", "Castello"),
        "Dorsoduro": ("Rialto/San Polo", "San Marco", "Santa Croce"),
        "Castello": ("San Marco", "Cannaregio"),
        "Santa Croce": ("Rialto/San Polo", "Cannaregio", "Dorsoduro"),
        "Giudecca": (),
    }

    SESTIERE_CENTROIDS: Mapping[str, Tuple[float, float]] = {
        "San Marco": (620.0, 350.0),
        "Rialto/San Polo": (390.0, 420.0),
        "Cannaregio": (320.0, 710.0),
        "Dorsoduro": (300.0, 150.0),
        "Castello": (920.0, 420.0),
        "Santa Croce": (120.0, 430.0),
        "Giudecca": (470.0, -120.0),
    }

    def __init__(
        self,
        num_delivery_nodes: Optional[int] = None,
        base_hour: Optional[float] = None,
        config: Optional[VeniceTSPConfig] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = VeniceTSPConfig(
                num_delivery_nodes=(15 if num_delivery_nodes is None else num_delivery_nodes),
                base_hour=(8.0 if base_hour is None else base_hour),
            )
        elif num_delivery_nodes is not None or base_hour is not None:
            raise ValueError("Pass either config or legacy num_delivery_nodes/base_hour, not both.")

        self.config = config
        self.num_fixed_hubs = 2
        self.num_docks = config.num_docks
        self.num_delivery_nodes = config.num_delivery_nodes
        self.total_nodes = self.num_fixed_hubs + self.num_docks + self.num_delivery_nodes
        self.delivery_start = self.num_fixed_hubs + self.num_docks
        self.delivery_nodes = np.arange(self.delivery_start, self.total_nodes, dtype=np.int64)
        self.depot_node = self.HUB_TRONCHETTO
        self.end_depot_node = self.HUB_TRONCHETTO

        self._topology_rng = np.random.default_rng(config.seed)
        self.nodes: Dict[int, NodeSpec] = {}
        self.arcs: Dict[Tuple[int, int, Mode], ArcSpec] = {}
        self.graph = nx.MultiDiGraph()
        self._build_static_topology()

        self.action_space = spaces.Discrete(self.total_nodes)
        self.observation_space = spaces.Dict(
            {
                "current_node": spaces.Discrete(self.total_nodes),
                "current_mode": spaces.Discrete(len(Mode)),
                "delivered": spaces.MultiBinary(self.num_delivery_nodes),
                "current_time_minutes": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "tide_flooded": spaces.MultiBinary(self.total_nodes),
                "action_mask": spaces.MultiBinary(self.total_nodes),
            }
        )

        self.current_node: int = self.depot_node
        self.current_mode: Mode = Mode.VAN
        self.delivered = np.zeros(self.num_delivery_nodes, dtype=np.int8)
        self.tide_flooded = np.zeros(self.total_nodes, dtype=np.int8)
        self.current_time_minutes: float = self.config.base_hour * 60.0
        self.episode_steps: int = 0
        self.max_episode_steps: int = max(10, self.config.max_episode_steps_factor * self.total_nodes)
        self.last_transition: Optional[TransitionResult] = None

    # ------------------------------------------------------------------
    # Topology generation
    # ------------------------------------------------------------------

    def _build_static_topology(self) -> None:
        """Build a deterministic synthetic instance with stable node ids."""
        cfg = self.config
        rng = self._topology_rng

        self.nodes[0] = NodeSpec(
            node_id=0,
            name="Tronchetto road-water hub",
            kind=NodeKind.HUB,
            sestiere="Mainland edge",
            x=-180.0,
            y=390.0,
            road_access=True,
            water_access=True,
            cart_access=False,
            time_window_start=0.0,
            time_window_end=1440.0,
        )
        self.nodes[1] = NodeSpec(
            node_id=1,
            name="Piazzale Roma road-foot-water hub",
            kind=NodeKind.HUB,
            sestiere="Santa Croce",
            x=0.0,
            y=420.0,
            road_access=True,
            water_access=True,
            cart_access=True,
            time_window_start=0.0,
            time_window_end=1440.0,
        )

        # Guarantee coverage by placing one dock near several sestieri before random extras.
        dock_sestieri = list(self.SESTIERI[: min(cfg.num_docks, len(self.SESTIERI))])
        while len(dock_sestieri) < cfg.num_docks:
            dock_sestieri.append(str(rng.choice(self.SESTIERI)))

        for k, sestiere in enumerate(dock_sestieri):
            node_id = self.num_fixed_hubs + k
            cx, cy = self.SESTIERE_CENTROIDS[sestiere]
            x = float(cx + rng.normal(0.0, 55.0))
            y = float(cy + rng.normal(0.0, 55.0))
            self.nodes[node_id] = NodeSpec(
                node_id=node_id,
                name=f"{sestiere} canal micro-hub {k + 1}",
                kind=NodeKind.DOCK,
                sestiere=sestiere,
                x=x,
                y=y,
                road_access=False,
                water_access=True,
                cart_access=True,
                time_window_start=0.0,
                time_window_end=1440.0,
            )

        delivery_window_starts = np.array([480.0, 540.0, 600.0, 660.0, 720.0, 780.0])

        # Only generate deliveries in sestieri that have a generated dock. Otherwise
        # some synthetic instances become structurally impossible before optimization
        # even begins. If you import real dock data later, keep this invariant: every
        # delivery cluster must have at least one reachable transfer node.
        available_delivery_sestieri = list(dict.fromkeys(dock_sestieri))
        base_weights = {
            "San Marco": 0.18,
            "Rialto/San Polo": 0.20,
            "Cannaregio": 0.17,
            "Dorsoduro": 0.15,
            "Castello": 0.15,
            "Santa Croce": 0.10,
            "Giudecca": 0.05,
        }
        sestiere_probs = np.array([base_weights[s] for s in available_delivery_sestieri], dtype=float)
        sestiere_probs = sestiere_probs / sestiere_probs.sum()

        for j, node_id in enumerate(range(self.delivery_start, self.total_nodes)):
            sestiere = str(rng.choice(available_delivery_sestieri, p=sestiere_probs))
            cx, cy = self.SESTIERE_CENTROIDS[sestiere]
            x = float(cx + rng.normal(0.0, 120.0))
            y = float(cy + rng.normal(0.0, 90.0))
            tw_start = float(rng.choice(delivery_window_starts))
            tw_width = float(rng.choice([90.0, 120.0, 150.0], p=[0.25, 0.55, 0.20]))
            demand = float(rng.integers(1, 5))
            nearest_dock = self._nearest_existing_dock(sestiere, x, y)
            self.nodes[node_id] = NodeSpec(
                node_id=node_id,
                name=f"Delivery {j + 1} - {sestiere}",
                kind=NodeKind.DELIVERY,
                sestiere=sestiere,
                x=x,
                y=y,
                road_access=False,
                water_access=False,
                cart_access=True,
                time_window_start=tw_start,
                time_window_end=tw_start + tw_width,
                demand=demand,
                service_time_minutes=cfg.service_time_minutes,
                nearest_dock=nearest_dock,
            )

        self._build_arcs()

    def _nearest_existing_dock(self, sestiere: str, x: float, y: float) -> int:
        dock_ids = [
            i
            for i, node in self.nodes.items()
            if node.kind == NodeKind.DOCK and node.sestiere == sestiere
        ]
        if not dock_ids:
            dock_ids = [i for i, node in self.nodes.items() if node.kind == NodeKind.DOCK]
        return min(dock_ids, key=lambda i: self._euclidean_xy(x, y, self.nodes[i].x, self.nodes[i].y))

    def _build_arcs(self) -> None:
        """Add all structurally feasible directed arcs, keyed by travel mode."""
        self.arcs.clear()
        self.graph.clear()

        for node in self.nodes.values():
            self.graph.add_node(node.node_id, **self.node_to_dict(node))

        for u in self.nodes:
            for v in self.nodes:
                if u == v:
                    continue
                for mode in Mode:
                    arc = self._make_arc_if_feasible(u, v, mode)
                    if arc is not None:
                        self.arcs[(u, v, mode)] = arc
                        edge_data = asdict(arc)
                        edge_data["mode"] = int(mode)
                        edge_data["mode_name"] = mode.name
                        self.graph.add_edge(
                            u,
                            v,
                            key=int(mode),
                            **edge_data,
                        )

    def _make_arc_if_feasible(self, u: int, v: int, mode: Mode) -> Optional[ArcSpec]:
        cfg = self.config
        a = self.nodes[u]
        b = self.nodes[v]
        distance = self._network_distance_m(a, b, mode)

        if mode == Mode.VAN:
            if not (a.road_access and b.road_access):
                return None
            base_time = distance / cfg.van_speed_m_per_min
            return ArcSpec(u, v, mode, distance, base_time, cfg.van_fixed_cost, bridges=0)

        if mode == Mode.BOAT:
            if not (a.water_access and b.water_access):
                return None
            # Cargo boats work between road-water hubs and canal docks. They do
            # not serve inland delivery addresses directly in this synthetic model.
            base_time = distance / cfg.boat_speed_m_per_min
            cost = cfg.boat_fixed_cost + distance * cfg.boat_cost_per_meter
            return ArcSpec(u, v, mode, distance, base_time, cost, bridges=0)

        if mode == Mode.CART:
            if not (a.cart_access and b.cart_access):
                return None
            if not self._cart_sestieri_connected(a.sestiere, b.sestiere):
                return None
            base_time = distance / cfg.cart_speed_m_per_min
            bridges = self._estimate_bridge_count(a, b, distance)
            return ArcSpec(u, v, mode, distance, base_time, cfg.cart_fixed_cost, bridges=bridges)

        raise ValueError(f"Unsupported mode: {mode}")

    def _cart_sestieri_connected(self, s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        return s2 in self.CART_ADJACENCY.get(s1, ())

    def _estimate_bridge_count(self, a: NodeSpec, b: NodeSpec, distance_m: float) -> int:
        if a.sestiere == b.sestiere:
            return int(max(0, round(distance_m / 500.0)))
        # Crossing between sestieri usually adds bridges and canal bottlenecks.
        return int(max(1, round(distance_m / 350.0)))

    def _network_distance_m(self, a: NodeSpec, b: NodeSpec, mode: Mode) -> float:
        straight = self._euclidean_xy(a.x, a.y, b.x, b.y)
        if mode == Mode.VAN:
            return max(100.0, straight * 1.15)
        if mode == Mode.BOAT:
            return max(150.0, straight * 1.35 + 120.0)
        if mode == Mode.CART:
            bridge_inflation = 1.20 if a.sestiere == b.sestiere else 1.55
            return max(40.0, straight * bridge_inflation + 50.0)
        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _euclidean_xy(x1: float, y1: float, x2: float, y2: float) -> float:
        return hypot(x1 - x2, y1 - y2)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})

        if options.get("regenerate_topology", False):
            topology_seed = options.get("topology_seed", seed if seed is not None else self.config.seed)
            self._topology_rng = np.random.default_rng(topology_seed)
            self.nodes.clear()
            self._build_static_topology()

        self.current_node = int(options.get("start_node", self.depot_node))
        self.current_mode = Mode(int(options.get("start_mode", Mode.VAN)))
        self.current_time_minutes = float(options.get("start_time_minutes", self.config.base_hour * 60.0))
        self.delivered = np.zeros(self.num_delivery_nodes, dtype=np.int8)
        self.episode_steps = 0
        self.last_transition = None

        if "tide_flooded" in options:
            tide = np.asarray(options["tide_flooded"], dtype=np.int8)
            if tide.shape != (self.total_nodes,):
                raise ValueError(f"tide_flooded must have shape ({self.total_nodes},)")
            self.tide_flooded = tide.copy()
        else:
            self.tide_flooded = np.zeros(self.total_nodes, dtype=np.int8)
            delivery_floods = self.np_random.binomial(
                1,
                self.config.flood_probability_delivery,
                size=self.num_delivery_nodes,
            ).astype(np.int8)
            self.tide_flooded[self.delivery_start :] = delivery_floods

        # Hubs and docks are treated as operationally usable even when tide is high.
        for node_id, node in self.nodes.items():
            if node.kind != NodeKind.DELIVERY:
                self.tide_flooded[node_id] = 0

        obs = self._get_obs()
        info = self._info(extra={"feasible_actions": self.feasible_actions()})
        return obs, info

    def step(self, action: int):
        action = int(action)
        self.episode_steps += 1

        if action < 0 or action >= self.total_nodes:
            return self._invalid_step("Invalid node index")

        if self._is_delivery(action) and self._is_delivered(action):
            return self._invalid_step("Delivery node already served", self.config.revisit_delivery_penalty)

        transition = self.transition_cost(
            self.current_node,
            action,
            self.current_mode,
            self.current_time_minutes,
        )

        if not transition.feasible:
            return self._invalid_step(transition.reason)

        self.current_node = action
        self.current_mode = transition.end_mode
        self.current_time_minutes = transition.completion_time_minutes
        self.last_transition = transition

        if self._is_delivery(action):
            self.delivered[self._delivery_index(action)] = 1

        terminated = self._is_terminal()
        truncated = self.episode_steps >= self.max_episode_steps
        info = self._info(
            extra={
                "transition": asdict(transition),
                "feasible_actions": self.feasible_actions(),
            }
        )
        return self._get_obs(), transition.reward, terminated, truncated, info

    def _invalid_step(self, reason: str, penalty: Optional[float] = None):
        penalty = self.config.invalid_action_penalty if penalty is None else penalty
        obs = self._get_obs()
        info = self._info(extra={"error": reason, "feasible_actions": self.feasible_actions()})
        return obs, -float(penalty), False, False, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "current_node": int(self.current_node),
            "current_mode": int(self.current_mode),
            "delivered": self.delivered.astype(np.int8).copy(),
            "current_time_minutes": np.array([self.current_time_minutes], dtype=np.float32),
            "tide_flooded": self.tide_flooded.astype(np.int8).copy(),
            "action_mask": self.action_mask().astype(np.int8),
        }

    def render(self) -> None:
        node = self.nodes[self.current_node]
        print(
            f"t={self.current_time_minutes:.1f} min | node={self.current_node} "
            f"({node.name}) | mode={self.current_mode.name} | "
            f"delivered={int(self.delivered.sum())}/{self.num_delivery_nodes}"
        )

    # ------------------------------------------------------------------
    # Feasibility and cost model
    # ------------------------------------------------------------------

    def feasible_actions(
        self,
        node: Optional[int] = None,
        mode: Optional[Mode] = None,
        time_minutes: Optional[float] = None,
        delivered: Optional[np.ndarray] = None,
    ) -> List[int]:
        node = self.current_node if node is None else int(node)
        mode = self.current_mode if mode is None else Mode(int(mode))
        time_minutes = self.current_time_minutes if time_minutes is None else float(time_minutes)
        delivered = self.delivered if delivered is None else delivered

        feasible: List[int] = []
        for v in range(self.total_nodes):
            if v == node:
                continue
            if self._is_delivery(v) and delivered[self._delivery_index(v)] == 1:
                continue
            result = self.transition_cost(node, v, mode, time_minutes)
            if result.feasible:
                feasible.append(v)
        return feasible

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.total_nodes, dtype=np.int8)
        for action in self.feasible_actions():
            mask[action] = 1
        return mask

    def transition_cost(
        self,
        u: int,
        v: int,
        current_mode: Mode,
        depart_time_minutes: float,
    ) -> TransitionResult:
        u = int(u)
        v = int(v)
        current_mode = Mode(int(current_mode))
        depart_time_minutes = float(depart_time_minutes)

        candidates: List[TransitionResult] = []
        for travel_mode in Mode:
            arc = self.arcs.get((u, v, travel_mode))
            if arc is None:
                continue
            handling = self._transfer_cost(u, current_mode, travel_mode)
            if handling is None:
                continue
            handling_time, handling_cost = handling
            candidates.append(
                self._score_arc(
                    arc=arc,
                    current_mode=current_mode,
                    handling_time_minutes=handling_time,
                    handling_cost=handling_cost,
                    depart_time_minutes=depart_time_minutes,
                )
            )

        if not candidates:
            return self._infeasible_transition(u, v, current_mode, depart_time_minutes, "No feasible mode/transfer for arc")

        return min(candidates, key=lambda x: x.objective_cost)

    def _score_arc(
        self,
        arc: ArcSpec,
        current_mode: Mode,
        handling_time_minutes: float,
        handling_cost: float,
        depart_time_minutes: float,
    ) -> TransitionResult:
        cfg = self.config
        v_node = self.nodes[arc.v]

        crowd_multiplier = self._crowd_multiplier(arc.v, arc.mode, depart_time_minutes)
        flood_multiplier = self._flood_multiplier(arc.v, arc.mode)
        bridge_delay = arc.bridges * cfg.bridge_delay_minutes if arc.mode == Mode.CART else 0.0

        travel_time = arc.base_time_minutes * crowd_multiplier * flood_multiplier + bridge_delay
        arrival_time = depart_time_minutes + handling_time_minutes + travel_time

        wait_time = 0.0
        if cfg.earliness_wait and arrival_time < v_node.time_window_start:
            wait_time = v_node.time_window_start - arrival_time

        service_time = v_node.service_time_minutes if v_node.kind == NodeKind.DELIVERY else 0.0
        completion_time = arrival_time + wait_time + service_time
        lateness = max(0.0, completion_time - v_node.time_window_end)
        duration = completion_time - depart_time_minutes
        monetary_cost = arc.monetary_cost + handling_cost
        objective_cost = (
            duration * cfg.time_cost_per_minute
            + monetary_cost * cfg.monetary_cost_weight
            + lateness * cfg.lateness_cost_per_minute
        )

        return TransitionResult(
            feasible=True,
            reason="ok",
            u=arc.u,
            v=arc.v,
            start_mode=current_mode,
            travel_mode=arc.mode,
            end_mode=arc.mode,
            depart_time_minutes=depart_time_minutes,
            arrival_time_minutes=arrival_time,
            completion_time_minutes=completion_time,
            duration_minutes=duration,
            travel_time_minutes=travel_time,
            handling_time_minutes=handling_time_minutes,
            service_time_minutes=service_time,
            wait_time_minutes=wait_time,
            lateness_minutes=lateness,
            monetary_cost=monetary_cost,
            objective_cost=objective_cost,
            reward=-objective_cost,
            bridges=arc.bridges,
            crowd_multiplier=crowd_multiplier,
            flood_multiplier=flood_multiplier,
        )

    def _transfer_cost(self, node_id: int, from_mode: Mode, to_mode: Mode) -> Optional[Tuple[float, float]]:
        if from_mode == to_mode:
            return 0.0, 0.0

        node = self.nodes[node_id]
        cfg = self.config

        has_from = self._node_supports_mode(node, from_mode)
        has_to = self._node_supports_mode(node, to_mode)
        if not (has_from and has_to):
            return None

        pair = frozenset((from_mode, to_mode))
        if pair == frozenset((Mode.VAN, Mode.BOAT)):
            return cfg.van_boat_transfer_minutes, cfg.van_boat_transfer_cost
        if pair == frozenset((Mode.BOAT, Mode.CART)):
            return cfg.boat_cart_transfer_minutes, cfg.boat_cart_transfer_cost
        if pair == frozenset((Mode.VAN, Mode.CART)):
            return cfg.van_cart_transfer_minutes, cfg.van_cart_transfer_cost
        return None

    @staticmethod
    def _node_supports_mode(node: NodeSpec, mode: Mode) -> bool:
        if mode == Mode.VAN:
            return node.road_access
        if mode == Mode.BOAT:
            return node.water_access
        if mode == Mode.CART:
            return node.cart_access
        raise ValueError(f"Unsupported mode: {mode}")

    def _crowd_multiplier(self, node_id: int, mode: Mode, time_minutes: float) -> float:
        if mode != Mode.CART:
            return 1.0
        hour = (time_minutes / 60.0) % 24.0
        node = self.nodes[node_id]
        if self.config.peak_start_hour <= hour <= self.config.peak_end_hour:
            if node.sestiere in ("San Marco", "Rialto/San Polo"):
                return self.config.tourist_crowd_multiplier
        return 1.0

    def _flood_multiplier(self, node_id: int, mode: Mode) -> float:
        if mode == Mode.CART and self.tide_flooded[node_id] == 1:
            return self.config.flood_multiplier
        return 1.0

    def _infeasible_transition(
        self,
        u: int,
        v: int,
        mode: Mode,
        depart_time_minutes: float,
        reason: str,
    ) -> TransitionResult:
        return TransitionResult(
            feasible=False,
            reason=reason,
            u=u,
            v=v,
            start_mode=mode,
            travel_mode=mode,
            end_mode=mode,
            depart_time_minutes=depart_time_minutes,
            arrival_time_minutes=depart_time_minutes,
            completion_time_minutes=depart_time_minutes,
            duration_minutes=0.0,
            travel_time_minutes=0.0,
            handling_time_minutes=0.0,
            service_time_minutes=0.0,
            wait_time_minutes=0.0,
            lateness_minutes=0.0,
            monetary_cost=0.0,
            objective_cost=float("inf"),
            reward=-self.config.invalid_action_penalty,
            bridges=0,
            crowd_multiplier=1.0,
            flood_multiplier=1.0,
        )

    # ------------------------------------------------------------------
    # Optimization-facing utilities
    # ------------------------------------------------------------------

    def simulate_route(
        self,
        route: Sequence[int],
        *,
        start_node: Optional[int] = None,
        start_mode: Mode = Mode.VAN,
        start_time_minutes: Optional[float] = None,
        require_all_deliveries: bool = True,
        require_return_to_depot: Optional[bool] = None,
        penalize_revisits: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate node sequence without mutating the live environment.

        `route` should contain the nodes to visit after `start_node`. Include
        transfer docks/hubs explicitly when needed.
        """
        node = self.depot_node if start_node is None else int(start_node)
        mode = Mode(int(start_mode))
        time = self.config.base_hour * 60.0 if start_time_minutes is None else float(start_time_minutes)
        require_return = self.config.require_return_to_depot if require_return_to_depot is None else require_return_to_depot

        delivered = np.zeros(self.num_delivery_nodes, dtype=np.int8)
        transitions: List[Dict[str, Any]] = []
        total_cost = 0.0
        total_reward = 0.0
        feasible = True
        violations: List[str] = []

        for raw_v in route:
            v = int(raw_v)
            if v < 0 or v >= self.total_nodes:
                feasible = False
                violations.append(f"Invalid node id: {v}")
                total_cost += self.config.invalid_action_penalty
                continue

            if self._is_delivery(v) and delivered[self._delivery_index(v)] == 1:
                msg = f"Delivery node revisited: {v}"
                if penalize_revisits:
                    violations.append(msg)
                    total_cost += self.config.revisit_delivery_penalty
                    total_reward -= self.config.revisit_delivery_penalty
                    continue

            transition = self.transition_cost(node, v, mode, time)
            transitions.append(asdict(transition))
            if not transition.feasible:
                feasible = False
                violations.append(f"{node}->{v}: {transition.reason}")
                total_cost += self.config.invalid_action_penalty
                total_reward -= self.config.invalid_action_penalty
                continue

            total_cost += transition.objective_cost
            total_reward += transition.reward
            node = v
            mode = transition.end_mode
            time = transition.completion_time_minutes
            if self._is_delivery(v):
                delivered[self._delivery_index(v)] = 1

        if require_all_deliveries and not np.all(delivered == 1):
            feasible = False
            missing = self.delivery_nodes[delivered == 0].tolist()
            violations.append(f"Missing deliveries: {missing}")

        if require_return and node != self.end_depot_node:
            feasible = False
            violations.append(f"Route does not end at depot {self.end_depot_node}; ended at {node}")

        return {
            "feasible": feasible,
            "objective_cost": float(total_cost),
            "reward": float(total_reward),
            "end_node": int(node),
            "end_mode": int(mode),
            "end_time_minutes": float(time),
            "delivered_count": int(delivered.sum()),
            "missing_delivery_nodes": self.delivery_nodes[delivered == 0].astype(int).tolist(),
            "violations": violations,
            "transitions": transitions,
        }

    def greedy_feasible_route(self) -> List[int]:
        """
        Produce a deterministic feasible baseline route.

        It deliberately uses Venice logistics structure instead of naive nearest
        neighbor: travel by boat between micro-hubs, then use cart legs inside
        a local delivery cluster. This is a warm start / sanity check, not the
        optimizer you should submit as final.
        """
        node = self.depot_node
        mode = Mode.VAN
        time = self.config.base_hour * 60.0
        delivered = np.zeros(self.num_delivery_nodes, dtype=np.int8)
        route: List[int] = []

        def advance(v: int) -> bool:
            nonlocal node, mode, time, route
            tr = self.transition_cost(node, v, mode, time)
            if not tr.feasible:
                return False
            route.append(v)
            node = v
            mode = tr.end_mode
            time = tr.completion_time_minutes
            if self._is_delivery(v):
                delivered[self._delivery_index(v)] = 1
            return True

        def move_to_dock(target_dock: int) -> bool:
            nonlocal node, mode
            if node == target_dock:
                return True
            current = self.nodes[node]
            if current.kind == NodeKind.DELIVERY:
                local_dock = current.nearest_dock
                if local_dock is not None and node != local_dock:
                    if not advance(local_dock):
                        return False
            # From any hub/dock that supports boat, selecting target_dock will
            # either continue by boat or perform an implicit transfer to boat.
            if node != target_dock:
                return advance(target_dock)
            return True

        # Earliest due-date ordering is a decent baseline for hard windows.
        ordered_deliveries = sorted(
            self.delivery_nodes.astype(int).tolist(),
            key=lambda n: (self.nodes[n].time_window_end, self.nodes[n].time_window_start, n),
        )

        for delivery in ordered_deliveries:
            if delivered[self._delivery_index(delivery)] == 1:
                continue
            target_dock = self.nodes[delivery].nearest_dock
            if target_dock is None:
                continue
            if not self._cart_sestieri_connected(self.nodes[node].sestiere, self.nodes[delivery].sestiere):
                if not move_to_dock(target_dock):
                    break
            # Even if the sestiere is connected, going through the nearest dock
            # is often better when we are currently in VAN/BOAT mode.
            if mode != Mode.CART and node != target_dock:
                if not move_to_dock(target_dock):
                    break
            if not advance(delivery):
                if not move_to_dock(target_dock):
                    break
                if not advance(delivery):
                    break

        if self.config.require_return_to_depot and node != self.end_depot_node:
            current = self.nodes[node]
            if current.kind == NodeKind.DELIVERY and current.nearest_dock is not None:
                advance(current.nearest_dock)
            if node != self.end_depot_node:
                advance(self.end_depot_node)

        return route

    def lower_bound_cost_matrix(
        self,
        *,
        start_time_minutes: Optional[float] = None,
        include_transfer_costs: bool = True,
    ) -> np.ndarray:
        """
        Return a static optimistic node-to-node objective matrix.

        This ignores delivered-state logic and uses the cheapest feasible incoming
        mode at a fixed departure time. Use it for heuristics, clustering, warm
        starts, or lower bounds, not as the final objective truth.
        """
        time = self.config.base_hour * 60.0 if start_time_minutes is None else float(start_time_minutes)
        matrix = np.full((self.total_nodes, self.total_nodes), np.inf, dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)

        for u in range(self.total_nodes):
            for v in range(self.total_nodes):
                if u == v:
                    continue
                best = np.inf
                for current_mode in Mode:
                    tr = self.transition_cost(u, v, current_mode, time)
                    if tr.feasible:
                        cost = tr.objective_cost
                        if not include_transfer_costs:
                            cost -= tr.handling_time_minutes * self.config.time_cost_per_minute
                            cost -= tr.monetary_cost * self.config.monetary_cost_weight
                        best = min(best, cost)
                matrix[u, v] = best
        return matrix

    def solver_data(self) -> Dict[str, Any]:
        """Return JSON-serializable data for OR-Tools, MILP, CP-SAT, etc."""
        return {
            "config": asdict(self.config),
            "depot_node": self.depot_node,
            "end_depot_node": self.end_depot_node,
            "delivery_nodes": self.delivery_nodes.astype(int).tolist(),
            "nodes": {str(i): self.node_to_dict(node) for i, node in self.nodes.items()},
            "arcs": [self.arc_to_dict(arc) for arc in self.arcs.values()],
            "mode_names": {int(mode): mode.name for mode in Mode},
            "node_kind_names": {int(kind): kind.name for kind in NodeKind},
        }

    def edge_list(self) -> List[Dict[str, Any]]:
        return [self.arc_to_dict(arc) for arc in self.arcs.values()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_delivery(self, node_id: int) -> bool:
        return node_id >= self.delivery_start

    def _delivery_index(self, node_id: int) -> int:
        return int(node_id - self.delivery_start)

    def _is_delivered(self, node_id: int) -> bool:
        if not self._is_delivery(node_id):
            return False
        return bool(self.delivered[self._delivery_index(node_id)] == 1)

    def _is_terminal(self) -> bool:
        all_delivered = bool(np.all(self.delivered == 1))
        if not all_delivered:
            return False
        if self.config.require_return_to_depot:
            return self.current_node == self.end_depot_node
        return True

    def _info(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "current_node_name": self.nodes[self.current_node].name,
            "current_mode_name": self.current_mode.name,
            "delivered_count": int(self.delivered.sum()),
            "remaining_delivery_nodes": self.delivery_nodes[self.delivered == 0].astype(int).tolist(),
            "all_delivered": bool(np.all(self.delivered == 1)),
        }
        if extra:
            info.update(extra)
        return info

    @staticmethod
    def node_to_dict(node: NodeSpec) -> Dict[str, Any]:
        data = asdict(node)
        data["kind"] = int(node.kind)
        data["kind_name"] = node.kind.name
        return data

    @staticmethod
    def arc_to_dict(arc: ArcSpec) -> Dict[str, Any]:
        data = asdict(arc)
        data["mode"] = int(arc.mode)
        data["mode_name"] = arc.mode.name
        return data


if __name__ == "__main__":
    env = VeniceTSPEnv(config=VeniceTSPConfig(num_delivery_nodes=8, num_docks=6, seed=42))
    obs, info = env.reset(seed=123)
    print("Initial info:", info)
    route = env.greedy_feasible_route()
    result = env.simulate_route(route)
    print("Greedy route:", route)
    print("Feasible:", result["feasible"])
    print("Cost:", round(result["objective_cost"], 2))
    print("Violations:", result["violations"])
