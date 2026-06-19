"""
OR-Tools optimizer bridge for the optimization-ready VeniceTSPEnv.

This solver treats the Gym environment as the source of truth and builds a
mode-expanded shortest-path graph around it. OR-Tools chooses the order of
required deliveries; this module then expands that order back into executable
hub/dock/delivery actions and validates the resulting route with env.simulate_route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from venice import Mode, NodeKind, VeniceTSPConfig, VeniceTSPEnv

State = Tuple[int, Mode]


@dataclass(frozen=True)
class LegPath:
    """Shortest executable path between two required physical nodes."""

    origin: int
    destination: int
    start_mode: Mode
    end_mode: Mode
    objective_cost: float
    time_minutes: float
    state_path: Tuple[State, ...]
    physical_actions: Tuple[int, ...]


@dataclass(frozen=True)
class SolverResult:
    """Structured result returned by solve_with_ortools."""

    success: bool
    objective_value: Optional[float]
    required_order: Tuple[int, ...]
    expanded_route: Tuple[int, ...]
    simulated: Mapping[str, Any]
    leg_paths: Tuple[LegPath, ...]
    message: str


def node_supports_mode(node: Any, mode: Mode) -> bool:
    if mode == Mode.VAN:
        return bool(node.road_access)
    if mode == Mode.BOAT:
        return bool(node.water_access)
    if mode == Mode.CART:
        return bool(node.cart_access)
    raise ValueError(f"Unsupported mode: {mode}")


def service_mode_for_required_node(env: VeniceTSPEnv, node_id: int, *, is_depot_end: bool = False) -> Mode:
    """
    The mode in which a required node is considered served.

    Deliveries in this environment are inland/cart-access nodes, so they must be
    reached and left as CART. The depot starts and ends as VAN. This avoids the
    common bug where a TSP matrix lets the solver arrive at a delivery as CART
    but leave it as BOAT/VAN on the next abstract leg.
    """

    if int(node_id) == int(env.depot_node):
        return Mode.VAN

    node = env.nodes[int(node_id)]
    if node.kind == NodeKind.DELIVERY:
        return Mode.CART

    # Not normally used for required nodes, but useful if you later require docks.
    if node.cart_access:
        return Mode.CART
    if node.water_access:
        return Mode.BOAT
    if node.road_access:
        return Mode.VAN
    raise ValueError(f"Node {node_id} supports no known mode")


def build_multimodal_state_graph(env: VeniceTSPEnv) -> nx.DiGraph:
    """
    Build a directed graph whose nodes are (physical_node_id, mode).

    Travel edges move between physical nodes in one mode. Transfer edges switch
    modes at the same physical node. This is the correct abstraction for Venice:
    the physical node alone is not enough state because a courier's next legal
    action depends on whether they are currently in a van, boat, or hand cart.
    """

    cfg = env.config
    graph = nx.DiGraph()

    for node_id, node in env.nodes.items():
        for mode in Mode:
            if node_supports_mode(node, mode):
                graph.add_node((int(node_id), mode))

    def add_transfer(node_id: int, a: Mode, b: Mode, minutes: float, money: float) -> None:
        node = env.nodes[int(node_id)]
        if not (node_supports_mode(node, a) and node_supports_mode(node, b)):
            return
        objective = minutes * cfg.time_cost_per_minute + money * cfg.monetary_cost_weight
        graph.add_edge(
            (int(node_id), a),
            (int(node_id), b),
            kind="transfer",
            time=float(minutes),
            money=float(money),
            cost=float(objective),
            action_node=None,
        )
        graph.add_edge(
            (int(node_id), b),
            (int(node_id), a),
            kind="transfer",
            time=float(minutes),
            money=float(money),
            cost=float(objective),
            action_node=None,
        )

    for node_id in env.nodes:
        add_transfer(node_id, Mode.VAN, Mode.BOAT, cfg.van_boat_transfer_minutes, cfg.van_boat_transfer_cost)
        add_transfer(node_id, Mode.BOAT, Mode.CART, cfg.boat_cart_transfer_minutes, cfg.boat_cart_transfer_cost)
        add_transfer(node_id, Mode.VAN, Mode.CART, cfg.van_cart_transfer_minutes, cfg.van_cart_transfer_cost)

    for (u, v, mode), arc in env.arcs.items():
        base_time = float(arc.base_time_minutes)
        bridge_time = float(arc.bridges * cfg.bridge_delay_minutes if mode == Mode.CART else 0.0)
        minutes = base_time + bridge_time
        objective = minutes * cfg.time_cost_per_minute + float(arc.monetary_cost) * cfg.monetary_cost_weight
        graph.add_edge(
            (int(u), mode),
            (int(v), mode),
            kind="travel",
            time=float(minutes),
            money=float(arc.monetary_cost),
            cost=float(objective),
            distance_m=float(arc.distance_m),
            bridges=int(arc.bridges),
            action_node=int(v),
        )

    return graph


def state_path_to_physical_actions(graph: nx.DiGraph, state_path: Sequence[State]) -> Tuple[int, ...]:
    """
    Convert a mode-expanded path into env.step/simulate_route physical actions.

    Transfers at the same physical node do not become actions. Only travel edges
    produce physical action nodes.
    """

    actions: List[int] = []
    for a, b in zip(state_path, state_path[1:]):
        edge = graph[a][b]
        action_node = edge.get("action_node")
        if action_node is not None:
            node = int(action_node)
            if not actions or actions[-1] != node:
                actions.append(node)
    return tuple(actions)


def shortest_leg_path(
    graph: nx.DiGraph,
    *,
    origin: int,
    destination: int,
    start_mode: Mode,
    end_mode: Mode,
    blocked_nodes: Iterable[int] = (),
) -> Optional[LegPath]:
    """
    Shortest executable path from (origin, start_mode) to (destination, end_mode).

    ``blocked_nodes`` lists physical node ids that the leg may not pass through.
    This is essential when stitching abstract TSP legs together: without it,
    Dijkstra happily routes one leg straight through another required delivery
    node, which then gets emitted as a physical action and is "served" out of
    order, producing revisits and broken mode continuity in the expanded route.
    """

    start: State = (int(origin), start_mode)
    end: State = (int(destination), end_mode)
    if not graph.has_node(start) or not graph.has_node(end):
        return None

    blocked = {int(n) for n in blocked_nodes} - {int(origin), int(destination)}
    if blocked:
        def _allow_state(state: State) -> bool:
            return int(state[0]) not in blocked

        search_graph: nx.DiGraph = nx.subgraph_view(graph, filter_node=_allow_state)
    else:
        search_graph = graph

    try:
        objective_cost, path = nx.single_source_dijkstra(search_graph, start, end, weight="cost")
    except nx.NetworkXNoPath:
        return None

    path_time = sum(float(graph[path[k]][path[k + 1]]["time"]) for k in range(len(path) - 1))
    actions = state_path_to_physical_actions(graph, path)
    return LegPath(
        origin=int(origin),
        destination=int(destination),
        start_mode=start_mode,
        end_mode=end_mode,
        objective_cost=float(objective_cost),
        time_minutes=float(path_time),
        state_path=tuple(path),
        physical_actions=actions,
    )


def leg_env_travel_minutes(env: VeniceTSPEnv, leg: LegPath, depart_time_minutes: float) -> float:
    """
    Replay a leg through ``env.transition_cost`` for environment-faithful timing.

    The static mode-expanded graph ignores tourist crowds, acqua alta, and
    earliness waiting. Use this when building OR-Tools time windows for strict
    or soft routing.
    """

    node = int(leg.origin)
    mode = leg.start_mode
    time = float(depart_time_minutes)
    for action in leg.physical_actions:
        transition = env.transition_cost(node, int(action), mode, time)
        if not transition.feasible:
            return float("inf")
        time = transition.completion_time_minutes
        mode = transition.end_mode
        node = int(action)
    return max(0.0, time - float(depart_time_minutes))


def build_required_node_matrices(
    env: VeniceTSPEnv,
    graph: nx.DiGraph,
    required_nodes: Sequence[int],
    *,
    cost_scale: int = 100,
    unreachable_cost: int = 10**9,
    use_env_travel_times: bool = False,
    depart_time_minutes: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], LegPath], List[Tuple[int, int]]]:
    """
    Build OR-Tools cost/time matrices over only depot + delivery nodes.

    Unlike the naive version, this does not minimize over arbitrary start/end
    modes. It uses the service mode of each physical required node, preserving
    mode continuity across abstract TSP legs.
    """

    n = len(required_nodes)
    cost_matrix = np.zeros((n, n), dtype=np.int64)
    time_matrix = np.zeros((n, n), dtype=np.int64)
    leg_paths: Dict[Tuple[int, int], LegPath] = {}
    unreachable_pairs: List[Tuple[int, int]] = []

    all_required = {int(node) for node in required_nodes}

    for i, origin in enumerate(required_nodes):
        for j, destination in enumerate(required_nodes):
            if i == j:
                continue

            start_mode = service_mode_for_required_node(env, origin)
            end_mode = service_mode_for_required_node(env, destination, is_depot_end=(destination == env.end_depot_node))
            leg = shortest_leg_path(
                graph,
                origin=int(origin),
                destination=int(destination),
                start_mode=start_mode,
                end_mode=end_mode,
                blocked_nodes=all_required,
            )
            if leg is None:
                cost_matrix[i, j] = unreachable_cost
                time_matrix[i, j] = unreachable_cost
                unreachable_pairs.append((int(origin), int(destination)))
                continue

            travel_minutes = leg.time_minutes
            if use_env_travel_times:
                start_time = (
                    env.config.base_hour * 60.0 if depart_time_minutes is None else float(depart_time_minutes)
                )
                travel_minutes = leg_env_travel_minutes(env, leg, start_time)

            cost_matrix[i, j] = max(0, int(round(leg.objective_cost * cost_scale)))
            time_matrix[i, j] = max(0, int(round(travel_minutes)))
            leg_paths[(i, j)] = leg

    return cost_matrix, time_matrix, leg_paths, unreachable_pairs


def earliest_tw_required_order(env: VeniceTSPEnv, required_nodes: Sequence[int]) -> Tuple[int, ...]:
    """Earliest-due-date delivery order with depot at both ends."""

    deliveries = [int(node) for node in required_nodes[1:]]
    deliveries.sort(
        key=lambda node_id: (
            env.nodes[node_id].time_window_end,
            env.nodes[node_id].time_window_start,
            node_id,
        )
    )
    return (int(required_nodes[0]), *deliveries, int(required_nodes[0]))


def _simulated_lateness_minutes(simulated: Mapping[str, Any]) -> float:
    transitions = list(simulated.get("transitions", []) or [])
    return float(sum(float(t.get("lateness_minutes", 0.0)) for t in transitions))


def _format_clock(minutes: float) -> str:
    minutes = float(minutes) % 1440.0
    h = int(minutes // 60)
    m = int(round(minutes - h * 60))
    if m == 60:
        h = (h + 1) % 24
        m = 0
    return f"{h:02d}:{m:02d}"


def _extract_required_order(manager: Any, routing: Any, solution: Any, required_nodes: Sequence[int]) -> Tuple[int, ...]:
    index = routing.Start(0)
    order: List[int] = []
    while not routing.IsEnd(index):
        order.append(int(required_nodes[manager.IndexToNode(index)]))
        index = solution.Value(routing.NextVar(index))
    order.append(int(required_nodes[manager.IndexToNode(index)]))
    return tuple(order)


def expand_required_order(
    required_order: Sequence[int],
    required_nodes: Sequence[int],
    leg_paths_by_index: Mapping[Tuple[int, int], LegPath],
) -> Tuple[Tuple[int, ...], Tuple[LegPath, ...]]:
    """Expand depot/delivery order into executable env physical actions."""

    required_index = {int(node): idx for idx, node in enumerate(required_nodes)}
    expanded: List[int] = []
    legs: List[LegPath] = []

    for origin, destination in zip(required_order, required_order[1:]):
        i = required_index[int(origin)]
        j = required_index[int(destination)]
        leg = leg_paths_by_index[(i, j)]
        legs.append(leg)
        for action in leg.physical_actions:
            if not expanded or expanded[-1] != int(action):
                expanded.append(int(action))

    return tuple(expanded), tuple(legs)


def _first_solution_strategies(time_window_mode: str) -> Tuple[int, ...]:
    from ortools.constraint_solver import routing_enums_pb2

    if time_window_mode == "strict":
        return (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        )
    return (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,)


def _evaluate_route(
    env: VeniceTSPEnv,
    *,
    required_order: Sequence[int],
    required_nodes: Sequence[int],
    leg_paths: Mapping[Tuple[int, int], LegPath],
    objective_value: Optional[float],
    message: str,
) -> SolverResult:
    expanded_route, used_legs = expand_required_order(required_order, required_nodes, leg_paths)
    simulated = env.simulate_route(
        expanded_route,
        start_node=env.depot_node,
        start_mode=Mode.VAN,
        start_time_minutes=env.config.base_hour * 60.0,
        require_all_deliveries=True,
        require_return_to_depot=env.config.require_return_to_depot,
        penalize_revisits=True,
    )
    return SolverResult(
        success=bool(simulated.get("feasible", False)),
        objective_value=objective_value,
        required_order=tuple(int(node) for node in required_order),
        expanded_route=expanded_route,
        simulated=simulated,
        leg_paths=used_legs,
        message=message,
    )


def solve_with_ortools(
    env: VeniceTSPEnv,
    *,
    strict_time_windows: bool = False,
    time_window_mode: Optional[str] = None,
    time_limit_seconds: int = 15,
    cost_scale: int = 100,
    route_horizon_minutes: int = 1440,
    slack_max_minutes: int = 720,
    log_search: bool = False,
    verbose: bool = True,
) -> SolverResult:
    """
    Solve the delivery order using OR-Tools, then validate in the environment.

    ``time_window_mode`` selects how delivery time windows are enforced:

    * ``"none"``   - ignore time windows entirely (pure cost minimization).
    * ``"soft"``   - allow lateness but penalize it (good for synthetic instances
                     because it returns a route with lateness penalties instead of
                     nothing).
    * ``"strict"`` - hard feasibility only; deliveries must land inside the window.

    If ``time_window_mode`` is None it is derived from ``strict_time_windows`` for
    backward compatibility (``True`` -> ``"strict"``, ``False`` -> ``"soft"``).
    """

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    if time_window_mode is None:
        time_window_mode = "strict" if strict_time_windows else "soft"
    time_window_mode = str(time_window_mode).lower()
    if time_window_mode not in {"none", "soft", "strict"}:
        raise ValueError(f"time_window_mode must be 'none', 'soft', or 'strict'; got {time_window_mode!r}")

    graph = build_multimodal_state_graph(env)
    delivery_nodes = [int(x) for x in env.delivery_nodes.tolist()]
    required_nodes = [int(env.depot_node)] + delivery_nodes
    use_env_travel_times = time_window_mode == "strict"

    cost_matrix, time_matrix, leg_paths, unreachable_pairs = build_required_node_matrices(
        env,
        graph,
        required_nodes,
        cost_scale=cost_scale,
        use_env_travel_times=use_env_travel_times,
    )

    if unreachable_pairs:
        message = f"Unreachable required pairs detected: {unreachable_pairs[:10]}"
        if verbose:
            print(message)
        return SolverResult(
            success=False,
            objective_value=None,
            required_order=tuple(),
            expanded_route=tuple(),
            simulated={"feasible": False, "violations": [message]},
            leg_paths=tuple(),
            message=message,
        )

    manager = pywrapcp.RoutingIndexManager(len(required_nodes), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cost_callback(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(cost_matrix[i, j])

    cost_callback_index = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

    def time_callback(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        physical_from_node = int(required_nodes[i])
        service_time = int(round(env.nodes[physical_from_node].service_time_minutes))
        return int(time_matrix[i, j]) + service_time

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        int(slack_max_minutes),
        int(route_horizon_minutes),
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    start_minutes = int(round(env.config.base_hour * 60.0))
    start_index = routing.Start(0)
    time_dimension.CumulVar(start_index).SetRange(start_minutes, start_minutes)

    for required_idx, physical_node in enumerate(required_nodes[1:], start=1):
        node = env.nodes[int(physical_node)]
        index = manager.NodeToIndex(required_idx)
        service = int(round(node.service_time_minutes))
        earliest_start = int(round(node.time_window_start))
        latest_start = max(earliest_start, int(round(node.time_window_end)) - service)

        if time_window_mode == "strict":
            time_dimension.CumulVar(index).SetRange(earliest_start, latest_start)
        elif time_window_mode == "soft":
            time_dimension.CumulVar(index).SetRange(earliest_start, int(route_horizon_minutes))
            penalty = max(1, int(round(env.config.lateness_cost_per_minute * cost_scale)))
            time_dimension.SetCumulVarSoftUpperBound(index, latest_start, penalty)
        else:  # "none": leave the delivery time unconstrained by its window
            time_dimension.CumulVar(index).SetRange(0, int(route_horizon_minutes))

    # Make the final depot time visible and bounded.
    end_index = routing.End(0)
    time_dimension.CumulVar(end_index).SetRange(start_minutes, int(route_horizon_minutes))

    if verbose:
        print(
            f"Solving {len(delivery_nodes)} deliveries with {len(graph.nodes)} mode-states "
            f"and {len(graph.edges)} state-edges (mode={time_window_mode})..."
        )

    solution = None
    objective_value: Optional[float] = None
    for strategy in _first_solution_strategies(time_window_mode):
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = strategy
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(int(time_limit_seconds))
        search_parameters.log_search = bool(log_search)
        solution = routing.SolveWithParameters(search_parameters)
        if solution is not None:
            objective_value = float(solution.ObjectiveValue()) / float(cost_scale)
            break

    if solution is not None:
        required_order = _extract_required_order(manager, routing, solution, required_nodes)
        result = _evaluate_route(
            env,
            required_order=required_order,
            required_nodes=required_nodes,
            leg_paths=leg_paths,
            objective_value=objective_value,
            message="ok",
        )
        if result.success:
            if time_window_mode == "strict":
                ortools_late = _simulated_lateness_minutes(result.simulated)
                if ortools_late <= 1e-6:
                    if verbose:
                        print_solution_summary(env, result)
                    return result
            else:
                if verbose:
                    print_solution_summary(env, result)
                return result
        if time_window_mode != "strict":
            result = SolverResult(
                success=False,
                objective_value=objective_value,
                required_order=result.required_order,
                expanded_route=result.expanded_route,
                simulated=result.simulated,
                leg_paths=result.leg_paths,
                message="OR-Tools order found, but environment simulation reports violations.",
            )
            if verbose:
                print_solution_summary(env, result)
            return result

    # Strict instances can be tight for PATH_CHEAPEST_ARC even when a feasible
    # insertion / EDD tour exists. Fall back to a time-window-aware construction.
    if time_window_mode == "strict":
        edd_order = earliest_tw_required_order(env, required_nodes)
        edd_result = _evaluate_route(
            env,
            required_order=edd_order,
            required_nodes=required_nodes,
            leg_paths=leg_paths,
            objective_value=objective_value,
            message="ok (EDD fallback)",
        )
        if edd_result.success:
            if verbose:
                print_solution_summary(env, edd_result)
            return edd_result

    message = "No OR-Tools solution found. Try soft/none time windows or a longer time limit."
    if verbose:
        print(message)
    return SolverResult(
        success=False,
        objective_value=None,
        required_order=tuple(),
        expanded_route=tuple(),
        simulated={"feasible": False, "violations": [message]},
        leg_paths=tuple(),
        message=message,
    )


def print_solution_summary(env: VeniceTSPEnv, result: SolverResult) -> None:
    """Human-readable display for notebooks/terminal runs."""

    if not result.required_order:
        print(result.message)
        return

    print("\nRequired-node order chosen by OR-Tools:")
    for node_id in result.required_order:
        node = env.nodes[int(node_id)]
        print(f"  {node_id:>3}  {node.name}")

    print("\nExpanded executable route, including hidden docks/hubs:")
    for step, node_id in enumerate(result.expanded_route, start=1):
        node = env.nodes[int(node_id)]
        print(f"  {step:>3}. {node_id:>3}  {node.name}")

    sim = result.simulated
    print("\nValidation against env.simulate_route:")
    print(f"  Feasible:       {sim.get('feasible')}")
    print(f"  Delivered:      {sim.get('delivered_count')}/{env.num_delivery_nodes}")
    print(f"  Objective cost: {float(sim.get('objective_cost', float('nan'))):.2f}")
    print(f"  End time:       {_format_clock(float(sim.get('end_time_minutes', 0.0)))}")
    print(f"  End node:       {sim.get('end_node')} ({env.nodes[int(sim.get('end_node'))].name if sim.get('end_node') in env.nodes else 'unknown'})")
    if sim.get("violations"):
        print("  Violations:")
        for violation in sim["violations"]:
            print(f"    - {violation}")


def main() -> None:
    config = VeniceTSPConfig(num_delivery_nodes=15, num_docks=6, seed=7, base_hour=8.0)
    env = VeniceTSPEnv(config=config)
    env.reset(seed=123)

    solve_with_ortools(
        env,
        strict_time_windows=False,
        time_limit_seconds=15,
        slack_max_minutes=720,
        log_search=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
