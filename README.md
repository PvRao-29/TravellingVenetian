# Travelling Venetian

Venice multimodal delivery routing as a realistic Travelling Salesman / vehicle-routing problem. Instead of abstract Euclidean TSP, deliveries move through **van → boat → hand cart** networks with bridges, tourist congestion, tidal flooding, and delivery time windows.

This repo is the submission artifact for the Aurelius technical task: a **Jupyter notebook** that builds the environment, benchmarks classical solvers, documents an RL attempt, and trains a **simple supervised ML model** that nearly matches OR-Tools.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (or Python 3.14+ with pip).

```bash
uv sync
uv run jupyter notebook venice_experiment_pipeline.ipynb
```

Run all cells top-to-bottom. Part A (~10 min) re-runs classical baselines; Part D (~3 min) trains the supervised ranker unless a checkpoint already exists at `runs/ml_ranker/ranker.pt` (bundled in this repo).

## Results (20 held-out instances, 15 deliveries)

| Method | Mean objective ↓ | Mean lateness (min) ↓ | Solve time |
|---|---:|---:|---:|
| OR-Tools soft (benchmark) | 777 | 31 | ~10 s |
| **Supervised ranker (ML)** | **893** | **44** | **~9 ms** |
| Earliest time window | 1,555 | 161 | ~8 ms |
| Nearest neighbor | 8,346 | 1,511 | ~8 ms |
| Random | 23,979 | 4,189 | ~11 ms |

The learned model is within ~15% of the OR-Tools benchmark while running ~1,000× faster. Full per-seed records: `results/venice_with_ranker_results.csv`.

## Repository layout

| Path | Purpose |
|---|---|
| `venice_experiment_pipeline.ipynb` | **Main submission** — environment, baselines, DQN notes, ML model, comparison |
| `venice.py` | Gymnasium environment (multimodal graph, constraints, `simulate_route`) |
| `ORtools.py` | OR-Tools bridge (none / soft / strict time windows) |
| `ml_ranker.py` | Supervised delivery-order model (imitates OR-Tools soft) |
| `dqn_agent.py`, `dqn_train.py` | Double-DQN attempt (documented failure in notebook) |
| `modal_train.py` | Optional GPU training on Modal + W&B |
| `TASK.md` | Problem motivation — why Venice, real-world constraints |
| `results/` | Experiment CSV outputs |
| `runs/ml_ranker/ranker.pt` | Pre-trained supervised ranker checkpoint |

## Approaches

1. **Classical baselines** — random, nearest-neighbor, earliest-due-date greedy.
2. **OR-Tools** — mode-expanded shortest-path graph + routing with soft/strict time windows.
3. **Supervised ranker (`ml_ranker.py`)** — OR-Tools soft solutions provide training labels; a small MLP learns which delivery to visit next from leg cost, time windows, slack/lateness, and flood flags. Greedy inference expands into an executable multimodal route.
4. **DQN (attempted)** — masked Double-DQN did not converge to a usable policy (0% greedy success after 26k GPU episodes). Kept in repo as an honest negative result; see notebook Part C.1.

## Optional commands

```bash
# Train the supervised ranker standalone (~2–4 min CPU)
uv run python ml_ranker.py

# Train DQN locally (not required for submission)
uv run python dqn_train.py --episodes 2000 --deliveries 15 --docks 6 --seed 7

# Overnight DQN on Modal (optional; requires Modal + W&B secrets)
uv run modal run modal_train.py
```

## Dependencies

Core: `gymnasium`, `networkx`, `numpy`, `ortools`, `torch`.

Optional cloud extras: `uv sync --extra cloud` installs `modal` and `wandb`.
