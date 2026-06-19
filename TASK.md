# TASK.md

## Overview

This project explores solving realistic Travelling Salesman Problems (TSPs) using machine learning and optimization techniques. Rather than using abstract Euclidean graphs, the goal is to construct geographically grounded routing problems that incorporate real-world constraints and operational considerations.

The objective is not simply to minimize distance, but to investigate how domain-specific knowledge, environmental constraints, and transportation networks influence route planning and solution quality.

---

## Geography Selection: Venice, Italy

### Why Venice?

I visited Venice last year and something I noticed was that it is one of the most unusual logistics environments in the world. That uniqueness makes it a particularly interesting setting for a TSP problem.

In most cities, routing can largely be reduced to finding efficient paths through a road network. Venice fundamentally changes that assumption. Cars and trucks cannot access most of the historic city center, forcing deliveries to move through a combination of roads, canals, docks, bridges, and pedestrian pathways.

As a result, Venice introduces several real-world constraints that directly influence route planning and transform a standard TSP into a much richer optimization problem.

### Real-World Constraints

#### Multi-Modal Transportation

Transportation in Venice is inherently multi-modal.

Three transportation modes are modeled:

1. **Van**

   * Mainland only
   * Used for depot staging

2. **Cargo Boat**

   * Canal transportation
   * Connects logistics hubs and docks

3. **Hand Cart / Walking Courier**

   * Final-mile delivery
   * Required for most delivery addresses

Because deliveries cannot be completed using a single mode of transport, routing decisions must account for transitions between different transportation networks.

#### Transfer Operations

Moving between transportation modes incurs additional costs.

Examples include:

* Van → Boat
* Boat → Hand Cart

These transfers introduce:

* Handling time
* Loading and unloading delays
* Operational costs

A route that appears shorter geographically may therefore be less efficient if it requires excessive transfers.

#### Bridges

Venice's pedestrian network is connected by hundreds of bridges.

For hand-cart deliveries, bridges introduce additional effort through:

* Stair climbing
* Cart handling effort
* Reduced movement speed

This means that two locations with similar geographic distances may have significantly different delivery costs.

#### Tourist Congestion

Certain districts experience heavy pedestrian traffic throughout the day.

In particular:

* San Marco
* Rialto / San Polo

Travel through these areas may be substantially slower during peak tourist hours, introducing time-dependent routing decisions.

#### Acqua Alta (Flooding)

Venice is periodically affected by tidal flooding events.

Flooding can:

* Slow pedestrian movement
* Increase route costs
* Create temporary operational disruptions

This introduces environmental uncertainty that is rarely present in traditional TSP formulations.

#### Delivery Time Windows

Each delivery location contains:

* Earliest service time
* Latest service time

Late deliveries incur penalties, forcing the routing strategy to balance efficiency with schedule adherence.

---

## Goal

Develop and train a simple machine learning model that attempts to solve the Venice TSP environment.

The model will be evaluated not only on route efficiency, but also on its ability to operate under the real-world constraints introduced by Venice's transportation network.

Performance will be compared against common TSP and routing solver libraries to establish a benchmark and better understand the strengths and limitations of a learned approach in a realistic logistics setting.
