# CHRONOS: Deep Reinforcement Learning Liquidity Engine

![Status](https://img.shields.io/badge/Status-Alpha%20Validation-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.4+-E25A1C.svg)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-Distributed-black.svg)

**Architect:** Bijan Arianlou  
**Role:** Principal Systems Architect  
**Status:** Alpha Validation (v0.9)  
**Core Logic:** Deep Q-Network (DQN) Dynamic Liquidity Policy  

---

## 1. Architectural Intent

Chronos is an autonomous execution agent engineered to optimize liquidity entry and exit points across fragmented financial venues using Deep Reinforcement Learning (DRL). Moving beyond static algorithmic trading heuristics, Chronos implements a dynamic policy network that continuously adjusts execution vectors under non-stationary market regimes, maximizing risk-adjusted return (Sharpe Ratio) while suppressing market-impact slippage and kinetic drag.

The core optimization framework optimizes risk-penalized execution returns:

> **Reward** = PnL_Efficiency − (λ₁ · Slippage_Penalty) − (λ₂ · Max_Drawdown_Penalty)

---

## 2. Language & System Integration

### Core Reference Engine (Python 3.10)

- **PyTorch / Gymnasium:** Deep Q-Network policy backbones, target network stabilization, and custom execution environments.  
- **NumPy / SciPy:** Vectorized Bellman optimality updates, state-space covariance tracking, and numeric order book transforms.  
- Manages experience replay buffers, decay-schedules, and continuous Markov Decision Process (MDP) states.

### Enterprise Execution Layer (Distributed Streaming)

- **Apache Spark (PySpark / Scala):** Distributed temporal feature engineering, historical microstructure aggregation, and state-space drift detection.  
- **Apache Kafka:** Fault-tolerant, low-latency market depth feeds and level-2 tick streaming for continuous state vectorization.  
- Orchestrates asynchronous model checkpointing and telemetry logging without blocking hot-path execution.

---

## 3. Agent-Environment Interaction

Chronos models the market microstructure as a continuous feedback loop. At each temporal slice, the agent samples the normalized limit order book state, processes high-frequency signals, and selects an optimal allocation tactic:

- **Ingress & Streaming:** Market depth (L2/L3) and executed tick clusters stream via Apache Kafka into Spark temporal aggregation pipelines.  
- **State Vectorization:** Real-time extraction of normalized features: Order Book Imbalance (OBI), bid/ask spread, volatility, and volume skew.  
- **Policy Evaluation:** PyTorch DQN selects an optimal execution action (Passive Limit, Mid-Peg, or Aggressive Fill) under volatility regime clamping.  
- **Execution & Feedback:** Orders route to the target venue, and realized execution metrics (fill latency, slippage, and adverse selection) emit shaped reward signals back to the experience replay memory.  

```mermaid
graph TD
    %% Streaming Ingress
    subgraph INGRESS["MARKET DATA INGRESS & STREAMING"]
        direction TB
        MKT["Market Feeds: L2/L3 Depth & Executed Ticks"]
        KAFKA["Apache Kafka: Low-Latency Streaming Cluster"]
        SPARK["PySpark / Scala: Temporal Feature Aggregation & Drift Detection"]
        MKT --> KAFKA --> SPARK
    end

    %% State Vectorization
    STATE["<b>Normalized State Vector (S<sub>t</sub>)</b><br/>Order Book Imbalance (OBI) · Bid/Ask Spread · Volatility Skew · Volume Queues"]
    SPARK --> STATE

    %% Agent Policy Core
    subgraph DRL["CHRONOS AUTONOMIC DRL AGENT"]
        direction TB
        DQN["PyTorch DQN Policy Network<br/>Q(s, a; &theta;) Backbone Clamping"]
        POLICY{"Regime-Switching<br/>Action Selection"}
        
        ACT_PASS["Passive Limit (Maker)"]
        ACT_PEG["Mid-Peg Allocation"]
        ACT_AGG["Aggressive Fill (Taker)"]

        DQN --> POLICY
        POLICY --> ACT_PASS
        POLICY --> ACT_PEG
        POLICY --> ACT_AGG
    end

    STATE --> DQN

    %% Venue Execution
    VENUE["<b>Execution Venue / Matching Engine</b><br/>Order Routing & Fill Telemetry"]
    ACT_PASS --> VENUE
    ACT_PEG --> VENUE
    ACT_AGG --> VENUE

    %% Closed-Loop Feedback
    subgraph REPLAY["CLOSED-LOOP FEEDBACK & TRAINING"]
        direction TB
        METRICS["Realized Metrics: Slippage &bull; Fill Latency &bull; Adverse Selection"]
        REWARD["<b>Kinetic Reward Shaping</b><br/>R_t = PnL_Efficiency − (λ₁ · Slippage) − (λ₂ · Drawdown)"]
        BUFFER["Prioritized Experience Replay Buffer<br/>Bellman Optimality Gradient Update"]

        METRICS --> REWARD --> BUFFER
    end

    VENUE --> METRICS
    BUFFER -.->|"Asynchronous Weight Updates & Target Sync"| DQN

    %% Styling
    style INGRESS fill:#161b22,stroke:#f0883e,stroke-width:1px,color:#ffa657
    style DRL fill:#0d1117,stroke:#58a6ff,stroke-width:2px,color:#79c0ff
    style REPLAY fill:#161b22,stroke:#bc8cff,stroke-width:1px,color:#d2a8ff
    style STATE fill:#161b22,stroke:#30363d,stroke-width:1px,color:#c9d1d9
    style VENUE fill:#161b22,stroke:#238636,stroke-width:1px,color:#3fb950
    style POLICY fill:#21262d,stroke:#58a6ff,stroke-width:1px,color:#79c0ff
    style REWARD fill:#21262d,stroke:#bc8cff,stroke-width:1px,color:#d2a8ff
```

---

## 4. Core Capabilities

- **Adaptive Regime-Switching Policy:** Uses Deep Q-Networks to discover latent state transitions and execute asymmetric routing between volatile and consolidated market states.  
- **Microstructure Awareness:** Evaluates real-time Order Book Imbalance (OBI), spread compression dynamics, and bid/ask volume queues.  
- **Kinetic Reward Shaping:** Mathematically penalizes transient drawdowns, adverse selection, and inventory risk holding costs.  
- **Distributed Drift Detection:** Continuously monitors feature distribution divergence across streaming Kafka pipelines to flag execution drift.  

---

## 5. Implementation Notice

This repository contains the **Reference Architecture and Environment Wrappers**. Production deployment mandates connection to low-latency matching engine gateways, specialized tick-data infrastructure, and hardware-accelerated state storage.

For institutional integration manifests, production distributed architectures, or proprietary backtest performance documentation:  
**Contact the Architect.**

---

## 6. Repository Structure

```text
/chronos-engine          # PyTorch DQN agents, policy graphs, and network weights
/environments           # Gymnasium continuous market simulation environments
/data-pipeline          # PySpark batch jobs and Kafka temporal stream consumers
/tests                  # Deterministic validation and invariant smoke checks
Dockerfile              # Multi-stage container deployment specification
requirements.txt        # Pinned dependency graph and build constraints
```
