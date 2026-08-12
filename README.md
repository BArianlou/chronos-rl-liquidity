[![Chronos Intelligence Audit](https://github.com/BArianlou/chronos-rl-liquidity/actions/workflows/chronos_ci.yml/badge.svg)](https://github.com/BArianlou/chronos-rl-liquidity/actions/workflows/chronos_ci.yml)
# Chronos: Deep Reinforcement Learning Liquidity Engine

![AI Framework](https://img.shields.io/badge/AI-TensorFlow%20%2F%20Keras-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-DQN%20%2B%20PPO-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Architect:** Bijan Arianlou | **Role:** Principal Systems Architect
**Status:** Alpha Validation (v0.9) | **Core Logic:** Deep Q-Network (DQN)

---

## 1. Architectural Intent
Chronos is an autonomous execution agent designed to optimize liquidity entry/exit points using Deep Reinforcement Learning (DRL). Unlike static algorithmic trading scripts, Chronos utilizes a **Dynamic Policy Network** that adapts to changing market volatility regimes (Regime Switching), maximizing Sharpe Ratio while minimizing slippage.

### Language & System Integration
* **Core Reference Engine:** Python 3.x (TensorFlow, Keras, Gymnasium) — *contained in this public repository*.
* **Enterprise Execution Layer:** Distributed data pipeline and streaming telemetry wrappers leverage **Scala (Apache Spark Core / Spark Streaming)** for high-throughput temporal data processing.

## 2. Agent-Environment Interaction
The system operates on a continuous feedback loop where the Agent observes the Market State (Order Book + Volatility) and outputs an Action (Limit/Market Order), receiving a Reward based on PnL efficiency.

### Learning Loop (Live Render)
```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> Market_Environment : Load Historical Ticks

    state "Reinforcement Learning Cycle" as RL_Loop {
        Market_Environment --> Agent_Observation : State Vector (t)
        Agent_Observation --> Policy_Network_DQN : Tensor Input
        Policy_Network_DQN --> Action_Space : Compute Q-Values (Argmax)
        Action_Space --> Market_Environment : Execute Trade
        Market_Environment --> Reward_Function : Calculate Risk-Adj Return
        Reward_Function --> Policy_Network_DQN : Backpropagate Gradients
    }

    RL_Loop --> Terminate : Episode Complete
    Terminate --> [*]
```

### Core Capabilities
*   **Adaptive Policy:** Uses Deep Q-Networks (DQN).
*   **Reward Shaping:** Minimizes drawdown.
*   **Microstructure Awareness:** Order Book Imbalance (OBI).

---

## Implementation Notice
This repository contains the **Environment Wrappers**.
> *Contact the Architect for backtest performance reports.*
  

