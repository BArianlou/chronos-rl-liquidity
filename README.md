# CHRONOS: Deep Reinforcement Learning Liquidity & Execution Engine

[![Enterprise MLOps & Integrity Pipeline](https://github.com/BArianlou/chronos-rl-liquidity/actions/workflows/enterprise-mlops.yml/badge.svg)](https://github.com/BArianlou/chronos-rl-liquidity/actions/workflows/enterprise-mlops.yml)
[![Chronos Intelligence Audit](https://github.com/BArianlou/chronos-rl-liquidity/actions/workflows/chronos_ci.yml/badge.svg)](https://github.com/BArianlou/chronos-rl-liquidity/actions/workflows/chronos_ci.yml)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Java 17](https://img.shields.io/badge/Java-17%20LTS-orange.svg)](https://aws.amazon.com/corretto/)
[![Spark 3.5](https://img.shields.io/badge/Apache%20Spark-3.5-red.svg)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-Distributed%20Bus-black.svg)](https://kafka.apache.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Cloud%20Native-326ce5.svg)](https://kubernetes.io/)

**Architect:** Bijan Arianlou, MSEE  
**Role:** Principal Solutions & Systems Architect  
**Status:** Production Reference Architecture (v1.0)  
**Core Logic:** Deep Q-Network (DQN) Dynamic Execution & Distributed Streaming Pipeline  

---

## 1. Architectural Intent

**CHRONOS** is an autonomous execution and distributed streaming architecture engineered to optimize liquidity entry and exit sequencing across fragmented financial venues using Deep Reinforcement Learning (DRL). Moving beyond static algorithmic trading heuristics (e.g., rigid TWAP/VWAP), CHRONOS implements a regularized Deep Q-Network policy coupled with a distributed **Apache Spark / Apache Kafka** data pipeline that continuously adjusts execution actions under non-stationary market regimes, maximizing risk-adjusted return while penalizing market-impact slippage and unhedged inventory risk.

The core optimization framework optimizes risk-penalized execution returns:

$$\text{Reward}_t = \Delta \text{PnL}_t - (\lambda_1 \cdot \text{Slippage}_t) - (\lambda_2 \cdot \text{InventoryRisk}_t)$$

Where:
* $\Delta \text{PnL}_t$: Mark-to-market performance of filled order volume.
* $\lambda_1$: Penalty scalar balancing market impact and aggressive fill fees.
* $\lambda_2$: Risk aversion parameter penalizing unhedged inventory duration.

---

## 2. Language & Enterprise System Integration

### Core RL & Simulation Engine (Python 3.10)
* **TensorFlow / Keras & Gymnasium:** Deep Q-Network (DQN) policy approximator (`ChronosAI`) featuring BatchNormalization, Dropout regularization, and Huber loss gradient clipping, paired with a custom Gymnasium market microstructure environment (`ChronosMarketEnv`).
* **NumPy / SciPy:** Contiguous mini-batch tensor stacking, Bellman optimality target updates, and stochastic price/liquidity diffusion modeling.
* **Experience Replay Controller (`RLAgent`):** Manages FIFO ring-buffer replay memory, epsilon-annealing exploration schedules, and online mini-batch gradient updates.

### Enterprise Data Bus & Cloud-Native Infrastructure
* **Apache Spark (PySpark 3.5):** Distributed temporal windowing (`ChronosSparkPipeline`), rolling 100-tick volatility/depth/momentum extraction, VectorAssembler tensor packing, and distributed StandardScaler Z-score normalization.
* **Apache Kafka (Java 17 / JVM):** High-throughput Spring Kafka consumer (`ChronosKafkaConsumer.java`) handling non-blocking network I/O, partition-level event ordering, and cross-runtime memory staging.
* **Kubernetes & Cloud Infrastructure:** Production 3-replica Kubernetes deployment manifest (`chronos-liquidity-engine`) with cgroup memory/CPU bounds, liveness/readiness probes, and inter-service CoreDNS broker bindings.
* **Automated Dual-Stack CI/CD:** Ephemeral multi-runtime validation workflows executing strict AST syntax linting (Flake8), unit assertions (pytest), OpenJDK 17 ABI parity, and 2-episode end-to-end integration smoke testing.

---

## 3. Agent-Environment Interaction Loop
1. **Ingress & Distributed Streaming:** Market tick streams ingest via Apache Kafka into PySpark partitioned rolling-window transformation pipelines.
2. **State Vectorization & Standardization:** Real-time extraction and Z-score normalization of core microstructure features: Price, Rolling Volatility, Order Book Liquidity Depth, and Volume Imbalance / Inventory Position.
3. **DQN Policy Evaluation:** The TensorFlow/Keras DQN evaluates the observation state vector and selects an optimal discrete execution primitive under inventory exposure constraints.
4. **Execution & Bellman Feedback:** The environment computes realized mark-to-market PnL net of depth-dependent slippage penalties and appends transition tuples $(s, a, r, s', \text{done})$ to the replay buffer for mini-batch backpropagation.

---

## 4. Core Engineering Capabilities

* **Regime-Resilient DQN Topology:** Combines input BatchNormalization, Dropout regularization, unconstrained linear Q-value projection, and Huber loss regression to prevent gradient explosion during tail-risk market dislocations.
* **Depth-Aware Slippage Modeling:** Penalizes aggressive order execution inversely to available order book depth ($\text{Impact} = \frac{\eta}{\text{Depth}} \cdot \text{Price}$), training the policy network to minimize adverse selection and execution cost.
* **Cross-Sectional Spark Isolation:** Enforces strict `asset_id` window partitioning and warm-up null filtering across distributed parquet data lakes to prevent cross-instrument data leakage.
* **Automated Dual-Stack CI/CD Integrity Gate:** Enforces Python AST syntax enforcement (Flake8), state-space bounds validation (pytest), JVM 17 runtime compatibility, and end-to-end integration execution on every commit.

---

## 5. Repository Architecture

```text
chronos-rl-liquidity/
├── .github/
│   └── workflows/
│       ├── chronos_ci.yml            # CI micro-execution and runtime smoke validation
│       └── enterprise-mlops.yml      # Dual-stack static analysis and linting gate
├── enterprise_integration/
│   └── ChronosKafkaConsumer.java     # Java 17 Spring-Kafka ingestion consumer
├── src/
│   ├── chronos_ai.py                 # DQN architecture and Huber loss model definition
│   ├── market_env.py                 # Gymnasium-compliant market MDP environment
│   ├── rl_agent.py                   # Experience replay and Bellman policy updates
│   ├── spark_feature_pipeline.py     # PySpark distributed feature engineering
│   └── training_loop.py              # End-to-end training loop and integration harness
├── tests/
│   └── test_environment_bounds.py   # State-space shape and boundary assertions
├── requirements.txt                  # Pinned Python dependency envelope
└── README.md                         # Architecture manifest and documentation
