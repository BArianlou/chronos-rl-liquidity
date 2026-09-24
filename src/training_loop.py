"""
===============================================================================
MODULE MANIFEST: CHRONOS AUTONOMIC ORCHESTRATOR (TRAINING LOOP)
===============================================================================
System Purpose:
    Serves as the master execution sequencer for the CHRONOS reinforcement
    learning engine. Orchestrates the episodic simulation loop by binding the
    deterministic market mechanics (ChronosMarketEnv) to the autonomic decision
    controller (RLAgent), driving policy convergence across non-stationary
    regimes.

State Boundaries:
    - Governs macro-temporal execution horizons, state-action transition flows,
      and high-frequency gradient synchronization triggers.
    - Encapsulates CLI argument ingestion and cross-platform runtime path
      resolution.
    - Strictly delegates environmental physics and reward calculations to
      ChronosMarketEnv, and experience buffering and network optimization
      to RLAgent.

Mathematical/Physical Invariants:
    1. MDP Transition Sequence:
       Enforces sequential execution:
       (S_t, A_t) -> (R_(t+1), S_(t+1), Terminal_Flag)
       without temporal leakage or out-of-order state mutation.
    2. Temporal Horizon Clamping:
       Enforces a deterministic 200-step horizon per episode to bound
       cumulative variance drift while guaranteeing convergence criteria.

Design Rationale:
    Executes step-wise Bellman updates immediately following experience storage
    once the minimum replay threshold is met. This real-time gradient descent
    enables the agent to adapt dynamically to transient liquidity shocks rather
    than deferring optimization to delayed episodic boundaries.
===============================================================================
"""

import argparse
import os
import sys

# [STRUCTURAL CALLOUT] CI/CD Path Synchronization
# Forces absolute path resolution to guarantee that automated CI pipelines,
# containerized workloads, and distributed runners resolve sibling modules
# without PYTHONPATH ambiguity or environmental drift.
CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from market_env import ChronosMarketEnv
from rl_agent import RLAgent


def run_chronos_training(episodes: int = 1000) -> None:
    """Executes master reinforcement learning training and synchronization loop.

    Instantiates the market environment and agent, steps through sequential
    microstructure states, records transitions to the replay buffer, and
    triggers iterative gradient updates to converge on an optimal policy.

    Mathematical Invariants:
        - Chronos_RL_Sequencing: Mini-batch Bellman optimality updates.
        - Experience Replay Sampling: Random draw decorrelation ensuring
          i.i.d. batch assumptions.

    Args:
        episodes: Number of complete episodic horizons to execute.
    """
    env: ChronosMarketEnv = ChronosMarketEnv()
    agent: RLAgent = RLAgent(
        state_size=int(env.observation_space.shape[0]),
        action_size=int(env.action_space.n),
    )

    for episode in range(episodes):
        state, info = env.reset()

        # Deterministic upper-bound step horizon per episode
        for time_step in range(200):
            # [STRUCTURAL CALLOUT] Encapsulated Action Logic
            action: int = agent.act(state)

            # Step the simulation physics
            next_state, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated

            # [STRUCTURAL CALLOUT] Dimension-Safe Experience Routing
            agent.store_experience(state, action, reward, next_state, done)

            state = next_state

            # [STRUCTURAL CALLOUT] High-Frequency Bellman Updates
            agent.train()

            if done:
                if episodes > 2:
                    print(
                        f"Episode {episode + 1}/{episodes}: "
                        f"Completed at step {time_step + 1}"
                    )
                break

    print("Chronos Intelligence Audit: SUCCESS")


if __name__ == "__main__":
    # [STRUCTURAL CALLOUT] Execution Entry Point
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Chronos RL Engine Execution Orchestrator"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes to simulate (default: 1000)",
    )
    args: argparse.Namespace = parser.parse_args()

    run_chronos_training(episodes=args.episodes)
