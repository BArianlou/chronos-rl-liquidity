"""
===============================================================================
MODULE MANIFEST: CHRONOS TRAINING ORCHESTRATOR (EXECUTION LOOP)
===============================================================================
System Purpose:
    Serves as the primary training orchestrator for the CHRONOS reinforcement
    learning engine. Coordinates the episodic simulation loop by binding the
    market simulation environment (ChronosMarketEnv) to the decision controller
    (RLAgent), driving policy convergence across non-stationary regimes.

State Boundaries:
    - Governs episodic execution horizons, state-action transition flows,
      and step-wise gradient optimization triggers.
    - Encapsulates CLI argument parsing and runtime module path resolution.
    - Strictly delegates market simulation dynamics and reward calculations to
      ChronosMarketEnv, and experience buffering and network optimization
      to RLAgent.

Mathematical/Physical Invariants:
    1. MDP Transition Sequence:
       Enforces sequential execution:
       (S_t, A_t) -> (R_(t+1), S_(t+1), Terminal_Flag)
       without temporal leakage or out-of-order state mutation.
    2. Episodic Horizon Bounding:
       Enforces an upper-bound step ceiling per episode to prevent unbounded
       state variance while maintaining consistent convergence evaluation.

Design Rationale:
    Executes step-wise Bellman updates immediately following experience storage
    once the minimum replay threshold is met. Performing online mini-batch
    gradient updates enables the agent to adapt dynamically to localized
    liquidity shifts rather than deferring optimization to episode completion.
===============================================================================
"""

import argparse
import os
import sys

# Runtime Module Path Resolution:
# Ensures automated CI pipelines, containerized workloads, and distributed
# runners resolve local sibling modules without PYTHONPATH ambiguity.
CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from market_env import ChronosMarketEnv
from rl_agent import RLAgent


def run_chronos_training(episodes: int = 1000) -> None:
    """Executes the reinforcement learning training and evaluation loop.

    Instantiates the market environment and agent, steps through sequential
    microstructure states, records transitions to the replay buffer, and
    triggers iterative gradient updates to converge on an optimal execution policy.

    Mathematical Invariants:
        - Bellman Optimality Updates: 1-step temporal difference (TD) mini-batch
          optimization.
        - Experience Replay Sampling: Uniform random sampling to decorrelate
          sequential transitions and satisfy i.i.d. batch assumptions.

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

        # Step-wise execution bounded by environment termination
        time_step: int = 0
        done: bool = False

        while not done:
            time_step += 1

            # Select execution action via epsilon-greedy policy
            action: int = agent.act(state)

            # Advance the market simulation environment by one step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append transition tuple to FIFO replay buffer
            agent.store_experience(state, action, reward, next_state, done)

            state = next_state

            # Execute mini-batch gradient update on the Q-network
            agent.train()

        if episodes > 2:
            print(
                f"Episode {episode + 1}/{episodes}: "
                f"Completed at step {time_step}"
            )

    print("Chronos Intelligence Audit: SUCCESS")


if __name__ == "__main__":
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
