"""
===============================================================================
MODULE MANIFEST: CHRONOS RL CORE (DQN BACKBONE)
===============================================================================
System Purpose:
    Serves as the primary Deep Q-Network (DQN) policy approximator for the
    CHRONOS execution engine. Maps continuous multi-dimensional market state
    vectors to discrete order execution actions to maximize risk-adjusted return
    while penalizing market impact and slippage in fragmented liquidity regimes.

State Boundaries:
    - Ingests pre-processed market feature tensors from upstream pipelines.
    - Emits unbounded action-utility estimates (Q-values) to downstream order
      routers or action-selection policies (e.g., epsilon-greedy).
    - Encapsulates network topology, optimization criteria, and loss definitions.

Mathematical/Physical Invariants:
    1. Chronos_RL_Sequencing:
       Q(s, a) <- Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
    2. Liquidity_Fragmentation_Penalty (LFP):
       Absorbed through bounded gradient updates; network topology dampens
       volatility scalar shocks without gradient explosion.

Design Rationale:
    Financial time series exhibit non-stationary regimes and heavy-tailed noise.
    This architecture utilizes a regularized Multi-Layer Perceptron (MLP)
    compiled with Huber loss. This maintains deterministic bounds during tail
    events, enforcing robust policy convergence over transient momentum spikes.
===============================================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses


class ChronosAI:
    """Deep Q-Network (DQN) policy approximator for algorithmic execution.

    Acts as the function approximator for the CHRONOS reinforcement learning
    agent, mapping the continuous observation space of market microstructure
    to a discrete action valuation space.

    Attributes:
        state_size (int): Dimensionality of the ingested market state vector.
        action_size (int): Cardinality of the discrete action space.
        model (tf.keras.Model): Compiled computational graph for Q-value estimation.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        """Initializes the ChronosAI network parameters and compiles the policy model.

        Args:
            state_size: Dimensionality of the input feature vector (e.g., normalized
                spread, volume imbalance, micro-price deviation).
            action_size: Number of discrete execution primitives available to the agent.
        """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.model: tf.keras.Model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        """Constructs and compiles the neural network graph for Q-value regression.

        Compiles the network using the Adam optimizer coupled with Huber loss to
        guarantee stable gradient updates during market dislocation events.

        Mathematical Invariants:
            - State-Action Mapping: f_theta: R^(state_size) -> R^(action_size)
            - Huber Loss Metric:
                L_delta(e) = 0.5 * e^2                 for |e| <= delta
                L_delta(e) = delta * (|e| - 0.5 * delta) otherwise

        Operational Invariants:
            - Input Space: Strict 1D vector envelope per batch instance.
            - Output Space: Unconstrained real values corresponding to expected
              discounted cumulative reward per discrete action.

        Returns:
            tf.keras.Model: The assembled and compiled TensorFlow Keras model.
        """
        model = models.Sequential([
            # [STRUCTURAL CALLOUT] Tensor Projection Envelope
            # Enforces explicit dimensionality boundaries for the input state vector.
            layers.Input(shape=(self.state_size,)),

            # [STRUCTURAL CALLOUT] Triune Fail-Safe 1: State-Space Clamping
            # Normalizes input feature distributions dynamically across minibatches.
            # During inference (training=False), frozen running statistics guarantee
            # deterministic policy evaluations despite high input variance.
            layers.BatchNormalization(),

            layers.Dense(256, activation="relu"),

            # [STRUCTURAL CALLOUT] Triune Fail-Safe 2: Regime Regularization
            # Drops 20% of node connections strictly during training steps.
            # Prevents co-adaptation on localized volatility regimes while
            # remaining fully deterministic during execution inference.
            layers.Dropout(0.2),

            layers.Dense(128, activation="relu"),

            # [STRUCTURAL CALLOUT] Unbounded Utility Projection
            # Linear activation is required. Q-values denote expected cumulative
            # returns and must remain unconstrained; saturating non-linearities
            # (e.g., sigmoid, tanh) would artificially clamp value estimation.
            layers.Dense(self.action_size, activation="linear")
        ])

        # [STRUCTURAL CALLOUT] Triune Fail-Safe 3: Kinetic Gradient Control
        # Huber loss transitions smoothly between L2 norm (small tracking errors)
        # and L1 norm (large dislocative shocks), clipping gradient magnitude
        # without introducing zero-gradient dead zones.
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.Huber(delta=1.0)
        )

        return model
