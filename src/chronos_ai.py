import tensorflow as tf
from tensorflow.keras import layers

class ChronosAI:
    """Deep Q-Network backbone for liquidity-sensitive decision logic."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            # Ge's Syntax Fix: Modern Keras Input declaration
            layers.Input(shape=(self.state_size,)),
            
            # Triune Fail-Safe 1: State-Space Clamping
            # Normalizes high-variance liquidity and volume metrics
            layers.BatchNormalization(),
            
            layers.Dense(256, activation="relu"),
            
            # Triune Fail-Safe 2: Regime Regularization
            # Prevents the agent from overfitting to a single market regime (e.g., only bull markets)
            layers.Dropout(0.2),
            
            layers.Dense(128, activation="relu"),
            
            # Linear activation outputs raw Q-values for execution sequencing
            layers.Dense(self.action_size, activation="linear") 
        ])
        
        # Triune Fail-Safe 3: Kinetic Gradient Control
        # Huber loss mathematically caps gradient explosions during flash crashes or liquidity spikes
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss=tf.keras.losses.Huber()
        )
        return model
