import tensorflow as tf
import numpy as np
from keras_tuner.engine.hyperparameters import HyperParameters

def build_binary_classifier(input_dim,
                hidden_layer_sizes=[],
                activation='relu',
                optimizer='Adam',
                learning_rate=0.01):
    """Build a binary logistic regression model using Keras.

    Args:
    input_dim: dimension of each example as a tuple (ie for the 28 x 28 images, input_dim is (28, 28). This gets flattened
    hidden_layer_sizes: A list with the number of units in each hidden layer.
    activation: The activation function to use for the hidden layers.
    optimizer: The optimizer to use (SGD, Adam).
    learning_rate: The desired learning rate for the optimizer.

    Returns:
    model: A tf.keras model (graph).
    """
    tf.keras.backend.clear_session()
    np.random.seed(0)
    tf.random.set_seed(0)

    model = tf.keras.Sequential()
    # Add input layer
    model.add(tf.keras.Input(shape=input_dim, name='Input'))
    model.add(tf.keras.layers.Flatten(name='Flatten'))
    # Add hidden layers
    for i, hidden_layer_size in enumerate(hidden_layer_sizes):
        # Don't need to specify input shape for hidden layers
        # Bc input shape for hidden layer = units of the previous layer
        model.add(tf.keras.layers.Dense(units=hidden_layer_size,
                                        activation=activation,
                                        name=f'Hidden_{i}'))

    # Add final layer
    model.add(tf.keras.layers.Dense(
          units=1,        # output dim
          use_bias=True,               # use a bias (intercept) param
          activation='sigmoid',
          kernel_initializer=tf.ones_initializer,
          bias_initializer='zeros',
        name='Output'
      ))

    if optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError('Unknown optimizer')

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model

def model_builder_factory(input_dim):
    """

    :param input_dim
    :return: callable model_builder
    """
    def model_builder(hp: HyperParameters):
        """
        Function that defines how to build the model
        hp: the chosen set of hyperparameters. The hp code below picks whats already chosen from the list
        """
        num_hidden = hp.Int("num_hidden_layers", min_value=0, max_value=3, step=1)

        hidden_layer_sizes = []
        for i in range(num_hidden):
            units = hp.Choice(f"units_{i}", values=[16, 32, 64, 128])
            hidden_layer_sizes.append(units)

        activation = hp.Choice("activation", values=["relu", "tanh"])
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        return build_binary_classifier(
            input_dim=input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
        )
    return model_builder

def model_builder(hp: HyperParameters, input_dim):
    """
    Function that defines how to build the model
    hp: the chosen set of hyperparameters. The hp code below picks whats already chosen from the list
    """
    num_hidden = hp.Int("num_hidden_layers", min_value=0, max_value=3, step=1)

    hidden_layer_sizes = []
    for i in range(num_hidden):
        units = hp.Choice(f"units_{i}", values=[16, 32, 64, 128])
        hidden_layer_sizes.append(units)

    activation = hp.Choice("activation", values=["relu", "tanh"])
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    return build_binary_classifier(
        input_dim=input_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate=learning_rate,
    )
