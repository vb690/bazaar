from tqdm import tqdm

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout


def build_MLP(X, y, layers=(50, 25, 15), dropout=0.1, activation='relu',
              out_act='softmax'):
    """Utility function for building a Multi-layer perceptron.

    Args:
        - X: numpy array, input to the model.
        - y: numpy array, target of the model.
        - layers: tuple of ints, number of hidden units for each layer.
        - dropout: float, dropout probability.
        - activation: string, activation for the hidden units.
        - out_act: string, activation for the output units.

    Returns:
        - model: a keras model, not compiled keras model
    """
    inp = Input(shape=(X.shape[1],))
    dense = inp

    for layer_n, units in enumerate(layers):

        dense = Dense(units)(dense)
        dense = Activation(activation)(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(dropout)(dense)

    out = Dense(len(np.unique(y)))(dense)
    out = Activation('softmax')(out)

    model = Model(inp, out)

    return model


def get_weights(model, X, y, layers=(0, 2, 4), epochs=1, batch_size=1,
                **compile_kwrags):
    """Utility function for extracting the weights during batch training.

    Args:
        - model: a keras model, model from which we will extract weights during
            training.
        - X: numpy array, input to the model.
        - y: numpy array, target of the model.
        - layers: tuple of ints, indices of the layers. # ugly
        - epochs: int, number of training epochs.
        - batch_size: int, size of the batch.
        - **compile_kwargs: keyword arguments passed to the compile method.

    Returns:
        - weights_dict: dict, weights changes for each layer at each step.
        - losses: numpy array, loss reached after each step.
        - steps: numpy array, indices of the training steps.
    """
    model.compile(
        **compile_kwrags
    )
    X_tr, X_ts = X
    y_tr, y_ts = y

    weights_dict = {index: [] for index in range(len(layers))}
    train_losses = []
    test_losses = []
    steps = []
    step = 0
    for epoch in range(epochs):

        scrambled_idx = np.random.permutation(X_tr.shape[0])
        X_tr, y_tr = X_tr[scrambled_idx], y_tr[scrambled_idx]

        for batch in tqdm(range(1, X_tr.shape[0] // batch_size), leave=True):

            start = batch*batch_size
            end = start+batch_size

            train_logs = model.train_on_batch(
                X_tr[start: end],
                y_tr[start: end],
                return_dict=True
            )
            test_logs = model.evaluate(
                X_ts,
                y_ts,
                return_dict=True,
                verbose=False,
                batch_size=512
            )
            weights = model.get_weights()

            for index, layer in enumerate(layers):

                weights_dict[index].append(
                    np.hstack(
                        [weights[layer].flatten(), weights[layer-1]]
                    )
                )

            test_losses.append(test_logs['loss'])
            train_losses.append(train_logs['loss'])
            steps.append(step)
            step += 1

    return weights_dict, train_losses, test_losses, steps
