import os
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from helpers.constants import NODES
from helpers.utils import generate_keys


def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


@tf.function
def get_logits(model, x):
    return model(x)


@tf.function
def loss_fn(logits, y):
    return K.categorical_crossentropy(y, logits)


def compute_hessian_diagonal(model, weights, x, y):
    with tf.GradientTape(persistent=True) as tape:
        logits = model(x)
        loss = loss_fn(logits, y)

        grads = tape.gradient(loss, weights)
        hessians = []

        for w, grad in zip(weights, grads):
            hess = tape.gradient(grad, w)
            hessians.append(hess.numpy())

        return hessians


def obd_pruning(model, weights, x, y, saliency_threshold):
    hessian_diagonal = 0  # compute_hessian_diagonal(model, x_train, y_train)

    # Identify weights to prune based on the saliency threshold
    indices = np.where(np.array(hessian_diagonal) > saliency_threshold)[0]

    return indices


def random_weight_selection(weights, fraction=0.25):
    percentage = max(0, min(100, fraction))

    num_elements = int(np.ceil(percentage * weights.size))

    flattened_array = weights.flatten()

    random_elements = np.random.choice(flattened_array, num_elements, replace=False)

    indices_in_flattened = np.where(np.isin(flattened_array, random_elements))[0]

    indices = np.unravel_index(indices_in_flattened, weights.shape)

    return indices


if __name__ == "__main__":
    keys_path = os.path.join(os.path.dirname(os.getcwd()), 'resources', 'keys', 'elliptical')
    # generate encryption keys for all clients
    for i in range(50):
        generate_keys(keys_path, 1 + i, 'elliptical')

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #
    # x_train = x_train.astype('float32') / 255.0
    # x_test = x_test.astype('float32') / 255.0
    #
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
    #
    # # Load trained model
    # model = get_model()
    # model.compile(
    #     optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy']
    # )
    #
    # model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=True)
    #
    # # weights = model.trainable_weights
    #
    # weights = model.layers[2].get_weights()[0]
    #
    # indexes = random_weight_selection(weights, 0.05)
    #
    # print(indexes)
    #
    # # sensitivity = compute_hessian_diagonal(model, weights, x_train[:1000], y_train[:1000])
    # # print(sensitivity)
