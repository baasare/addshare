import numpy as np
import tensorflow as tf
from tensorflow import GradientTape


# Calculates the second derivative of the network's error function
# with respect to each weight to quantify sensitivity.

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


# Define the loss function
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)


def calculate_hessian(model, x, y):
    weights = model.trainable_weights
    gradients = tf.keras.backend.gradients(custom_loss(model.input, model.output), weights)
    flat_gradients = [tf.keras.backend.flatten(g) for g in gradients if g is not None]
    flat_gradients = tf.keras.backend.concatenate(flat_gradients)
    hessian = tf.keras.backend.hessian(custom_loss(model.input, model.output), flat_gradients)
    return tf.keras.backend.eval(hessian, feed_dict={model.input: x, tf.keras.backend.learning_phase(): 0})


def compute_OBD_saliency(model, validation_data):
    weights = model.trainable_weights
    loss = model.evaluate(tf.convert_to_tensor(validation_data[0], dtype=tf.float32),
                          tf.convert_to_tensor(validation_data[1], dtype=tf.float32),
                          verbose=0)


    hessians = []
    for i in range(len(weights)):
        with tf.GradientTape() as outer_tape:
            first_order_gradient = outer_tape.gradient(loss, weights)[i]

        with tf.GradientTape() as inner_tape:
            hessian_entry = inner_tape.gradient(first_order_gradient, weights)

        hessians.append(hessian_entry)

    get_hessians = tf.keras.backend.function(model.inputs, hessians)

    hessians_vals = np.array(get_hessians(validation_data))

    saliency = {}
    for i, weight in enumerate(weights):
        parameter_name = weight.name
        saliency[parameter_name] = np.mean(np.abs(hessians_vals[i]), axis=0)

    return saliency


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    test_model = get_model()
    test_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                       metrics=['accuracy'])

    test_model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=True)

    sensitivity_value = compute_OBD_saliency(test_model, validation_data=(x_test, y_test))

    print(sensitivity_value)
