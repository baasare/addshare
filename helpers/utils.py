import itertools
import os
import signal
import random
import pickle
import codecs
import socket
import logging
import requests
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
import tensorflow as tf
from timeit import default_timer
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def post_with_retries(url, data, max_retries=3):
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def decode_layer(b64_str):
    return pickle.loads(codecs.decode(b64_str.encode(), "base64"))


def encode_layer(layer):
    return codecs.encode(pickle.dumps(layer), "base64").decode()


def check_port(address, port):
    # Check if the port is in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((address, port))
    except socket.error:
        return True
    finally:
        sock.close()
    return False


def terminate_process_on_port(port):
    try:
        pid = int(os.popen(f"lsof -t -i:{port}").read())
        os.kill(pid, signal.SIGKILL)
        print(f"Terminated process on port {port}")
    except Exception as e:
        print(f"Error terminating process on port {port}: {e}")


def fetch_index(dataset):
    current_dir = os.path.abspath(os.getcwd())
    path = current_dir + f'/resources/dataset/{dataset}/iid_balanced.txt'
    clients = np.loadtxt(path, dtype=object)
    clients = clients.astype(np.float64)
    clients = clients.astype(np.int64)
    return clients


def fetch_dataset(dataset):
    if dataset == "cifar-10":
        return tf.keras.datasets.cifar10.load_data()
    elif dataset == "mnist":
        return tf.keras.datasets.mnist.load_data()
    elif dataset == "f-mnist":
        return tf.keras.datasets.fashion_mnist.load_data()
    else:
        current_dir = os.path.abspath(os.getcwd())
        train_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/train_32x32.mat')
        test_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/test_32x32.mat')
        x_train = np.array(train_data['X'])
        y_train = np.array(train_data['y'])

        x_test = np.array(test_data['X'])
        y_test = np.array(test_data['y'])

        return (x_train, y_train), (x_test, y_test)


def get_dataset(index, dataset, x_train, y_train, x_test, y_test):
    if dataset == "cifar-10":
        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    elif dataset == 'svhn':
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)

        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = tf.keras.utils.to_categorical((y_train - 1), 10)
        y_test = tf.keras.utils.to_categorical((y_test - 1), 10)

    else:
        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def get_lenet5(dataset):
    """
    Creates LeNet-5 model.

    :param dataset: dataset being used for model
    :return: Keras model
    """

    if dataset == "cifar-10" or dataset == 'svhn':
        pixel = 32
        rgb = 3
    else:
        pixel = 28
        rgb = 1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # LeNet-5
    model = tf.keras.models.Sequential()
    # Convolutional layer 1
    model.add(tf.keras.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        strides=1,
        activation='relu',
        name=f'conv2d_{0}',
        input_shape=(pixel, pixel, rgb)
    ))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layer 2
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name=f'conv2d_{1}'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer 1
    model.add(tf.keras.layers.Dense(units=120, activation='relu', name=f'dense_{0}'))

    # Fully connected layer 2
    model.add(tf.keras.layers.Dense(units=84, activation='relu', name=f'dense_{1}'))

    # Output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax', name=f'dense_{2}'))
    return model


def get_lenet3(dataset):
    """
    Creates LeNet-5 model.

    :param dataset: dataset being used for model
    :return: Keras model
    """

    if dataset == "cifar-10" or dataset == 'svhn':
        pixel = 32
        rgb = 3
    else:
        pixel = 28
        rgb = 1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # LeNet-5
    model = tf.keras.models.Sequential()
    # Convolutional layer 1
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation='relu',
                                     input_shape=(pixel, pixel, rgb)))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer 1
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))

    # Output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    return model


def generate_keys(save_path, name, nbits=4096):
    """
    :param save_path:
    :param nbits:
    :param name:
    :return:
    """
    os.makedirs(save_path, exist_ok=True)

    # Generate an RSA key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=nbits
    )
    public_key = private_key.public_key()

    public_pem_path = os.path.join(save_path, 'client_' + str(name) + '_public.pem')
    private_pem_path = os.path.join(save_path, 'client_' + str(name) + '_private.pem')

    try:
        # Serialize the private key to PEM format and save it to a file
        with open(private_pem_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Serialize the public key to PEM format and save it to a file
        with open(public_pem_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
    except Exception as ex:
        logging.error(ex)

    return public_pem_path, private_pem_path


def get_public_key(client_id):
    current_dir = os.path.abspath(os.getcwd())
    path = current_dir + f'/resources/keys/client_{str(client_id)}_public.pem'

    with open(path, 'rb') as f:
        return serialization.load_pem_public_key(f.read())


def get_private_key(client_id):
    current_dir = os.path.abspath(os.getcwd())
    path = current_dir + f'/resources/keys/client_{str(client_id)}_private.pem'

    with open(path, 'rb') as f:
        return serialization.load_pem_private_key(f.read(), password=None)


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs=None):
        super().__init__()
        if logs is None:
            logs = {}
        self.start_time = None
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.start_time = default_timer()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(default_timer() - self.start_time)


def convert(imgfile, labelfile, outfile, n):
    f = open(imgfile, "rb")
    o = open(outfile, "w")
    l = open(labelfile, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


def generate_additive_shares(value, n):
    arr = np.asarray(value)
    rand_arr = np.random.uniform(low=-np.abs(arr), high=np.abs(arr), size=(n - 1,) + arr.shape)
    shares = np.concatenate((rand_arr, [arr - rand_arr.sum(axis=0)]), axis=0)
    return shares


def random_weight_selection(weights, fraction=0.25):
    # Get number of weights to select
    num_select = int(weights.shape[0] * fraction)

    # Generate random indices to select
    indexes = np.random.choice(weights.shape[0], size=num_select, replace=False)

    return indexes


def magnitude_weight_selection(weights, fraction=0.25):
    # Get number of weights to select
    num_select = int(weights.shape[0] * fraction)

    # Get indices of weights sorted by magnitude
    sorted_indices = np.argsort(np.abs(weights))

    # Select the first 'num_select' indexes (smallest magnitudes)
    indexes = sorted_indices[:num_select]

    return indexes


def combine_csv_files(experiment, dataset):
    parent_dir = os.path.abspath(os.getcwd())
    folder_path = parent_dir + f'/resources/results/{experiment}/{dataset}'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Initialize an empty list to store the data frames
    data_frames = []

    # Iterate over each file
    for file in files:
        if file.startswith('client') and file.endswith('.csv'):  # Ensure that only CSV files are considered
            file_path = os.path.join(folder_path, file)

            # Read the CSV file and append it to the list
            df = pd.read_csv(file_path)
            df = df.drop(columns='round')
            data_frames.append(df)

    # Concatenate all data frames into one
    if len(data_frames) != 0:
        combined_df = pd.concat(data_frames, axis=1)
        combined_df.to_csv(f"{folder_path}/combined.csv")

    for file in files:
        if file.startswith('client') and file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

    # return combined_df


def combine_find_mean(experiment, dataset):
    parent_dir = os.path.abspath(os.getcwd())
    folder_path = parent_dir + f'/resources/results/{experiment}/{dataset}'

    combine_csv_files(experiment, dataset)

    csv_dir = os.path.join(folder_path, 'combined.csv')

    if os.path.exists(csv_dir):
        df = pd.read_csv(csv_dir)
        df['Average Accuracy'] = df.apply(lambda row: row['accuracy'].mean(), axis=1)
        df['Average Training'] = df.apply(lambda row: row['training'].mean(), axis=1)
        if 'secret_sharing' in df:
            df['Average Secret Sharing'] = df.apply(lambda row: row['secret_sharing'].mean(), axis=1)
        df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
        df.to_csv(csv_dir, index=False)


def convert_png_to_eps(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.startswith('multiline') and file.endswith('.png'):  # Ensure that only png files are considered
            # file name
            png_file = os.path.join(folder_path, file)

            # Open the PNG file
            image = Image.open(png_file)

            # Create a new EPS file with the same dimensions and mode as the PNG
            eps_image = Image.new("RGB", image.size)
            eps_image.paste(image)

            # Save the EPS file
            eps_image.save(png_file.replace("png", "eps"), format='EPS')


def generate_groups(nodes, group_size):
    # shuffle clients
    random.shuffle(nodes)

    # Create a repeating iterator over the clients list
    client_iterator = itertools.cycle(nodes)

    # Keep track of which clients have been selected
    selected_clients = set()

    # Keep selecting groups of clients until each client has been selected at least once
    selected_groups = []
    while len(selected_clients) < len(nodes):
        # Select the next group of clients
        group = list(itertools.islice(client_iterator, group_size))

        # Add the selected clients to the set of selected clients
        selected_clients.update(group)

        # Add the selected group to the list of selected groups
        selected_groups.append(group)

    # All clients have been selected at least once
    return selected_groups


def iid_balanced(client_number, train_size, dataset):
    # used to generate indexes
    rand_array = np.arange(train_size)
    np.random.shuffle(rand_array)

    clients = [[] for _ in range(client_number)]

    for i in range(client_number):
        clients[i] = rand_array[
                     int(i * train_size / client_number):int((i + 1) * train_size / client_number)]

    np.savetxt(f"resources/dataset/{dataset}/iid_balanced.txt", clients)
