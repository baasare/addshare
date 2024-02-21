import json
import os
import time
import uvicorn
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers import constants
from helpers.utils import get_lenet5_regression, combine_find_mean, regularization_weight_selection, post_with_retries
from helpers.utils import random_weight_selection, magnitude_weight_selection, obd_weight_selection, get_area_x_dataset
from helpers.utils import check_port, terminate_process_on_port, decode_layer, TimingCallback, encode_layer, \
    generate_additive_shares


class ServerAddsharePlus:
    def __init__(self, server_id, address, port, max_nodes, client_type, pruning_type, dataset, x_train,
                 y_train, x_test, y_test):
        self.id = server_id
        self.app = FastAPI()
        self.port = port
        self.address = address
        self.connected_nodes = 0
        self.max_nodes = max_nodes
        self.nodes = list()

        self.start_time = None
        self.end_time = None
        self.pending_nodes = set()
        self.average_weights = dict()
        _, _, self.X_test, self.y_test = x_train, y_train, x_test, y_test

        self.global_model = get_lenet5_regression()
        self.max_rounds = constants.ROUNDS
        self.round = 0
        self.training_completed_count = 0
        self.client_type = client_type
        self.pruning_type = pruning_type
        self.dataset = dataset

        self.record = list()
        self.loss, self.mse, self.mae, self.mape = 0, 0, 0, 0
        self.threshold = 0

        @self.app.post("/message")
        def message(data: dict):
            print(f"SERVER RECEIVED: {data['message']} from PORT: {data['port']}")

            if data["message"] == constants.MESSAGE_FL_UPDATE:
                self.fl_update(data["port"], data["model_weights"])

            elif data["message"] == constants.MESSAGE_TRAINING_COMPLETED:
                self.start_secret_sharing()

            elif data["message"] == constants.MESSAGE_SHARING_COMPLETE:
                self.start_assembly(data["port"])

            return {"status": "ok"}

    def start(self):
        if check_port(self.address, self.port):
            terminate_process_on_port(self.port)

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    def send_to_node(self, data, port=None):
        if port is None:
            for port in self.nodes:
                post_with_retries(
                    data=data,
                    url=f"http://{constants.ADDRESS}:{port}/message",
                    max_retries=3
                )
                time.sleep(2)
        else:
            post_with_retries(
                data=data,
                url=f"http://{constants.ADDRESS}:{port}/message",
                max_retries=3
            )

    def start_round(self, nodes=None):
        if nodes:
            self.nodes = nodes

        print(f'Starting round ({self.round + 1})')

        indexes = {}
        self.start_time = timer()
        self.pending_nodes = self.nodes.copy()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                if self.pruning_type == constants.RANDOM:
                    kernel_indices = random_weight_selection(layer.get_weights()[0], constants.THRESHOLD)
                    bias_indices = random_weight_selection(layer.get_weights()[1], constants.THRESHOLD)
                    indexes[layer.name] = [kernel_indices, bias_indices]
                elif self.pruning_type == constants.MAGNITUDE:
                    kernel_indices = magnitude_weight_selection(layer.get_weights()[0], constants.THRESHOLD)
                    bias_indices = magnitude_weight_selection(layer.get_weights()[1], constants.THRESHOLD)
                    indexes[layer.name] = [kernel_indices, bias_indices]
                elif self.pruning_type == constants.OBD:
                    kernel_indices = obd_weight_selection(
                        self.global_model,
                        self.X_test,
                        self.y_test,
                        layer.trainable_weights[0],
                        constants.THRESHOLD
                    )
                    bias_indices = obd_weight_selection(
                        self.global_model,
                        self.X_test,
                        self.y_test,
                        layer.trainable_weights[1],
                        constants.THRESHOLD
                    )
                    indexes[layer.name] = [kernel_indices, bias_indices]
                else:
                    kernel_indices = regularization_weight_selection(
                        self.global_model,
                        self.X_test,
                        self.y_test,
                        self.pruning_type,
                        layer.trainable_weights[0],
                        constants.THRESHOLD
                    )
                    bias_indices = regularization_weight_selection(
                        self.global_model,
                        self.X_test,
                        self.y_test,
                        self.pruning_type,
                        layer.trainable_weights[1],
                        constants.THRESHOLD
                    )
                    indexes[layer.name] = [kernel_indices, bias_indices]
                self.average_weights[layer.name] = [[], []]

        data = {
            "port": "SERVER",
            "nodes": self.nodes,
            "message": constants.MESSAGE_START_TRAINING,
            "indexes": json.dumps(indexes),
            "model_architecture": self.global_model.to_json(),
            "model_weights": encode_layer(self.global_model.get_weights()),
        }
        self.send_to_node(data)

    def fl_update(self, node, data):

        for layer in data.keys():
            temp_weight = decode_layer(data[layer])

            if len(self.average_weights[layer][0]) == 0 and len(self.average_weights[layer][1]) == 0:
                self.average_weights[layer][0] = temp_weight[0] / len(self.nodes)
                self.average_weights[layer][1] = temp_weight[1] / len(self.nodes)
            else:
                self.average_weights[layer][0] += temp_weight[0] / len(self.nodes)
                self.average_weights[layer][1] += temp_weight[1] / len(self.nodes)

        self.pending_nodes.remove(node)
        if not self.pending_nodes:
            self.apply_updates()

    def apply_updates(self):
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])
        self.evaluate()

    def evaluate(self):
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
        self.global_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse', 'mae', 'mape', ])
        self.loss, self.mse, self.mae, self.mape = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
        self.end_time = timer() - self.start_time
        print('Loss: ', self.loss)
        print('Mean Squared Error: ', self.mse)
        print('Mean Absolute Error: ', self.mae)
        print('Mean Absolute Percentage Error: ', self.mape)
        print(f'Round ({self.round + 1}) Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'loss': self.loss,
            'mse': self.mse,
            'mae': self.mae,
            'mape': self.mape,
            'fl': self.end_time,
        })
        self.end_round()

    def end_round(self):
        print("ROUND ENDED")
        self.round += 1
        if self.round < self.max_rounds:
            self.start_round()
        else:
            self.end_session()

    def end_session(self):
        print("SESSION ENDED")
        data = {
            "port": "SERVER",
            "message": constants.MESSAGE_END_SESSION,
            "model_weights": encode_layer(self.global_model.get_weights()),
        }

        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}_{self.pruning_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = 'server.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)

        self.send_to_node(data)
        combine_find_mean(f"{self.client_type}_{self.pruning_type}", f"{self.dataset}")
        terminate_process_on_port(self.port)

    def start_assembly(self, port):
        data = {
            "port": "SERVER",
            "message": constants.MESSAGE_START_ASSEMBLY,
        }
        self.send_to_node(data, port=port)

    def start_secret_sharing(self):
        self.training_completed_count += 1
        if self.training_completed_count == len(self.nodes):
            self.training_completed_count = 0
            data = {
                "port": "SERVER",
                "message": constants.MESSAGE_START_SECRET_SHARING,
            }
            self.send_to_node(data)


class AddSharePlusNode:

    def __init__(self, address, port, client_type, pruning_type, dataset, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address

        self.dataset = dataset
        self.client_type = client_type
        self.pruning_type = pruning_type

        self.model = None
        self.epochs = constants.EPOCHS

        self.indexes = list()
        self.own_shares = dict()
        self.other_shares = dict()

        self.fl_nodes = list()
        self.share_count = 0

        self.start_time = None
        self.secret_sharing_time = 0.0

        self.record = list()

        self.round = 0
        self.loss, self.mse, self.mae, self.mape = 0, 0, 0, 0
        self.current_training_time = 0

        self.X_train, self.y_train, self.X_test, self.y_test = x_train, y_train, x_test, y_test

        @self.app.post("/message")
        def message(data: dict):
            print(f"PORT {self.port} RECEIVED: {data['message']} from {data['port']}")
            if data["message"] == constants.MESSAGE_START_TRAINING:
                self.start_training(data)

            elif data["message"] == constants.MESSAGE_START_SECRET_SHARING:
                self.start_secret_sharing()

            elif data["message"] == constants.MESSAGE_END_SESSION:
                self.end_session(data)

            elif data["message"] == constants.MESSAGE_MODEL_SHARE:
                self.accept_shares(data['model_share'])

            elif data["message"] == constants.MESSAGE_START_ASSEMBLY:
                self.reassemble_shares()

            return {"status": "ok"}

    def start(self):
        if check_port(self.address, self.port):
            terminate_process_on_port(self.port)

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    @staticmethod
    def send_to_node(address, port, data):
        post_with_retries(
            data=data,
            url=f"http://{address}:{port}/message",
            max_retries=3
        )

    def start_training(self, data):
        self.fl_nodes = [port for port in data["nodes"] if port != self.port]
        self.indexes = json.loads(data["indexes"])
        self.round += 1
        self.model = tf.keras.models.model_from_json(data["model_architecture"])
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse', 'mae', 'mape', ])
        self.model.set_weights(decode_layer(data["model_weights"]))

        cb = TimingCallback()
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False)
        self.loss, self.mse, self.mae, self.mape = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        print('Loss: ', self.loss)
        print('Mean Squared Error: ', self.mse)
        print('Mean Absolute Error: ', self.mae)
        print('Mean Absolute Percentage Error: ', self.mape)
        self.current_training_time = sum(cb.logs)

        for layer in self.model.layers:
            if layer.trainable_weights:
                self.own_shares[layer.name] = [[], []]
                self.other_shares[layer.name] = [None, None]

        data = {
            "port": self.port,
            "message": constants.MESSAGE_TRAINING_COMPLETED
        }

        self.send_to_node(address=constants.ADDRESS, port=constants.SERVER_PORT, data=data)

    def start_secret_sharing(self):
        self.start_time = timer()
        shares = int(len(self.fl_nodes) + 1)

        for layer in self.model.layers:
            if layer.trainable_weights:
                # get random indexes
                selected_kernel_index = tuple(np.array(li) for li in self.indexes[layer.name][0])
                selected_bias_index = tuple(np.array(li) for li in self.indexes[layer.name][1])

                # get selected columns
                selected_kernels = layer.get_weights()[0][selected_kernel_index]
                selected_bias = layer.get_weights()[1][selected_bias_index]

                # generate additive shares of selected weights
                weight_shares = list(generate_additive_shares(selected_kernels, NODES))
                bias_shares = list(generate_additive_shares(selected_bias, NODES))

                self.own_shares[layer.name] = [[], []]
                self.own_shares[layer.name][0].append(weight_shares.pop())
                self.own_shares[layer.name][1].append(bias_shares.pop())

                self.other_shares[layer.name] = [None, None]
                self.other_shares[layer.name][0] = weight_shares
                self.other_shares[layer.name][1] = bias_shares

        self.share_count += 1

        self.secret_sharing_time = timer() - self.start_time
        self.start_exchanging_shares()

    def start_exchanging_shares(self):
        self.start_time = timer()
        for client in self.fl_nodes:
            layer_weights = dict()

            for layer in self.other_shares.keys():
                weight_bias = [None, None]
                weight_bias[0] = self.other_shares[layer][0].pop()
                weight_bias[1] = self.other_shares[layer][1].pop()

                layer_weights[layer] = encode_layer(weight_bias)

            data = {
                "port": self.port,
                "message": constants.MESSAGE_MODEL_SHARE,
                "model_share": layer_weights,
            }

            self.send_to_node(data=data, address=constants.ADDRESS, port=client)

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        if self.share_count == int(len(self.fl_nodes) + 1):
            self.share_count = 0
            data = {
                "port": self.port,
                "message": constants.MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(data=data, address=constants.ADDRESS, port=constants.SERVER_PORT)

    def accept_shares(self, data):

        self.start_time = timer()
        for layer in data.keys():
            weight_bias = decode_layer(data[layer])
            self.own_shares[layer][0].append(weight_bias[0])
            self.own_shares[layer][1].append(weight_bias[1])

        self.share_count += 1

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        if self.share_count == int(len(self.fl_nodes) + 1):
            self.share_count = 0
            data = {
                "port": self.port,
                "message": constants.MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(data=data, address=constants.ADDRESS, port=constants.SERVER_PORT)

    def reassemble_shares(self):
        self.start_time = timer()
        layer_weights = dict()

        for layer in self.own_shares.keys():
            kernel = np.sum((self.own_shares[layer][0]), axis=0)
            bias = np.sum((self.own_shares[layer][1]), axis=0)

            selected_kernel_index = tuple(np.array(li) for li in self.indexes[layer][0])
            selected_bias_index = tuple(np.array(li) for li in self.indexes[layer][1])

            # Get original model weights
            temp_weight_bias = [
                self.model.get_layer(layer).get_weights()[0],
                self.model.get_layer(layer).get_weights()[1]
            ]

            # Replace original selected weights with assembled additive shares
            temp_weight_bias[0][selected_kernel_index] = kernel
            temp_weight_bias[1][selected_bias_index] = bias
            layer_weights[layer] = encode_layer(temp_weight_bias)

        self.secret_sharing_time = self.secret_sharing_time + (timer() - self.start_time)

        self.record.append({
            'round': self.round,
            'loss': self.loss,
            'mse': self.mse,
            'mae': self.mae,
            'mape': self.mape,
            'training': self.current_training_time,
            'secret_sharing': self.secret_sharing_time
        })

        data = {
            "port": self.port,
            "message": constants.MESSAGE_FL_UPDATE,
            "model_weights": layer_weights,
        }
        self.send_to_node(address=constants.ADDRESS, port=constants.SERVER_PORT, data=data)

    def send_updates(self):
        model_weights = dict()
        for layer in self.model.layers:
            if layer.trainable_weights:
                model_weights[layer.name] = encode_layer(layer.get_weights())

        self.record.append({
            'round': self.round,
            'loss': self.loss,
            'mse': self.mse,
            'mae': self.mae,
            'mape': self.mape,
            'training': self.current_training_time,
        })

        data = {
            "port": self.port,
            "message": constants.MESSAGE_FL_UPDATE,
            "model_weights": model_weights,
        }

        self.send_to_node(constants.ADDRESS, constants.SERVER_PORT, data)

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()

    def disconnect(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_folder = current_dir + f"/resources/results/{self.client_type}_{self.pruning_type}/{self.dataset}"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f'client_{self.port - constants.CLIENT_PORT}.csv'
        csv_path = os.path.join(output_folder, csv_filename)
        pd.DataFrame(self.record).to_csv(csv_path, index=False, header=True)


if __name__ == "__main__":
    NODES = 5
    DATASET = "area_x"
    SELECTION_TYPE = "magnitude"
    x_train, y_train, x_test, y_test = get_area_x_dataset(1, month="june")

    nodes = []
    ports = []
    threads = []

    server = ServerAddsharePlus(
        server_id=constants.SERVER_ID,
        address=constants.ADDRESS,
        port=constants.SERVER_PORT,
        max_nodes=NODES,
        client_type='addshare_plus',
        pruning_type=SELECTION_TYPE,
        dataset=DATASET,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
    server_thread = threading.Thread(target=server.start)

    for i in constants.FIELDS:
        X_train, Y_train, X_test, Y_test = get_area_x_dataset(i, month="june")

        node = AddSharePlusNode(
            address=constants.ADDRESS,
            port=constants.CLIENT_PORT + int(i),
            client_type="addshare_plus",
            pruning_type=SELECTION_TYPE,
            dataset=DATASET,
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )
        ports.append(constants.CLIENT_PORT + int(i))
        nodes.append(node)

    server_thread.start()
    for node in nodes:
        time.sleep(2)
        t = threading.Thread(target=node.start)
        t.start()
        threads.append(t)

    server.start_round(ports)

    server_thread.join()
    for t in threads:
        t.join()
