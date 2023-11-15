import json

import numpy as np
import uvicorn
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from timeit import default_timer as timer

from helpers.utils import check_port, terminate_process_on_port, decode_layer, TimingCallback, encode_layer, \
    post_with_retries, generate_additive_shares
from helpers.constants import EPOCHS, ADDRESS, SERVER_PORT, MESSAGE_START_TRAINING, MESSAGE_END_SESSION, \
    MESSAGE_FL_UPDATE, CLIENT_PORT, MESSAGE_MODEL_SHARE, MESSAGE_SHARING_COMPLETE, MESSAGE_ALL_NODES, \
    MESSAGE_START_ASSEMBLY


class Node:

    def __init__(self, address, port, client_type, dataset, x_train, y_train, x_test, y_test):
        self.app = FastAPI()
        self.port = port
        self.address = address

        self.dataset = dataset
        self.client_type = client_type

        self.model = None
        self.epochs = EPOCHS

        self.own_shares = dict()
        self.other_shares = dict()

        self.fl_nodes = list()
        self.share_count = 0

        self.start_time = None
        self.end_time = None

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

        self.X_train, self.y_train, self.X_test, self.y_test = x_train, y_train, x_test, y_test

        @self.app.post("/message")
        def message(data: dict):
            print(f"PORT {self.port} RECEIVED: {data['message']}")
            if data["message"] == MESSAGE_START_TRAINING:
                self.start_training(data)

            elif data["message"] == MESSAGE_END_SESSION:
                self.end_session(data)

            elif data["message"] == MESSAGE_ALL_NODES:
                self.start_secret_sharing(data["nodes"])

            elif data["message"] == MESSAGE_MODEL_SHARE:
                self.accept_shares(data['model_share'])

            elif data["message"] == MESSAGE_START_ASSEMBLY:
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

    def start_training(self, global_model):
        self.round += 1
        self.model = tf.keras.models.model_from_json(global_model["model_architecture"])
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        self.send_updates()

    def start_secret_sharing(self, data):
        self.fl_nodes = data
        self.start_time = timer()

        shares = int(len(self.fl_nodes) + 1)

        for layer in self.model.layers:
            if layer.trainable_weights:
                weight_shares = list(generate_additive_shares(layer.weights[0], shares))
                bias_shares = list(generate_additive_shares(layer.weights[1], shares))

                self.own_shares[layer.name] = [[], []]
                self.own_shares[layer.name][0].append(weight_shares.pop())
                self.own_shares[layer.name][1].append(bias_shares.pop())

                self.other_shares[layer.name] = [None, None]
                self.other_shares[layer.name][0] = weight_shares
                self.other_shares[layer.name][1] = bias_shares

        self.share_count += 1

        self.start_exchanging_shares()

    def start_exchanging_shares(self):
        for node in self.fl_nodes:
            layer_weights = dict()

            for layer in self.other_shares.keys():
                weight_bias = [None, None]
                weight_bias[0] = self.other_shares[layer][0].pop()
                weight_bias[1] = self.other_shares[layer][1].pop()

                layer_weights[layer] = encode_layer(weight_bias)

            data = {
                "message": MESSAGE_MODEL_SHARE,
                "model_share": layer_weights,
            }

            client_node = self.connect_get_node(node['address'], int(node['port']))
            self.send_to_node(client_node, json.dumps(data))

    def accept_shares(self, data):

        for layer in data.keys():
            weight_bias = decode_layer(data[layer])
            self.own_shares[layer][0].append(weight_bias[0])
            self.own_shares[layer][1].append(weight_bias[1])

        self.share_count += 1

        if self.share_count == int(len(self.fl_nodes) + 1):
            self.share_count = 0
            payload = {
                "message": MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(self.server, json.dumps(payload))

    def reassemble_shares(self):
        layer_weights = dict()

        for layer in self.own_shares.keys():
            temp_weight_bias = [None, None]
            temp_weight_bias[0] = np.sum((self.own_shares[layer][0]), axis=0)
            temp_weight_bias[1] = np.sum((self.own_shares[layer][1]), axis=0)
            layer_weights[layer] = encode_layer(temp_weight_bias)

        self.end_time = timer() - self.start_time

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
            'secret_sharing': self.end_time
        })

        payload = {
            "message": MESSAGE_FL_UPDATE,
            "model_weights": layer_weights,
        }
        self.send_to_node(self.server, json.dumps(payload))

    def send_updates(self):
        model_weights = dict()
        for layer in self.model.layers:
            if layer.trainable_weights:
                model_weights[layer.name] = encode_layer(layer.get_weights())

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
        })

        data = {
            "port": self.port,
            "message": MESSAGE_FL_UPDATE,
            "model_weights": model_weights,
        }

        print(f"PORT {self.port} Sending update to server")

        self.send_to_node(ADDRESS, SERVER_PORT, data)

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()

    def disconnect(self):
        pd.DataFrame(self.record).to_csv(
            f"resources/results/{self.client_type}/{self.dataset}/client_{self.port - CLIENT_PORT}.csv",
            index=False,
            header=True
        )
