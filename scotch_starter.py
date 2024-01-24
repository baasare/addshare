import sys
import time
import threading

from scotch import ScotchNode
from scotch_server import ScotchServer

from helpers.utils import fetch_index, fetch_dataset, get_dataset
from helpers.constants import ADDRESS, SERVER_PORT, NODES, CLIENT_PORT, SERVERS

if __name__ == "__main__":

    DATASET = "mnist"  # str(sys.argv[1])
    print(f"DATASET: {DATASET}")

    indexes = fetch_index(DATASET)
    (x_train, y_train), (x_test, y_test) = fetch_dataset(DATASET)

    servers, nodes = [], []
    server_ports, node_ports = [], []
    threads = []

    for i in range(1, SERVERS + 1):
        server = ScotchServer(
            address=ADDRESS,
            port=SERVER_PORT + i,
            max_nodes=NODES,
            client_type='scotch',
            dataset=DATASET,
            indexes=indexes,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

        server_ports.append(SERVER_PORT + i)
        servers.append(server)

    for i in range(1, NODES + 1):
        X_train, Y_train, X_test, Y_test = get_dataset(
            indexes[i - 1],
            DATASET,
            x_train,
            y_train,
            x_test,
            y_test
        )

        node = ScotchNode(
            address=ADDRESS,
            port=CLIENT_PORT + i,
            client_type="scotch",
            dataset=DATASET,
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )
        node_ports.append(CLIENT_PORT + i)
        nodes.append(node)

    for server in servers:
        time.sleep(2)
        t = threading.Thread(target=server.start, args=(node_ports,))
        t.start()
        threads.append(t)

    for node in nodes:
        time.sleep(2)
        t = threading.Thread(target=node.start, args=(server_ports,))
        t.start()
        threads.append(t)

    for node in nodes:
        time.sleep(2)
        node.start_training()

    for t in threads:
        t.join()
