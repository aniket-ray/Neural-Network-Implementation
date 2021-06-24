import json
import NeuralNet
import numpy as np
from typing import Optional
from collections import deque


class JsonEncoderDecoder:
    class NNDataEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, deque):
                return {"__deque__": list(o)}
            elif isinstance(o, NeuralNet.NNData):
                return {"__NNData__": o.__dict__}
            elif isinstance(o, np.ndarray):
                return {"__NDarray__": o.tolist()}
            else:
                return super().default(o)

    @staticmethod
    def NNDataDecoder(o):
        if "__deque__" in o:
            return deque[o["__deque__"]]
        elif "__NNData__" in o:
            decoded_obj = o["__NNData__"]
            train_factor = decoded_obj["_train_factor"]
            train_indices = decoded_obj["_train_indices"]
            test_indices = decoded_obj["_test_indices"]
            train_pool_deque = decoded_obj["_train_pool"]
            test_pool_deque = decoded_obj["_test_pool"]
            features_ndarray = decoded_obj["_features"]["__NDarray__"]
            labels_ndarray = decoded_obj["_labels"]["__NDarray__"]

            NNdata_json = NeuralNet.NNData(features_ndarray,
                                           labels_ndarray,
                                           train_factor)
            NNdata_json._train_indices = train_indices
            NNdata_json._test_indices = test_indices
            NNdata_json._train_pool = train_pool_deque
            NNdata_json._test_pool = test_pool_deque
            return NNdata_json
        else:
            return o

    @staticmethod
    def decode(file_name: str):
        with open(file_name, "r") as file:
            data = json.load(file, object_hook=JsonEncoderDecoder.NNDataDecoder)
            return data

    @staticmethod
    def encode(file_name: str, obj=Optional[NeuralNet.NNData]):
        if obj is not None:
            with open(file_name, "w") as file:
                json.dump(obj, file, cls=JsonEncoderDecoder.NNDataEncoder)


def run_XOR():
    network = NeuralNet.FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    XOR_x = [[1, 1], [1, 0], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0], [0, 0],
             [0, 1], [0, 1], [1, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 1], [0, 0], [1, 0], [0, 1], [0, 1], [1, 1],
             [1, 0], [0, 1], [0, 1], [1, 1], [1, 1], [0, 1], [1, 1], [0, 0], [1, 1], [0, 0], [0, 1], [1, 1], [1, 1],
             [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 1], [1, 1],
             [1, 1], [1, 0], [0, 1], [1, 1], [1, 1], [0, 0], [1, 0], [0, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 1],
             [0, 0], [1, 1], [0, 1], [1, 1], [0, 1], [1, 1], [0, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 0], [0, 1],
             [1, 0], [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [1, 0], [1, 1], [1, 1],
             [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [1, 1], [1, 0], [1, 0], [1, 0], [0, 0], [1, 0],
             [1, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 0], [1, 1], [1, 1], [0, 1], [0, 0], [1, 0], [1, 0], [0, 1],
             [0, 1], [0, 1], [1, 1], [1, 1], [0, 0], [1, 0], [0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [0, 1], [1, 1],
             [0, 0], [1, 1], [1, 0], [1, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 0], [0, 0],
             [0, 1], [1, 1], [0, 1], [1, 1], [0, 0], [1, 1], [1, 0], [1, 0], [0, 1], [1, 1], [1, 1], [0, 0], [0, 0],
             [0, 1], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0], [0, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 0], [0, 0],
             [1, 1], [0, 1], [1, 1], [1, 1], [0, 1], [1, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 0], [0, 0],
             [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [0, 0], [1, 0], [0, 0], [1, 1], [0, 0], [0, 1], [1, 1], [0, 1],
             [0, 0], [0, 1], [1, 1], [0, 1], [1, 1], [1, 1]]
    XOR_y = [[0], [1], [1], [0], [1], [0], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [1],
             [0], [1], [1], [1], [0], [1], [1], [1], [0], [0], [1], [0], [0], [0], [0], [1], [0], [0], [1], [1], [1],
             [1], [1], [0], [0], [1], [0], [1], [0], [0], [0], [0], [1], [1], [0], [0], [0], [1], [0], [1], [0], [0],
             [1], [0], [0], [0], [1], [0], [1], [0], [1], [0], [1], [1], [0], [1], [1], [1], [1], [1], [0], [1], [1],
             [1], [1], [1], [0], [1], [0], [0], [1], [0], [1], [1], [0], [0], [1], [0], [1], [1], [1], [0], [1], [0],
             [1], [1], [1], [1], [0], [0], [0], [1], [0], [1], [1], [1], [1], [1], [0], [0], [0], [1], [0], [0], [0],
             [0], [0], [1], [0], [0], [0], [1], [0], [1], [1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0],
             [0], [0], [1], [1], [1], [0], [0], [0], [0], [1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [1],
             [0], [0], [1], [0], [0], [1], [0], [0], [1], [1], [0], [1], [0], [0], [1], [0], [1], [1], [1], [0], [1],
             [0], [0], [0], [1], [0], [1], [0], [1], [0], [1], [0], [0]]
    xor_data = NeuralNet.NNData(XOR_x, XOR_y, .1)
    JsonEncoderDecoder.encode("XOR.json", xor_data)


if __name__ == "__main__":
    # Load XOR and Encode to "xor.json"
    run_XOR()

    # Decode XOR data from "xor.json"
    xor_data_decoded = JsonEncoderDecoder.decode("xor.json")
    network = NeuralNet.FFBPNetwork(2, 1)
    network.add_hidden_layer(3)

    # Train and Test the data
    network.train(xor_data_decoded, 1001)
    network.test(xor_data_decoded)

    # Decode Sine data
    sin_data_decoded = JsonEncoderDecoder.decode("SINE.json")
    network = NeuralNet.FFBPNetwork(1, 1)
    network.add_hidden_layer(3)

    # Train and Test the data
    network.train(sin_data_decoded, 1001)
    network.test(sin_data_decoded)
