import numpy as np
from pathlib import Path
import os
from trace_reader import MobaTraceReader
from config import sim_config

if __name__ == "__main__":
    # data = np.load("traces/moba-512-3/beyond-4096/trace0004_layer27_blocks.npy")
    # print(data.shape)
    config = sim_config()
    config.from_yaml("config.yaml")
    print(config)
    reader = MobaTraceReader()
    # print(reader.read_one_req())
    # print(f'Number of requests: {reader.get_num_of_req()}')
    for req in reader:
        print(f'Request(obj_size={req.obj_size}, obj_id={req.obj_id})')
        pass