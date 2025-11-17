import re
import os
from libcachesim.protocols import ReaderProtocol
from libcachesim import Request
import numpy as np
from config import sim_config
from kvcache import PagedKVCache, BsaKVCache
from typing import Iterator
from collections import defaultdict
from tqdm import tqdm


class MobaTraceReader(ReaderProtocol):
    c_reader: bool = False

    def __init__(self):
        self.config = sim_config()
        self.cur_batch = -1
        self.max_output_len = -1
        self.cur_iter = 0
        self.num_reqs = 0
        self.cur_req = 0
        self.cur_layer = 0
        self.cur_head = 0
        self.reached_end = False
        self.trace_files = defaultdict(lambda: defaultdict(dict))
        self.traces = defaultdict(dict)
        self.kvcache_to_read = []

        self._validate_traces()
        self._preload_traces()

    def _validate_traces(self):
        all_files = set(os.listdir(self.config.trace_dir))

        # Extract unique request numbers
        trace_pattern = re.compile(r"trace(\d+)_layer(\d+)_(blocks|scores)\.npy")

        for filename in all_files:
            match = trace_pattern.match(filename)
            if match:
                request_num = int(match.group(1))
                layer_idx = int(match.group(2))
                trace_type = match.group(3)
                assert trace_type in [
                    "blocks",
                    "scores",
                ], "Unknown trace type: {trace_type}. Should be blocks or scores"

                self.trace_files[request_num][layer_idx][trace_type] = os.path.join(
                    self.config.trace_dir, filename
                )

        if not self.trace_files:
            raise ValueError(f"No trace files found in {self.config.trace_dir}")

        for request_num in sorted(self.trace_files.keys()):
            for layer_idx in range(self.config.num_layers):
                if layer_idx not in self.trace_files[request_num]:
                    raise ValueError(
                        f"Layer {layer_idx} not found for request {request_num}"
                    )
                layer_data = self.trace_files[request_num][layer_idx]
                if "blocks" not in layer_data:
                    raise ValueError(
                        f"Blocks not found for request {request_num} and layer {layer_idx}"
                    )
                if "scores" not in layer_data:
                    raise ValueError(
                        f"Scores not found for request {request_num} and layer {layer_idx}"
                    )

    def _preload_traces(self):
        num_traces = len(self.trace_files)
        for trace_id in tqdm(range(num_traces), desc="Loading traces"):
            for i in range(self.config.num_layers):
                block_trace = np.load(self.trace_files[trace_id][i]["blocks"])
                score_trace = np.load(self.trace_files[trace_id][i]["scores"])
                self.traces[trace_id][i] = (block_trace, score_trace)
            self.num_reqs += (
                self.traces[trace_id][0][0].shape[2]
                * self.config.top_k
                * self.config.block_size
                // self.config.page_size
                * self.config.num_heads
                * self.config.num_layers
            )

    def read_one_req(self) -> Request:
        if self.reached_end:
            raise StopIteration("No more requests to read")

        # Note: Here we assume we follow a sequential order of accessing each head and paged/bsa kv caches
        # within that head, despite the fact that they are read in parallel from CPU DRAM.

        if self.config.store_all_kvheads:
            # case for standard paged attention
            if len(self.kvcache_to_read) == 0:
                block_ids = self.traces[self.cur_req][self.cur_layer][0][
                    :, self.cur_head, self.cur_iter
                ]

                # print(f"shape of block_ids: {block_ids.shape}")
                # print(block_ids)
                block_ids = block_ids.tolist()
                last_block_id = (
                    self.config.context_length - 1
                ) // self.config.block_size
                block_ids.append(last_block_id)
                # print(block_ids)

                self.kvcache_to_read = []
                for block_id in block_ids:
                    start_token_pos = block_id * self.config.block_size
                    end_token_pos = start_token_pos + self.config.block_size
                    num_pages = (
                        end_token_pos - start_token_pos
                    ) // self.config.page_size
                    if (end_token_pos - start_token_pos) % self.config.page_size != 0:
                        num_pages += 1

                    # print("start token pos: ", start_token_pos)
                    page_id = start_token_pos // self.config.page_size
                    # print("page id: ", page_id)
                    for _ in range(num_pages):
                        page = PagedKVCache(self.cur_req, page_id, self.cur_layer)
                        self.kvcache_to_read.append(page)
                        page_id += 1
                self.cur_head += 1
                if self.cur_head == self.config.num_heads:
                    self.cur_head = 0
                    self.cur_layer += 1
                    if self.cur_layer == self.config.num_layers:
                        self.cur_layer = 0
                        self.cur_req += 1
                        if self.cur_req == len(self.trace_files):
                            self.reached_end = True

            kvcache: PagedKVCache = self.kvcache_to_read.pop(0)
            print(f'Reading paged kvcache: {kvcache}')
            req = Request(obj_size=PagedKVCache.bytes(), obj_id=kvcache.get_obj_id())
            return req

        else:
            # case for block sparse attention
            pass

    def skip_n_req(self, n: int) -> int:
        raise NotImplementedError("Why are you calling skip_n_req()?")

    def get_num_of_req(self) -> int:
        return self.num_reqs

    def reset(self) -> None:
        raise NotImplementedError("Why are you calling reset()?")

    def close(self) -> None:
        pass

    def clone(self) -> "ReaderProtocol":
        raise NotImplementedError("Why are you calling clone()?")

    def get_working_set_size(self) -> tuple[int, int]:
        raise NotImplementedError("Why are you calling get_working_set_size()?")

    def __iter__(self) -> Iterator[Request]:
        return self

    def __next__(self) -> Request:
        return self.read_one_req()

    def __len__(self) -> int:
        return self.num_reqs
