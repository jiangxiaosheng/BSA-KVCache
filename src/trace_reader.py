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
import logging

logger = logging.getLogger(__name__)


class MobaTraceReader(ReaderProtocol):
    c_reader: bool = False

    def __init__(self, trace_dir: str, verbose: bool = False):
        self.config = sim_config()
        self.trace_dir = trace_dir
        self.verbose = verbose
        self.seq_id = -1
        self.trace_files = defaultdict(lambda: defaultdict(dict))
        self.traces = defaultdict(dict)
        self._request_generator = None

        self._validate_traces()
        self._preload_traces()

    def _validate_traces(self):
        all_files = set(os.listdir(self.trace_dir))

        # Extract unique request numbers
        trace_pattern = re.compile(r"trace(\d+)_layer(\d+)_(blocks|scores)\.npy")

        for filename in all_files:
            match = trace_pattern.match(filename)
            if match:
                seq_id = int(match.group(1))
                if self.seq_id == -1:
                    self.seq_id = seq_id
                elif self.seq_id != seq_id:
                    raise ValueError(f"Sequence Id mismatch: {self.seq_id} != {seq_id}")

                layer_idx = int(match.group(2))
                trace_type = match.group(3)
                assert trace_type in [
                    "blocks",
                    "scores",
                ], "Unknown trace type: {trace_type}. Should be blocks or scores"

                self.trace_files[layer_idx][trace_type] = os.path.join(
                    self.trace_dir, filename
                )

        if not self.trace_files:
            raise ValueError(f"No trace files found in {self.trace_dir}")

        for layer_idx in range(self.config.num_layers):
            if layer_idx not in self.trace_files:
                raise ValueError(
                    f"Layer {layer_idx} not found for sequence {self.seq_id}"
                )
            layer_data = self.trace_files[layer_idx]
            if "blocks" not in layer_data:
                raise ValueError(
                    f"Blocks not found for sequence {self.seq_id} and layer {layer_idx}"
                )
            if "scores" not in layer_data:
                raise ValueError(
                    f"Scores not found for sequence {self.seq_id} and layer {layer_idx}"
                )

    def _preload_traces(self):
        for i in range(self.config.num_layers):
            block_trace = np.load(self.trace_files[i]["blocks"])
            score_trace = np.load(self.trace_files[i]["scores"])
            self.traces[i] = (block_trace, score_trace)

    def _get_last_block_id(self) -> int:
        return (self.config.context_length - 1) // self.config.block_size

    def _generate_paged_kvcache_requests(self):
        num_reqs = 0

        # all layers must have the same number of iterations
        num_iters = self.traces[0][0].shape[2]

        for cur_iter in range(num_iters):
            for cur_layer in range(self.config.num_layers):
                block_ids = self.traces[cur_layer][0][:, :, cur_iter]

                block_ids = block_ids.flatten().tolist()
                # add the last block id as it's always selected
                block_ids.append(self._get_last_block_id())
                block_ids = set(block_ids)

                # yield all pages for the current layer
                for block_id in block_ids:
                    start_token_pos = block_id * self.config.block_size
                    end_token_pos = start_token_pos + self.config.block_size
                    num_pages = (
                        end_token_pos - start_token_pos
                    ) // self.config.page_size
                    if (end_token_pos - start_token_pos) % self.config.page_size != 0:
                        num_pages += 1

                    page_id = start_token_pos // self.config.page_size
                    for _ in range(num_pages):
                        page = PagedKVCache(self.seq_id, page_id, cur_layer)
                        if self.verbose:
                            logger.info(f"Read request id: {num_reqs}, page: {page}")
                        yield page
                        num_reqs += 1
                        page_id += 1

    def _generate_bsa_kvcache_requests(self):
        num_reqs = 0

        # all layers must have the same number of iterations
        num_iters = self.traces[0][0].shape[2]

        for cur_iter in range(num_iters):
            for cur_layer in range(self.config.num_layers):
                block_ids = self.traces[cur_layer][0][:, :, cur_iter]

                # follow GQA pattern, i.e. one single KV head can be shared across multiple Q heads
                for kvhead_id in range(
                    self.config.num_heads // self.config.kv_group_size
                ):
                    start_col = kvhead_id * self.config.kv_group_size
                    end_col = (kvhead_id + 1) * self.config.kv_group_size
                    cur_kvhead_block_ids = block_ids[:, start_col:end_col]

                    cur_kvhead_block_ids = cur_kvhead_block_ids.flatten().tolist()
                    cur_kvhead_block_ids.append(self._get_last_block_id())
                    cur_kvhead_block_ids = set(cur_kvhead_block_ids)
                    for block_id in cur_kvhead_block_ids:
                        block = BsaKVCache(self.seq_id, block_id, cur_layer, kvhead_id)
                        if self.verbose:
                            logger.info(f"Read request id: {num_reqs}, block: {block}")
                        yield block
                        num_reqs += 1

    def _generate_all_requests(self):
        if self.config.store_all_kvheads:
            yield from self._generate_paged_kvcache_requests()
        else:
            yield from self._generate_bsa_kvcache_requests()

    def read_one_req(self) -> Request:
        if self._request_generator is None:
            self._request_generator = self._generate_all_requests()

        try:
            kvcache: PagedKVCache = next(self._request_generator)
            req = Request(obj_size=PagedKVCache.bytes(), obj_id=kvcache.get_obj_id())
            return req
        except StopIteration:
            raise StopIteration("No more requests to read")

    def skip_n_req(self, n: int) -> int:
        if self._request_generator is None:
            self._request_generator = self._generate_all_requests()
        try:
            for _ in range(n):
                next(self._request_generator)
            return n
        except StopIteration:
            return n

    def get_num_of_req(self) -> int:
        raise NotImplementedError(
            "Why are you calling get_num_of_req()? I don't know that either"
        )

    def reset(self) -> None:
        self._request_generator = None

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
        raise NotImplementedError(
            "Why are you calling __len__()? I don't know that either"
        )
