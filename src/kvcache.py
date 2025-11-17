from config import sim_config
from math import ceil, log2


# In this simulation we do not consider cross-request KV cache reuse
# so each KV cache makes sense only for the request it belongs to.


# This is the vanilla paged attention KV cache
# It has shape [page_size, 2, num_kvheads, hidden_size] and is per layer
# Note that it contains all KV heads in that layer
class PagedKVCache:
    def __init__(self, req_id: int, page_id: int, layer_id: int):
        self.req_id = req_id
        self.page_id = page_id
        self.layer_id = layer_id

    @classmethod
    def bytes(self) -> int:
        config = sim_config()
        return (
            config.page_size
            * 2
            * config.num_kvheads
            * config.hidden_size
            * config.dtype_bytes
        )

    def __repr__(self):
        return f"PagedKVCache(req_id={self.req_id}, page_id={self.page_id}, layer_id={self.layer_id}, obj_id={self.get_obj_id()})"

    def get_obj_id(self) -> int:
        config = sim_config()
        bits_page = (
            ceil(log2(config.context_length // config.page_size))
            if config.context_length // config.page_size > 1
            else 1
        )
        bits_layer = ceil(log2(config.num_layers)) if config.num_layers > 1 else 1
        bits_seq = ceil(log2(config.batch_size)) if config.batch_size > 1 else 1

        shift_page = 0
        shift_layer = shift_page + bits_page
        shift_seq = shift_layer + bits_layer

        total_bits = shift_seq + bits_seq
        assert (
            total_bits <= 64
        ), f"Object Id overflow! Need {total_bits} bits but only 64 available"

        return (
            (self.req_id << shift_seq)
            | (self.layer_id << shift_layer)
            | (self.page_id << shift_page)
        )


class BsaKVCache:
    def __init__(self, req_id: int, layer_id: int):
        config = sim_config()
        self.block_size = config.block_size
        self.req_id = req_id
        self.layer_id = layer_id

    def __str__(self):
        return f"BsaKVCache(req_id={self.req_id}, layer_id={self.layer_id})"
