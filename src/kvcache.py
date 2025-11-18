from config import sim_config
from math import ceil, log2


# In this simulation we do not consider cross-request KV cache reuse
# so each KV cache makes sense only for the request it belongs to.


# This is the vanilla paged attention KV cache
# It has shape [page_size, 2, num_kvheads, hidden_size] and is per layer
# Note that it contains all KV heads in that layer
class PagedKVCache:
    def __init__(self, seq_id: int, page_id: int, layer_id: int):
        self.seq_id = seq_id
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
        return f"PagedKVCache(seq_id={self.seq_id}, page_id={self.page_id}, layer_id={self.layer_id}, obj_id={self.get_obj_id()})"

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
            (self.seq_id << shift_seq)
            | (self.layer_id << shift_layer)
            | (self.page_id << shift_page)
        )


# KV cache of a block-sparse-attention block
# shape: [block_size, 2, hidden_size] so it's per layer per KV head
class BsaKVCache:
    def __init__(self, seq_id: int, block_id: int, layer_id: int, head_id: int):
        self.seq_id = seq_id
        self.block_id = block_id
        self.layer_id = layer_id
        self.head_id = head_id

    def __repr__(self):
        return f"BsaKVCache(seq_id={self.seq_id}, block_id={self.block_id}, layer_id={self.layer_id}, head_id={self.head_id})"

    @classmethod
    def bytes(self) -> int:
        config = sim_config()
        return (
            config.block_size
            * 2
            * config.hidden_size
            * config.dtype_bytes
        )

    def get_obj_id(self) -> int:
        config = sim_config()
        bits_seq = ceil(log2(config.batch_size)) if config.batch_size > 1 else 1
        bits_block = ceil(log2(config.num_blocks_per_head)) if config.num_blocks_per_head > 1 else 1
        bits_layer = ceil(log2(config.num_layers)) if config.num_layers > 1 else 1
        bits_head = ceil(log2(config.num_kvheads)) if config.num_kvheads > 1 else 1

        shift_block = 0
        shift_head = shift_block + bits_block
        shift_layer = shift_head + bits_head
        shift_seq = shift_layer + bits_layer

        total_bits = shift_seq + bits_seq
        assert (
            total_bits <= 64
        ), f"Object Id overflow! Need {total_bits} bits but only 64 available"

        return (
            (self.seq_id << shift_seq)
            | (self.block_id << shift_block)
            | (self.layer_id << shift_layer)
            | (self.head_id << shift_head)
        )
