from dataclasses import dataclass
import yaml


@dataclass
class SimConfig:
    # Trace parameters
    trace_dir: str = ""

    # Model parameters
    hidden_size: int = 0
    num_layers: int = 0
    # Number of attention heads, equals to the number of query heads
    num_heads: int = 0
    # Number of queries heads sharing the same KV head in grouped attention
    kv_group_size: int = 0
    # Number of tokens in a block
    block_size: int = 0
    context_length: int = 0
    top_k: int = 3

    # kv cache parameters
    # paged attention page size, in tokens
    page_size: int = 32
    # whether to store all kvheads in the kv cache
    # Set to True for standard paged attention
    # Set to False for block sparse attention
    store_all_kvheads: bool = True
    # bytes per dtype. e.g. float16 is 2 bytes, float32 is 4 bytes
    dtype_bytes: int = 2

    # Cache parameters
    cache_size: int = 0  # in GB
    eviction_algorithm: str = "lru"

    # Misc parameters
    # Number of requests in a batch during decoding
    # Currently we only support batch size = 1 and each request
    # may have different output lengths
    batch_size: int = 1
    # Whether to use continuous batching
    # This is just a hint for myself to notice that people in practice
    # will use this technique to better gpu utilization, but we may never
    # have enough time to implement it
    continuous_batching: bool = False

    def __post_init__(self):
        if self.batch_size != 1:
            raise NotImplementedError("Only batch size = 1 is supported for now")

    @property
    def num_kvheads(self) -> int:
        assert (
            self.kv_group_size > 0 and self.num_heads % self.kv_group_size == 0
        ), "num_heads (={self.num_heads}) must be a multiple of kv_group_size (={self.kv_group_size})"
        return self.num_heads // self.kv_group_size

    @property
    def num_blocks_per_head(self) -> int:
        assert (
            self.context_length % self.block_size == 0
        ), "context_length (={self.context_length}) must be a multiple of block_size (={self.block_size})"
        return self.context_length // self.block_size

    def from_yaml(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        self.from_dict(config_dict)

    def from_dict(self, config_dict: dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")


_global_config: SimConfig = SimConfig()


def sim_config() -> SimConfig:
    return _global_config
