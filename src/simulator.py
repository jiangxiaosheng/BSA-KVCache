from config import sim_config, SimConfig
from argparse import ArgumentParser
import libcachesim as lcs
from trace_reader import MobaTraceReader
from momentum_cache import MomentumDecayCache
import logging
import os
from tqdm import tqdm
from typing import Union

logger = logging.getLogger(__name__)

# Custom eviction algorithms (not from libcachesim)
CUSTOM_ALGORITHMS = {"momentum_decay"}

# libcachesim-based eviction algorithms
LCS_ALGORITHMS = {
    "lru": lcs.LRU,
    "fifo": lcs.FIFO,
    "lfu": lcs.LFU,
    "arc": lcs.ARC,
    "s3fifo": lcs.S3FIFO,
    "sieve": lcs.Sieve,
    "lirs": lcs.LIRS,
    "twoq": lcs.TwoQ,
    "slru": lcs.SLRU,
    "random": lcs.Random,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%H:%M:%S",
    )


def is_custom_algorithm(algorithm: str) -> bool:
    return algorithm.lower() in CUSTOM_ALGORITHMS


def setup_cache(config: SimConfig) -> Union[lcs.CacheBase, MomentumDecayCache]:
    algorithm = config.eviction_algorithm.lower()
    cache_size_bytes = int(config.cache_size * 1024 * 1024 * 1024)
    
    if algorithm == "momentum_decay":
        cache = MomentumDecayCache(cache_size=cache_size_bytes, beta=0.9)
        logger.info(f"Using cache eviction algorithm: momentum_decay (beta=0.9)")
        return cache
    
    cache_class = LCS_ALGORITHMS.get(algorithm)
    if cache_class is None:
        raise ValueError(
            f"Unknown cache eviction algorithm: {config.eviction_algorithm}. "
            f"Available: {list(LCS_ALGORITHMS.keys()) + list(CUSTOM_ALGORITHMS)}"
        )
    cache = cache_class(cache_size=cache_size_bytes)
    logger.info(f"Using cache eviction algorithm: {config.eviction_algorithm}")
    return cache


def process_trace_with_momentum(cache: MomentumDecayCache, reader: MobaTraceReader):
    """Process trace using MomentumDecayCache with score information."""
    for obj_id, obj_size, score in reader.generate_requests_with_scores():
        cache.access(obj_id, obj_size, score)
    return cache.get_miss_ratio()


def get_num_traces(config: SimConfig) -> int:
    return len(os.listdir(config.trace_dir))


def main():
    setup_logging()

    args = parse_args()
    config = sim_config()
    config.from_yaml(args.config)

    logger.info(f"Simulation config: {config}")

    num_traces = get_num_traces(config)
    total_req_miss_ratio = 0
    total_bytes_miss_ratio = 0
    use_custom = is_custom_algorithm(config.eviction_algorithm)
    
    for i in tqdm(range(num_traces), desc="Simulating traces"):
        trace_dir = os.path.join(config.trace_dir, f"trace{i:04d}")
        print(f"Processing trace {trace_dir}")
        reader = MobaTraceReader(trace_dir=trace_dir, verbose=config.verbose)
        cache = setup_cache(config)
        
        if use_custom:
            req_miss_ratio, bytes_miss_ratio = process_trace_with_momentum(cache, reader)
        else:
            req_miss_ratio, bytes_miss_ratio = cache.process_trace(reader)
        
        total_req_miss_ratio += req_miss_ratio
        total_bytes_miss_ratio += bytes_miss_ratio
        logger.info(f"Cache occupied {cache.get_occupied_byte()} bytes")
    
    logger.info(f"Average request miss ratio: {total_req_miss_ratio / num_traces}")
    logger.info(f"Average bytes miss ratio: {total_bytes_miss_ratio / num_traces}")


if __name__ == "__main__":
    main()
