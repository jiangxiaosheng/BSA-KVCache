from config import sim_config, SimConfig
from argparse import ArgumentParser
import libcachesim as lcs
from trace_reader import MobaTraceReader
import logging
import os
from tqdm import tqdm
logger = logging.getLogger(__name__)


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


def setup_cache(config: SimConfig) -> lcs.CacheBase:
    EVICTION_ALGORITHMS = {
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
    cache_class = EVICTION_ALGORITHMS.get(config.eviction_algorithm.lower())
    if cache_class is None:
        raise ValueError(
            f"Unknown cache eviction algorithm: {config.eviction_algorithm}"
        )
    # TODO: may need unique parameters for different algorithms
    cache = cache_class(cache_size=int(config.cache_size * 1024 * 1024 * 1024))
    logger.info(f"Using cache eviction algorithm: {config.eviction_algorithm}")
    return cache


def get_num_traces(config: SimConfig) -> int:
    return len(os.listdir(config.trace_dir))


def main():
    setup_logging()

    args = parse_args()
    config = sim_config()
    config.from_yaml(args.config)

    logger.info(f"Simulation config: {config}")

    # num_traces = get_num_traces(config)
    num_traces = 1
    total_req_miss_ratio = 0
    total_bytes_miss_ratio = 0
    for i in tqdm(range(num_traces), desc="Simulating traces"):
        trace_dir = os.path.join(config.trace_dir, f"trace{i:04d}")
        print(f"Processing trace {trace_dir}")
        reader = MobaTraceReader(trace_dir=trace_dir, verbose=config.verbose)
        cache = setup_cache(config)
        req_miss_ratio, bytes_miss_ratio = cache.process_trace(reader)
        total_req_miss_ratio += req_miss_ratio
        total_bytes_miss_ratio += bytes_miss_ratio
        # logger.info(f"Request miss ratio: {req_miss_ratio}")
        # logger.info(f"Bytes miss ratio: {bytes_miss_ratio}")
        logger.info(f"Cache occupied {cache.get_occupied_byte()} bytes")
        # logger.info(f"Cache occupied {cache.get_n_obj()} objects")
    logger.info(f"Average request miss ratio: {total_req_miss_ratio / num_traces}")
    logger.info(f"Average bytes miss ratio: {total_bytes_miss_ratio / num_traces}")


if __name__ == "__main__":
    main()
