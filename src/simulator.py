from config import sim_config, SimConfig
from argparse import ArgumentParser
import libcachesim as lcs
from trace_reader import MobaTraceReader
import logging

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
        raise ValueError(f"Unknown cache eviction algorithm: {config.eviction_algorithm}")
    # TODO: may need unique parameters for different algorithms
    cache = cache_class(cache_size=int(config.cache_size * 1024 * 1024 * 1024))
    logger.info(f"Using cache eviction algorithm: {config.eviction_algorithm}")
    return cache


def main():
    setup_logging()

    args = parse_args()
    config = sim_config()
    config.from_yaml(args.config)

    logger.info(f"Simulation config: {config}")

    reader = MobaTraceReader(verbose=config.verbose)
    cache = setup_cache(config)
    req_miss_ratio, bytes_miss_ratio = cache.process_trace(reader)
    logger.info(f"Request miss ratio: {req_miss_ratio}")
    logger.info(f"Bytes miss ratio: {bytes_miss_ratio}")
    logger.info(f"Cache occupied {cache.get_occupied_byte()} bytes")
    logger.info(f"Cache occupied {cache.get_n_obj()} objects")
    # logger.info(f"Cache: {cache.print_cache()}")


if __name__ == "__main__":
    main()
