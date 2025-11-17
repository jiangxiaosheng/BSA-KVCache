from config import sim_config
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
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    )


def main():
    setup_logging()

    args = parse_args()
    config = sim_config()
    config.from_yaml(args.config)

    logger.info(f"Simulation config: {config}")

    reader = MobaTraceReader(verbose=config.verbose)
    cache = lcs.LRU(cache_size=int(config.cache_size * 1024 * 1024 * 1024))
    req_miss_ratio, bytes_miss_ratio = cache.process_trace(reader)
    logger.info(f"Request miss ratio: {req_miss_ratio}")
    logger.info(f"Bytes miss ratio: {bytes_miss_ratio}")
    logger.info(f"Cache occupied {cache.get_occupied_byte()} bytes")
    logger.info(f"Cache occupied {cache.get_n_obj()} objects")
    # logger.info(f"Cache: {cache.print_cache()}")


if __name__ == "__main__":
    main()
