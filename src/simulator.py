from config import SimConfig
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    config = SimConfig()
    config.from_yaml(args.config)

if __name__ == "__main__":
    main()