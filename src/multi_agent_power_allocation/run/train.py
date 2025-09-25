import os
import argparse
import pprint

from multi_agent_power_allocation.utils.trainer import Trainer, parse_config
from multi_agent_power_allocation import BASE_DIR


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-cp",
        "--config_path",
        type=str,
        default=os.path.join(BASE_DIR, "run", "default_config.yaml"),
        required=False,
        help="Base path for configs and data",
    )
    arg_parser.add_argument(
        "-rn",
        "--run_name",
        type=str,
        default="train",
        required=False,
        help="Name of the run (for logging purpose)",
    )
    args = arg_parser.parse_args()

    config_path = args.config_path
    config: dict = parse_config(config_path)

    trainer = Trainer(**config)

    result = trainer.train(args.run_name)

    result.pprint_asdict()


if __name__ == "__main__":
    main()
