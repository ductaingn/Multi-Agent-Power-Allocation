import os
import argparse

from multi_agent_power_allocation.utils.trainer import Trainer, process_default_config
from multi_agent_power_allocation import BASE_DIR


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-dc",
    "--config_path",
    type=str,
    default=os.path.join(BASE_DIR, "run", "default_config.yaml"),
    required=False,
    help="Base path for configs and data",
)
arg_parser.add_argument(
    "-n",
    "--name_of_run",
    type=str,
    default="train",
    required=False,
    help="Name of the run (for logging purpose)",
)
args = arg_parser.parse_args()


if __name__ == "__main__":
    default_config_path = args.config_path
    default_config: dict = process_default_config(default_config_path)
    config = default_config

    trainer = Trainer(**config)
    result = trainer.train(args.name_of_run)
