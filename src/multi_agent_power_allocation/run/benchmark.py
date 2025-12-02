import os
import argparse

from multi_agent_power_allocation.utils.trainer import Trainer, parse_config
from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.algorithms.algorithm_register import Algorithm


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
        default="Train with dynamic obstacles",
        required=False,
        help="Name of the run (for logging purpose)",
    )
    args = arg_parser.parse_args()

    config_path = args.config_path
    config: dict = parse_config(config_path)

    for algorithm in Algorithm:
        config["env_config"]["algorithm_list"] = [algorithm] * config["env_config"][
            "num_cluster"
        ]
        trainer = Trainer(**config)

        run_name = args.run_name + f"with 4 {algorithm.name} clusters"

        trainer.train(run_name)


if __name__ == "__main__":
    main()
