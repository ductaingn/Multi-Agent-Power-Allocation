import os
import argparse

from multi_agent_power_allocation.utils.trainer import Trainer
from multi_agent_power_allocation.utils.train_config import TrainConfig
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
        default="debugging",
        required=False,
        help="Name of the run (for logging purpose)",
    )
    args = arg_parser.parse_args()

    config_path = args.config_path
    config = TrainConfig(config_path)

    trainer = Trainer(
        env_config=config.env_config,
        model_config=config.model_config,
        n_warm_up_step=config.n_warm_up_step,
        wandb_config=config.wandb_config,
        SAC_config=config.SAC_config,
        device=config.device,
        num_env=config.num_env,
    )

    trainer.train(args.run_name)


if __name__ == "__main__":
    main()
