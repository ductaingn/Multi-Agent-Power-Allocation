import os
import argparse

from multi_agent_power_allocation.utils.trainer import Trainer
from multi_agent_power_allocation.utils.train_config import TrainConfig, load_config
from multi_agent_power_allocation.utils.seed import create_generator, set_seed
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
    arg_parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Overrides `env_config.seed`",
    )
    arg_parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config entry, e.g. `-o env_config.dynamic_obstacles=true "
        "-o checkpoint_config.push_to_hub=true -o checkpoint_config.hf_repo_id=user/repo`",
    )
    args = arg_parser.parse_args()

    overrides = list(args.override)
    if args.seed is not None:
        overrides.append(f"env_config.seed={args.seed}")
    raw_config = load_config(args.config_path, overrides)

    # Set non-numpy seeds and create the RNG generator early
    seed = raw_config.get("env_config", {}).get("seed")
    if seed is not None:
        set_seed(seed)
        rng = create_generator(seed)
    else:
        rng = None

    config = TrainConfig(config_dict=raw_config, rng=rng)

    trainer = Trainer(
        env_config=config.env_config,
        model_config=config.model_config,
        n_warm_up_step=config.n_warm_up_step,
        wandb_config=config.wandb_config,
        SAC_config=config.SAC_config,
        device=config.device,
        num_env=config.num_env,
        checkpoint_config=config.checkpoint_config,
        raw_config=config.raw_config,
    )

    trainer.train(args.run_name)


if __name__ == "__main__":
    main()
