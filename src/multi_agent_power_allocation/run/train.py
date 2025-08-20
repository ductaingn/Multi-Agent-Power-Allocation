from multi_agent_power_allocation.utils.trainer import Trainer, process_default_config
from multi_agent_power_allocation import BASE_DIR
import os
import argparse
import yaml


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-dc', '--use_default_config', type=bool, default=True, required=False, help='Base path for configs and data')
args = arg_parser.parse_args()


if __name__=="__main__":
    if args.use_default_config:
        default_config_path = os.path.join(BASE_DIR, "run", "default_config.yaml")
        default_config:dict = process_default_config(default_config_path)
        config = default_config

    trainer = Trainer(**config)
    trainer.train()