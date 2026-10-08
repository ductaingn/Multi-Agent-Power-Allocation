> [!IMPORTANT]
> **This repository has moved to [ductaingn/MultiConnectRL](https://github.com/ductaingn/MultiConnectRL.git).**
> It is archived and no longer maintained. Please open issues and pull requests in the new repository.

# Multi-Agent Power Allocation

Multi-agent DRL for joint interface selection, packet allocation and power control in integrated
Sub-6GHz/mmWave networks. Each AP is an agent of a PettingZoo `ParallelEnv`.

Algorithms (`env_config.algorithm_list`, one per AP): `SACPA` (SACRA), `SACPF` (SACRA without power
control, "SACRA-Va"), `RAQL`, `DQN`, `Random`.

## Training

```bash
poetry install
python -m multi_agent_power_allocation.run.train -rn "my run" -s 1 \
  -o "env_config.algorithm_list=[SACPA,SACPA,SACPA,SACPA]" \
  -o env_config.dynamic_obstacles=true
```

- `-cp` config file (default `src/multi_agent_power_allocation/run/default_config.yaml`)
- `-s` seed (overrides `env_config.seed`)
- `-o key.subkey=value` overrides any config entry; values are parsed as YAML (repeatable)

`env_config.wc_cluster_config.P_sum` is the power budget per AP in dBm. `env_config.baseline_full_power: true`
makes RAQL/DQN use the whole budget (split like SACPF) instead of `P_sum / (N + M)` per link.

## Checkpoints and Hugging Face Hub

Configured by `checkpoint_config` (defaults in `utils/checkpoint.py`):

| key | default | |
|---|---|---|
| `enabled` | `true` | |
| `save_dir` | `checkpoints` | a run is saved under `<save_dir>/<run name>/` |
| `save_freq` | `5000` | env steps between periodic saves (`step_<n>/`); 0 disables |
| `keep_last` | `2` | periodic checkpoints kept |
| `save_best` | `true` | `best/`: highest mean reward over the last `save_freq` steps |
| `upload_to_wandb` | `false` | log the run folder as a WandB model artifact |
| `push_to_hub` | `false` | push the run folder to the Hugging Face Hub at the end |
| `hf_repo_id` | `null` | e.g. `user/sacra-power-allocation` (required with `push_to_hub`) |
| `hf_private` | `true` | |
| `hf_push_every_save` | `false` | also push each periodic checkpoint |
| `hf_token_env` | `HF_TOKEN` | env var holding the token (else the `huggingface-cli login` cache) |

A final checkpoint is always written to `final/`, with the YAML config of the run in `config.yaml`.
Pushing requires `pip install huggingface_hub`.

```bash
HF_TOKEN=hf_... python -m multi_agent_power_allocation.run.train -rn "sacra 3 devices" \
  -o checkpoint_config.push_to_hub=true -o checkpoint_config.hf_repo_id=<user>/sacra-power-allocation
```

Restoring policies (e.g. for evaluation):

```python
import yaml
from multi_agent_power_allocation.utils.train_config import TrainConfig
from multi_agent_power_allocation.utils.checkpoint import load_policies

run_dir = "checkpoints/sacra_3_devices"
config = TrainConfig(config_dict=yaml.safe_load(open(f"{run_dir}/config.yaml")))
policies = config.env_config["algorithm_mapping"]
load_policies(f"{run_dir}/final", policies)
```

## Channel data

The channel realizations are not part of the package: they are on the Hugging Face Hub dataset
[`ductaingn/sacra-channel-data`](https://huggingface.co/datasets/ductaingn/sacra-channel-data), and a
scenario is downloaded on first use (only that scenario) into the data root:

1. `env_config.data_dir` in the config,
2. else the `MAPA_DATA_DIR` environment variable,
3. else the `data/` folder of a source checkout (ignored by git),
4. else `~/.cache/multi_agent_power_allocation`.

| Scenario | APs | Devices per AP | Subchannels / beams | Frames |
|---|---|---|---|---|
| `scenario_1`, `scenario_2` | 2 | 3 | 5 | 10,000 |
| `scenario_3` | 3 | 3 | 5 | 10,000 |
| `scenario_4` | 4 | 3 | 5 | 10,000 |
| `scenario_4_30k` | 4 | 3 | 4 | 30,000 |
| `scenario_5` | 4 | 10 | 16 | 10,000 |
| `scenario_5_30k` | 4 | 10 | 16 | 30,000 |

New scenarios (which must cover `max_num_step` frames) can be generated into the data root:

```python
from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
    WirelessCommunicationCluster,
)

WirelessCommunicationCluster.generate_data(
    scenario_name="my_scenario", num_cluster=4, num_timestep=30000,
    num_device=3, num_subchannel=4, num_beam=4, seed=1,
)
```
