"""
Location of the channel data (scenarios).

A scenario is a folder `<scenario>/cluster_<i>/{positions.json, h_tilde.pickle}` under the data
root, which is, in order of priority:

1. the `data_dir` argument (`env_config.data_dir` in the config),
2. the `MAPA_DATA_DIR` environment variable,
3. the `data/` folder at the root of a source checkout of this repository,
4. `~/.cache/multi_agent_power_allocation`.

A missing scenario is downloaded from the Hugging Face Hub dataset `HF_DATASET_REPO` (only that
scenario), or can be generated with `WirelessCommunicationCluster.generate_data`.
"""

import os
from pathlib import Path

HF_DATASET_REPO = "ductaingn/sacra-channel-data"
DATA_DIR_ENV_VAR = "MAPA_DATA_DIR"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "multi_agent_power_allocation"

_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def data_root(data_dir: str | os.PathLike | None = None) -> Path:
    """Folder that holds the scenarios (see the module docstring for the priority order)."""
    if data_dir:
        return Path(data_dir).expanduser()
    if os.environ.get(DATA_DIR_ENV_VAR):
        return Path(os.environ[DATA_DIR_ENV_VAR]).expanduser()
    checkout_data = _SOURCE_ROOT / "data"
    if checkout_data.is_dir() and (_SOURCE_ROOT / "pyproject.toml").is_file():
        return checkout_data
    return DEFAULT_DATA_DIR


def download_scenario(scenario: str, root: Path) -> None:
    """Download one scenario of `HF_DATASET_REPO` into `root/<scenario>`."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Downloading the channel data requires `huggingface_hub` "
            "(pip install huggingface_hub)"
        ) from e

    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[f"{scenario}/*"],
        local_dir=root,
    )
    if not (root / scenario).is_dir():
        raise FileNotFoundError(
            f"Scenario `{scenario}` is not in the dataset {HF_DATASET_REPO}; generate it "
            "with `WirelessCommunicationCluster.generate_data`"
        )


def scenario_dir(
    scenario: str, data_dir: str | os.PathLike | None = None, download: bool = True
) -> Path:
    """Folder of `scenario`, downloaded from the Hugging Face Hub if it is missing."""
    root = data_root(data_dir)
    path = root / scenario
    if not path.is_dir():
        if not download:
            raise FileNotFoundError(f"Scenario `{scenario}` not found in {root}")
        print(f"Scenario `{scenario}` not found in {root}: downloading it...")
        download_scenario(scenario, root)
    return path
