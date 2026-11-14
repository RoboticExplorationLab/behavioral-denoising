"""Configuration loading and applied-action feedback for the four paper tasks."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_experiment_config(experiment, config_path=None):
    path = Path(config_path) if config_path is not None else ROOT / "config" / experiment / "experiment.yaml"
    with path.open() as stream:
        configuration = yaml.safe_load(stream)
    if not isinstance(configuration, dict):
        raise ValueError(f"Expected a configuration mapping in {path}")
    if "extends" in configuration:
        raise ValueError("Release configurations are complete files; inheritance is not supported")
    return configuration



def load_method_configs(experiment, method_ids=None):
    directory = ROOT / "config" / experiment
    if method_ids is None:
        method_ids = load_experiment_config(experiment)["methods"]
    if len(method_ids) != len(set(method_ids)):
        raise ValueError("Each method must be selected only once")
    configurations = []
    for method_id in method_ids:
        if Path(method_id).name != method_id:
            raise ValueError(f"Invalid method identifier: {method_id}")
        path = directory / f"{method_id}.yaml"
        with path.open() as stream:
            configuration = yaml.safe_load(stream)
        if not isinstance(configuration, dict):
            raise ValueError(f"Expected a method configuration mapping in {path}")
        configuration.setdefault("id", method_id)
        if configuration["id"] != method_id:
            raise ValueError(f"Method identifier does not match its filename: {path}")
        configurations.append(configuration)
    return configurations


def record_applied_action(policy, action):
    """Feed the executed input back to policies that retain an input history."""
    callback = getattr(policy, "record_applied_action", None)
    if callback is not None:
        callback(action)
