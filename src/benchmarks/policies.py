"""Dictionary construction shared by the two simulator benchmarks."""

from pathlib import Path

import numpy as np

from src.data import DeePCConstraints, DeePCControllerArgs, DeePCCost, DeePCDims, TrajectoryDataSet
from src.optimization.hankel import HankelMatrixGenerator
from src.selectors import LkSelector, IsoMapEmbeddedDistances


def fixed_dataset(data, settings):
    """Apply an explicit column budget; None retains the complete offline bank."""
    maximum_columns = settings.get("maximum_columns", 800)
    if maximum_columns is None:
        length = settings["past"] + settings["future"]
        columns = sum(max(0, min(len(item.input_trajectory), settings["hankel_length"]) - length + 1)
                      for item in data.dataset)
        return data, columns
    window_length = settings["past"] + settings["future"]
    columns = 0
    episode_count = 0
    for trajectory in data.dataset:
        if columns >= maximum_columns:
            break
        episode_length = min(len(trajectory.input_trajectory), settings["hankel_length"])
        columns += max(0, episode_length - window_length + 1)
        episode_count += 1
    subset = TrajectoryDataSet(data.dataset[:episode_count], data.dataset_name, data.timing)
    return subset, columns


def dictionary(data, past, future, cache=None):
    """Reuse a prepared dictionary within one dataset's experiment run."""
    key = (past, future)
    if cache is None:
        return HankelMatrixGenerator(past, future).generate_hankel_matrices(data)
    if key not in cache:
        cache[key] = HankelMatrixGenerator(past, future).generate_hankel_matrices(data)
    return cache[key]


def build_deepc(policy_class, settings, data, task, cache=None):
    """Construct the explicitly chosen DeePC class using its experiment configuration."""
    if data.timing != "post_action":
        raise ValueError("Prepare the dataset before constructing DeePC")
    if settings.get("factorization", "none") != "none":
        raise ValueError("Paper experiments use dictionary-coordinate optimization")
    family = settings["family"]
    if family == "deepc":
        data, _ = fixed_dataset(data, settings)
    dimensions = DeePCDims(settings["past"], settings["future"], task.observation_dim, task.action_dim)
    costs = DeePCCost(np.diag(settings["Q"]), np.diag(settings["R"]),
                     settings["slack"], settings["g_l1"], settings["g_projection"])
    bounds = DeePCConstraints(*(None if settings[key] is None else np.asarray(settings[key])
                               for key in ("A_u", "b_u", "A_y", "b_y")))
    input_reference = settings["input_reference"]
    if isinstance(input_reference, str) and input_reference == "rocket_hover":
        environment = task.dynamics.env.unwrapped
        mass, _ = environment.get_mass_properties()
        input_reference = [-environment.cfg.gravity * mass / environment.cfg.main_engine_thrust, 0, 0]
    arguments = DeePCControllerArgs(data, dimensions, settings["hankel_length"], costs, bounds,
                                   np.asarray(input_reference), False)
    if family == "deepc":
        return policy_class.create_from_data(arguments, False, settings["solver"])
    input_windows, output_windows = dictionary(data, dimensions.T_past, dimensions.T_fut, cache)
    if family == "windowed_deepc":
        columns = settings["window_columns"]
        return policy_class(input_windows[:, :columns], output_windows[:, :columns], arguments,
                            settings["solver"], factorization_threshold=None)
    if family != "select_dpc":
        raise ValueError(f"Unknown DeePC family: {family}")
    selector_options = dict(settings.get("selector_options", {}))
    if settings["selector"] == "norm":
        selector = LkSelector(deepc_dims=dimensions, **selector_options)
    elif settings["selector"] == "isomap":
        # This changes setup cost only. An explicit null disables persistence.
        selector_options.setdefault("cache_dir", Path(__file__).resolve().parents[2] / ".cache" / "isomap")
        selector = IsoMapEmbeddedDistances(data, dimensions, hankel_matrices=(input_windows, output_windows),
                                          **selector_options)
    else:
        raise ValueError(f"Unsupported benchmark selector: {settings['selector']}")
    return policy_class(arguments, selector, settings["columns"], settings["iterations"],
                        solver_options=settings["solver"], hankel_matrices=(input_windows, output_windows),
                        factorization_threshold=None)


def build_imitation(policy_class, settings, parameters, data, observation_dim, action_dim, cache=None):
    """Create an imitation method; the script supplies its class explicitly."""
    options = dict(parameters)
    dimensions = DeePCDims(1, 1, observation_dim, action_dim)
    input_windows, output_windows = dictionary(data, dimensions.T_past, dimensions.T_fut, cache)
    return policy_class(input_windows, output_windows, dimensions, **options)
