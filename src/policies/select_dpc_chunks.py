"""Select-DPC with action chunks for Push-T.

Standardize the dictionary, initialize each query from measured history, refine
its neighborhood, and execute a plan prefix. LkSelector selects windows and
DeePC optimizes the plan. If a query fails, hold the pusher at its measured
position.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np

from ..data import (DeePCConstraints, DeePCCost, DeePCDims, TrajectoryData,
                    TrajectoryDataSet, prepare_dataset)
from ..interfaces import Policy
from ..optimization.hankel import HankelMatrixGenerator
from ..selectors import LkSelector
from .deepc import DeePC


def _vector(value, size, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return vector


class SelectDPCChunks(Policy):
    """Push-T Select-DPC with two-position inputs and six state features.

    ``data`` records explicit pre/post-action timing; conversion never crosses
    episodes. Features are agent x/y, block x/y, cos(theta), sin(theta).
    ``reference`` uses these physical coordinates. The agent coordinates have
    zero tracking weight; the action cost penalizes distance from the mean
    demonstration action in standardized units.

    As in the preliminary experiment, observation and input normalization use
    all aligned samples (not per-lag dictionary normalization). The default
    parameters are its best new-start, eight-action execution configuration.
    """

    def __init__(self, data, *, reference, past=2, execute_steps=8, horizon=30,
                 g_l1=100.0, columns=200, iterations=5, output_weight=100.0,
                 input_weight=0.01, slack=5e6, g_projection=1e4,
                 solver_options=None):
        for name, value in (("past", past), ("horizon", horizon),
                            ("execute_steps", execute_steps), ("columns", columns),
                            ("iterations", iterations)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if execute_steps > horizon:
            raise ValueError("execute_steps must not exceed horizon")
        for name, value in (("g_l1", g_l1), ("output_weight", output_weight),
                            ("input_weight", input_weight), ("slack", slack),
                            ("g_projection", g_projection)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self._reference = _vector(reference, 6, "reference").copy()
        aligned = prepare_dataset(data)
        observations = np.vstack([item.state_trajectory for item in aligned.dataset])
        actions = np.vstack([item.input_trajectory for item in aligned.dataset])
        if observations.shape[1] != 6 or actions.shape[1] != 2:
            raise ValueError("Push-T requires six observation and two action coordinates")
        self._observation_mean = observations.mean(axis=0)
        self._observation_scale = observations.std(axis=0)
        self._action_mean = actions.mean(axis=0)
        self._action_scale = actions.std(axis=0)
        self._observation_scale[self._observation_scale < 1e-12] = 1.0
        self._action_scale[self._action_scale < 1e-12] = 1.0
        standardized = TrajectoryDataSet(
            [TrajectoryData(
                (item.state_trajectory - self._observation_mean) / self._observation_scale,
                (item.input_trajectory - self._action_mean) / self._action_scale,
            ) for item in aligned.dataset],
            aligned.dataset_name, "post_action", aligned.episode_ids,
        )
        self._dims = DeePCDims(past, horizon, 6, 2)
        self._inputs, self._outputs = HankelMatrixGenerator(past, horizon).generate_hankel_matrices(standardized)
        self._selector = LkSelector(order=1, deepc_dims=self._dims, forgetting_factor=1.0)
        self._costs = DeePCCost(
            np.diag(output_weight * np.array([0, 0, 1, 1, 1, 1.0])),
            input_weight * np.eye(2), slack, g_l1, g_projection,
        )
        lower = -self._action_mean / self._action_scale
        upper = (512.0 - self._action_mean) / self._action_scale
        self._bounds = DeePCConstraints(
            np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]]),
            np.array([upper[0], -lower[0], upper[1], -lower[1]]), None, None,
        )
        self._past, self._horizon = past, horizon
        self._execute_steps, self._columns, self._iterations = execute_steps, columns, iterations
        self._solver_options = {"solver": "OSQP", **(solver_options or {})}
        self.reset()

    def reset(self, seed=None):
        self._observations = self._actions = None
        self._chunk = None
        self._chunk_index = 0
        self._awaiting_applied_action = False
        self._queries = self._failed_queries = self._failed_solves = self._steps = 0
        self._query_seconds = 0.0
        self._new_query = False
        self._last_query = None
        self._proposed_action = self._applied_action = None

    @property
    def normalization(self):
        """Coordinate means and scales used to standardize observations and actions."""
        return {"observation_mean": self._observation_mean.copy(),
                "observation_scale": self._observation_scale.copy(),
                "action_mean": self._action_mean.copy(),
                "action_scale": self._action_scale.copy()}

    @property
    def latest_query(self):
        """Last complete query record, including the full plan in physical units."""
        return deepcopy(self._last_query)

    def _plan(self, observation, reference):
        """Refine a local dictionary from explicit measured histories.

        A fresh controller starts each query. Accepted plans carry only between
        its selection rounds, matching the preliminary Push-T adapter.
        """
        started = perf_counter()
        standardized_observation = (observation - self._observation_mean) / self._observation_scale
        standardized_reference = (reference - self._observation_mean) / self._observation_scale
        observations = self._observations.reshape(-1)
        actions = self._actions.reshape(-1)
        controller = None
        rounds = []
        for _ in range(self._iterations):
            if controller is None or not controller.diagnostics()["plan_valid"]:
                output_query = np.r_[observations, np.tile(standardized_observation, self._horizon)]
                input_query = np.r_[actions, np.zeros(2 * self._horizon)]
            else:
                output_query = controller.get_planned_state_trajectory().reshape(-1)
                input_query = controller.get_planned_input_trajectory().reshape(-1)
            selection_started = perf_counter()
            indices, _ = self._selector(input_query[:, None], output_query[:, None],
                                        self._inputs, self._outputs, standardized_reference)
            indices = indices[:self._columns]
            selection_seconds = perf_counter() - selection_started
            previous = controller
            controller = DeePC(
                self._inputs[:, indices], self._outputs[:, indices], self._dims,
                self._costs, self._bounds, np.zeros(2), solver_options=self._solver_options,
            )
            if previous is not None:
                controller.carry_plan(previous)
            controller.set_initial_state(observations, actions)
            solve_started = perf_counter()
            accepted = controller.solve(standardized_reference, selected_indices=indices)
            solve_seconds = perf_counter() - solve_started
            details = controller.diagnostics()
            rounds.append({"status": details["status"], "accepted": bool(accepted),
                           "objective": details["objective"], "selected_indices": indices.copy(),
                           "selection_seconds": selection_seconds, "solve_seconds": solve_seconds})
            self._failed_solves += not accepted
        plan_valid = details["plan_valid"]
        if plan_valid:
            planned_actions = controller.get_planned_input_trajectory()[self._past:]
            predicted_observations = controller.get_planned_state_trajectory()[self._past:]
            predicted_observations = predicted_observations * self._observation_scale + self._observation_mean
            chunk = planned_actions[:self._execute_steps]
            planned_actions = planned_actions * self._action_scale + self._action_mean
        else:
            hold = (observation[:2] - self._action_mean) / self._action_scale
            chunk = np.tile(hold, (self._execute_steps, 1))
            planned_actions = predicted_observations = None
            self._failed_queries += 1
        seconds = perf_counter() - started
        self._last_query = {
            "query_index": self._queries, "step": self._steps, "seconds": seconds,
            "history_observations": self._observations * self._observation_scale + self._observation_mean,
            "history_actions": self._actions * self._action_scale + self._action_mean,
            "reference": reference.copy(), "plan_valid": bool(plan_valid),
            "plan_status": details["plan_status"], "rounds": rounds,
            "planned_actions": planned_actions, "predicted_observations": predicted_observations,
            "commanded_chunk": chunk * self._action_scale + self._action_mean,
            "selected_indices": details["accepted_selected_indices"],
            "coefficients": details["coefficients"], "past_output_slack": details["past_output_slack"],
            "action_source": "planned" if plan_valid else "hold_position",
        }
        self._queries += 1
        self._query_seconds += seconds
        return chunk.copy()

    def act(self, observation, reference=None):
        observation = _vector(observation, 6, "observation")
        reference = self._reference if reference is None else _vector(reference, 6, "reference")
        standardized = (observation - self._observation_mean) / self._observation_scale
        if self._observations is None:
            self._observations = np.tile(standardized, (self._past, 1))
            hold = (observation[:2] - self._action_mean) / self._action_scale
            self._actions = np.tile(hold, (self._past, 1))
        else:
            self._observations = np.vstack((self._observations[1:], standardized))
        self._new_query = self._chunk is None or self._chunk_index == self._execute_steps
        if self._new_query:
            self._chunk = self._plan(observation, reference)
            self._chunk_index = 0
        command = self._chunk[self._chunk_index].copy()
        self._chunk_index += 1
        # Default to the proposed input; feedback replaces it after clipping.
        self._actions = np.vstack((self._actions[1:], command))
        self._proposed_action = command * self._action_scale + self._action_mean
        self._applied_action = None
        self._awaiting_applied_action = True
        self._steps += 1
        return self._proposed_action.copy()

    def record_applied_action(self, action):
        action = _vector(action, 2, "applied action")
        if not self._awaiting_applied_action:
            return False
        self._actions[-1] = (action - self._action_mean) / self._action_scale
        self._applied_action = action.copy()
        self._awaiting_applied_action = False
        return True

    def diagnostics(self):
        last = self._last_query
        return {
            "dictionary_columns": self._inputs.shape[1], "selected_columns": min(self._columns, self._inputs.shape[1]),
            "queries": self._queries, "solver_failures": self._failed_queries,
            "failed_solves": self._failed_solves, "query_ms": 1e3 * self._query_seconds / max(1, self._queries),
            "new_query": self._new_query, "query_index": None if last is None else last["query_index"],
            "query_seconds": last["seconds"] if last is not None and self._new_query else 0.0,
            "chunk_index": self._chunk_index - 1 if last is not None else None,
            "plan_valid": last is not None and last["plan_valid"],
            "action_source": None if last is None else last["action_source"],
            "proposed_action": None if self._proposed_action is None else self._proposed_action.copy(),
            "applied_action": None if self._applied_action is None else self._applied_action.copy(),
        }
