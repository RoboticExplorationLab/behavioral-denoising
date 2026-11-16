"""Select-DPC with action chunks for Push-T.

Standardize the dictionary, initialize each query from measured history, refine
its neighborhood, and execute a plan prefix. LkSelector selects windows and
DeePC optimizes the plan. If a query fails, hold the pusher at its measured
position.
"""

from copy import deepcopy
from dataclasses import replace
from time import perf_counter

import numpy as np

from ..data import (DeePCConstraints, DeePCCost, DeePCDims, TrajectoryData,
                    TrajectoryDataSet, prepare_dataset)
from ..interfaces import Policy
from ..optimization.hankel import HankelMatrixGenerator
from ..selectors import LkSelector
from ..tasks.pusht_coordinates import PushTCoordinates
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
    ``act`` and ``reference`` always use physical coordinates. The shared
    coordinate adapter determines the dictionary and optimization variables.
    Agent position has zero tracking weight. The action penalty retains its
    physical meaning in either representation: distance from the mean world
    demonstration target, scaled by the world demonstration standard deviation.

    As in the preliminary experiment, observation and input normalization use
    all aligned samples in world mode. Relative observation statistics also
    use aligned samples; relative action statistics pool every past and future
    target in the transformed windows. Each relative window has one fixed
    planning frame, so its input coordinates cannot be assigned per episode.
    """

    def __init__(self, data, *, reference, past=2, execute_steps=8, horizon=30,
                 g_l1=100.0, columns=200, iterations=5, output_weight=100.0,
                 input_weight=0.01, slack=5e6, g_projection=1e4,
                 solver_options=None, coordinates=None):
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
        self.coordinates = (coordinates if isinstance(coordinates, PushTCoordinates)
                            else PushTCoordinates("world" if coordinates is None else coordinates))
        self._relative = self.coordinates.name == "block_relative"
        self._reference = _vector(reference, 6, "reference").copy()
        aligned = prepare_dataset(data)
        observations = np.vstack([item.state_trajectory for item in aligned.dataset])
        actions = np.vstack([item.input_trajectory for item in aligned.dataset])
        if observations.shape[1] != 6 or actions.shape[1] != 2:
            raise ValueError("Push-T requires six observation and two action coordinates")
        self._world_action_mean = actions.mean(axis=0)
        self._world_action_scale = actions.std(axis=0)
        self._world_action_scale[self._world_action_scale < 1e-12] = 1.0
        if self._relative:
            observations = self.coordinates.observations(observations)
            inputs, outputs = self.coordinates.dictionary(data, past, horizon,
                                                           transform_past_inputs=True)
            actions = inputs.T.reshape(-1, 2)
        self._observation_mean = observations.mean(axis=0)
        self._observation_scale = observations.std(axis=0)
        self._action_mean = actions.mean(axis=0)
        self._action_scale = actions.std(axis=0)
        self._observation_scale[self._observation_scale < 1e-12] = 1.0
        self._action_scale[self._action_scale < 1e-12] = 1.0
        self._dims = DeePCDims(past, horizon, 6, 2)
        if self._relative:
            input_rows = inputs.T.reshape(-1, 2)
            output_rows = outputs.T.reshape(-1, 6)
            standardized_inputs = (input_rows - self._action_mean) / self._action_scale
            standardized_outputs = (output_rows - self._observation_mean) / self._observation_scale
            self._inputs = standardized_inputs.reshape(inputs.shape[1], -1).T
            self._outputs = standardized_outputs.reshape(outputs.shape[1], -1).T
        else:
            standardized = TrajectoryDataSet(
                [TrajectoryData(
                    (item.state_trajectory - self._observation_mean) / self._observation_scale,
                    (item.input_trajectory - self._action_mean) / self._action_scale,
                ) for item in aligned.dataset],
                aligned.dataset_name, "post_action", aligned.episode_ids,
            )
            self._inputs, self._outputs = HankelMatrixGenerator(past, horizon).generate_hankel_matrices(standardized)
        self._selector = LkSelector(order=1, deepc_dims=self._dims, forgetting_factor=1.0)
        self._costs = DeePCCost(
            np.diag(output_weight * np.array([0, 0, 1, 1, 1, 1.0])),
            input_weight * np.eye(2), slack, g_l1, g_projection,
        )
        self._input_weight = input_weight
        self._bounds = None
        if not self._relative:
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
        self._world_observations = self._world_actions = None
        self._action_anchor = None
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
        values = {"observation_mean": self._observation_mean.copy(),
                  "observation_scale": self._observation_scale.copy(),
                  "action_mean": self._action_mean.copy(),
                  "action_scale": self._action_scale.copy()}
        if self._relative:
            values.update(world_action_mean=self._world_action_mean.copy(),
                          world_action_scale=self._world_action_scale.copy())
        return values

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
        self._action_anchor = observation.copy()
        policy_observation, policy_reference = observation, reference
        costs, bounds, input_reference = self._costs, self._bounds, np.zeros(2)
        if self._relative:
            policy_observation = self.coordinates.observations(observation)
            policy_reference = self.coordinates.observations(reference)
            history_actions = self.coordinates.encode_actions(self._world_actions, self._action_anchor)
            self._actions = (history_actions - self._action_mean) / self._action_scale
            input_matrix, input_bound = self.coordinates.input_constraints(
                self._action_anchor, self._action_mean, self._action_scale)
            bounds = DeePCConstraints(input_matrix, input_bound, None, None)
            input_cost, input_reference = self.coordinates.input_cost(
                self._action_anchor, self._action_mean, self._action_scale,
                self._world_action_mean, self._world_action_scale, self._input_weight)
            costs = replace(self._costs, R=input_cost)
        standardized_observation = (policy_observation - self._observation_mean) / self._observation_scale
        standardized_reference = (policy_reference - self._observation_mean) / self._observation_scale
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
                costs, bounds, input_reference, solver_options=self._solver_options,
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
        coordinate_actions = predicted_policy_observations = None
        if plan_valid:
            planned_actions = controller.get_planned_input_trajectory()[self._past:]
            predicted_observations = controller.get_planned_state_trajectory()[self._past:]
            predicted_observations = predicted_observations * self._observation_scale + self._observation_mean
            chunk = planned_actions[:self._execute_steps]
            planned_actions = planned_actions * self._action_scale + self._action_mean
            if self._relative:
                coordinate_actions = planned_actions.copy()
                planned_actions = self.coordinates.decode_actions(coordinate_actions, self._action_anchor)
                chunk = planned_actions[:self._execute_steps].copy()
                # A linear prediction need not have unit-length cos/sin.
                # Keep its actual policy features instead of inventing a
                # physical pusher position from an invalid predicted rotation.
                predicted_policy_observations = predicted_observations
                predicted_observations = None
        else:
            hold = (observation[:2] if self._relative
                    else (observation[:2] - self._action_mean) / self._action_scale)
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
            "commanded_chunk": (chunk.copy() if self._relative
                                  else chunk * self._action_scale + self._action_mean),
            "selected_indices": details["accepted_selected_indices"],
            "coefficients": details["coefficients"], "past_output_slack": details["past_output_slack"],
            "action_source": "planned" if plan_valid else "hold_position",
        }
        self._last_query.update(representation=self.coordinates.name,
                                action_anchor_observation=self._action_anchor.copy())
        if self._relative:
            self._last_query.update(
                history_observations=self._world_observations.copy(),
                history_actions=self._world_actions.copy(),
                history_policy_observations=self._observations * self._observation_scale + self._observation_mean,
                history_policy_actions=self._actions * self._action_scale + self._action_mean,
                coordinate_actions=coordinate_actions,
                predicted_policy_observations=predicted_policy_observations,
                input_cost_matrix=costs.R.copy(), input_reference=input_reference.copy(),
                input_constraint_matrix=bounds.A_u.copy(), input_constraint_bound=bounds.b_u.copy(),
            )
        self._queries += 1
        self._query_seconds += seconds
        return chunk.copy()

    def act(self, observation, reference=None):
        observation = _vector(observation, 6, "observation")
        reference = self._reference if reference is None else _vector(reference, 6, "reference")
        policy_observation = self.coordinates.observations(observation) if self._relative else observation
        standardized = (policy_observation - self._observation_mean) / self._observation_scale
        if self._observations is None:
            self._observations = np.tile(standardized, (self._past, 1))
            hold_position = (self.coordinates.encode_actions(observation[:2], observation)
                             if self._relative else observation[:2])
            hold = (hold_position - self._action_mean) / self._action_scale
            self._actions = np.tile(hold, (self._past, 1))
            if self._relative:
                self._world_observations = np.tile(observation, (self._past, 1))
                self._world_actions = np.tile(observation[:2], (self._past, 1))
        else:
            self._observations = np.vstack((self._observations[1:], standardized))
            if self._relative:
                self._world_observations = np.vstack((self._world_observations[1:], observation))
        self._new_query = self._chunk is None or self._chunk_index == self._execute_steps
        if self._new_query:
            self._chunk = self._plan(observation, reference)
            self._chunk_index = 0
        command = self._chunk[self._chunk_index].copy()
        self._chunk_index += 1
        # Default to the proposed input; feedback replaces it after clipping.
        if self._relative:
            self._world_actions = np.vstack((self._world_actions[1:], command))
            coordinate_action = self.coordinates.encode_actions(command, self._action_anchor)
            standardized_action = (coordinate_action - self._action_mean) / self._action_scale
            self._actions = np.vstack((self._actions[1:], standardized_action))
            self._proposed_action = command.copy()
        else:
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
        if self._relative:
            self._world_actions[-1] = action
            coordinate_action = self.coordinates.encode_actions(action, self._action_anchor)
            self._actions[-1] = (coordinate_action - self._action_mean) / self._action_scale
        else:
            self._actions[-1] = (action - self._action_mean) / self._action_scale
        self._applied_action = action.copy()
        self._awaiting_applied_action = False
        return True

    def diagnostics(self):
        last = self._last_query
        return {
            "representation": self.coordinates.name,
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
