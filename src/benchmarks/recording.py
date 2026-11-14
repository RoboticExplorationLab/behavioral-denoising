"""Read simulator state at existing transitions; never advance it for recording."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from .records import _atomic_write, _write_json


def body_geometry(name, body):
    fixtures = []
    for fixture in body.fixtures:
        shape = fixture.shape
        if hasattr(shape, "vertices"):
            kind = "edge" if "edge" in type(shape).__name__.lower() else "polygon"
            fixtures.append({"type": kind, "vertices": [list(vertex) for vertex in shape.vertices]})
        elif hasattr(shape, "radius"):
            fixtures.append({"type": "circle", "radius": float(shape.radius), "center": list(shape.pos)})
        else:
            raise ValueError(f"Unsupported simulator geometry: {type(shape).__name__}")
    return {"name": name, "fixtures": fixtures, "position": list(body.position),
            "angle": float(body.angle), "color1": list(getattr(body, "color1", (100, 100, 100))),
            "color2": list(getattr(body, "color2", (0, 0, 0)))}


class TraceRecorder:
    """Save controller proposals, executed inputs, references and rendering poses.

    The experiment calls these methods immediately after its existing reset,
    reference and step calls. There are no extra policy calls, random draws,
    reference updates or physics steps.
    """

    def __init__(self, task_name):
        self.task_name = task_name
        self.frames = []
        self.steps = []
        self.decisions = []
        self.reference_calls = []
        self.pending_decision = None
        self.bodies = []
        self.model_xml = None
        self.rendering = {"schema_version": 1, "task": task_name,
                          "coordinate_system": "Saved simulator world coordinates; playback does not integrate dynamics"}

    def start(self, task, raw):
        self.task = task
        env = task.dynamics.env.unwrapped
        if self.task_name in {"cartpole", "reacher"}:
            model = env.model
            self.model_xml = Path(env.fullpath).read_bytes()
            self.rendering.update(simulator="MuJoCo", timestep=float(env.dt),
                                  integration_timestep=float(model.opt.timestep), frame_skip=int(env.frame_skip),
                                  model_xml="rendering_assets/model.xml", model_binary="rendering_assets/model.mjb",
                                  nq=int(model.nq), nv=int(model.nv),
                                  geometry={"type": model.geom_type.copy(), "size": model.geom_size.copy(),
                                            "rgba": model.geom_rgba.copy(), "body_id": model.geom_bodyid.copy()},
                                  actuator_ctrlrange=model.actuator_ctrlrange.copy())
        else:
            self.bodies = [("lander", env.lander), ("leg_0", env.legs[0]), ("leg_1", env.legs[1]),
                           ("nozzle", env.nozzle), ("barge", env.barge)]
            self.rendering.update(
                simulator="Box2D", timestep=1 / float(env.cfg.fps),
                width=float(env.cfg.width), height=float(env.cfg.height),
                rocket_bodies=[body_geometry(name, body) for name, body in self.bodies],
                static_bodies=[body_geometry(f"static_{index}", body) for index, body in enumerate(env.world.bodies)
                               if body.type == 0 and body != env.barge],
                body_order=[name for name, _ in self.bodies],
                sky_polys=deepcopy(env.sky_polys), sea_polys=deepcopy(env.sea_polys), clouds=deepcopy(env.clouds))
        if hasattr(task, "targets"):
            self.rendering.update(targets=task.targets.copy(), target_tolerance=float(task.target_tolerance))
        if self.task_name == "rocket":
            mass, _ = env.get_mass_properties()
            limits = (env.cfg.main_engine_thrust_limits, env.cfg.side_engine_thrust_limits,
                      env.cfg.nozzle_angle_limits)
            self.reconstruction = {
                "landing_position": task._landing_position.copy(),
                "hover_input": [-env.cfg.gravity * mass / env.cfg.main_engine_thrust, 0.0, 0.0],
                "action_bounds": {"lower": [bound[0] for bound in limits], "upper": [bound[1] for bound in limits]},
            }
        else:
            self.reconstruction = {}
        self.reconstruction["control_timestep_s"] = self.rendering["timestep"]
        self.capture_frame(raw)

    def capture_frame(self, raw):
        env = self.task.dynamics.env.unwrapped
        frame = {"raw_observations": np.asarray(raw).copy(),
                 "observations": self.task.observation(raw).copy()}
        if self.task_name in {"cartpole", "reacher"}:
            frame.update(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), time=float(env.data.time),
                         geom_xpos=env.data.geom_xpos.copy(), geom_xmat=env.data.geom_xmat.copy(),
                         body_xpos=env.data.xpos.copy(), body_xquat=env.data.xquat.copy(),
                         ctrl=env.data.ctrl.copy(), actuator_force=env.data.actuator_force.copy())
        else:
            frame.update(time=len(self.steps) * self.rendering["timestep"],
                         body_position=np.asarray([list(body.position) for _, body in self.bodies]),
                         body_angle=np.asarray([body.angle for _, body in self.bodies]),
                         body_linear_velocity=np.asarray([list(body.linearVelocity) for _, body in self.bodies]),
                         body_angular_velocity=np.asarray([body.angularVelocity for _, body in self.bodies]),
                         body_awake=np.asarray([body.awake for _, body in self.bodies], dtype=bool),
                         leg_contacts=np.asarray([leg.ground_contact for leg in self.task.dynamics.env.unwrapped.legs], dtype=bool))
        self.frames.append(frame)

    def reference(self, value, step):
        self.reference_calls.append({"step": int(step), "reference": np.asarray(value).copy(),
                                     "target_index": getattr(self.task, "target_index", None)})

    def decision(self, policy, observation, reference, proposed, seconds, error=None):
        event = {"event": "decision", "step": len(self.steps), "decision_index": len(self.decisions),
                 "observation": np.asarray(observation).copy(), "reference": np.asarray(reference).copy(),
                 "seconds": float(seconds), "failed": error is not None, "error": str(error) if error else None,
                 "proposed_action": None if proposed is None else np.asarray(proposed).copy(),
                 "diagnostics": deepcopy(policy.diagnostics()) if hasattr(policy, "diagnostics") else {}}
        self.decisions.append(event)
        self.pending_decision = event

    def transition(self, sent, feedback, reference, transition, *, override=False):
        """Capture the transition the experiment has already executed."""
        sent = np.asarray(sent).copy()
        env = self.task.dynamics.env.unwrapped
        if self.task_name == "rocket":
            effective = np.array([
                np.clip(sent[0], *env.cfg.main_engine_thrust_limits) if sent[0] > 0 else 0.0,
                np.clip(sent[1], *env.cfg.side_engine_thrust_limits) if abs(sent[1]) > 0 else 0.0,
                np.clip(sent[2], *env.cfg.nozzle_angle_limits),
            ])
        else:
            effective = sent.copy()
            limited = np.asarray(env.model.actuator_ctrllimited, dtype=bool)
            bounds = np.asarray(env.model.actuator_ctrlrange)
            effective[limited] = np.clip(effective[limited], bounds[limited, 0], bounds[limited, 1])
        decision = None if override else self.pending_decision
        self.steps.append({
            "event": "step", "step": len(self.steps), "sent_action": sent,
            "effective_action": effective, "feedback_action": np.asarray(feedback).copy(),
            "reference": np.asarray(reference).copy(), "target_index": getattr(self.task, "target_index", None),
            "reward": float(transition.reward), "terminated": bool(transition.terminated),
            "truncated": bool(transition.truncated), "task_override": bool(override),
            "policy_decision": decision is not None,
            "decision_index": None if decision is None else decision["decision_index"],
            "proposed_action": None if decision is None else decision["proposed_action"],
            "decision_seconds": None if decision is None else decision["seconds"],
            "diagnostics": {} if decision is None else decision["diagnostics"],
        })
        self.pending_decision = None
        self.capture_frame(transition.observation)

    def references(self):
        return np.asarray([step["reference"] for step in self.steps]).reshape(-1, self.task.observation_dim)

    def finalize(self, directory):
        """Add replay data to an exported episode before its completion is published."""
        directory = Path(directory)
        if len(self.frames) != len(self.steps) + 1:
            raise ValueError("Every action must have exactly one following simulator frame")
        render = {key: np.asarray([frame[key] for frame in self.frames]) for key in self.frames[0]}
        if self.bodies:
            render["body_names"] = np.asarray([name for name, _ in self.bodies])
        _atomic_write(directory / "render_state.npz", lambda stream: np.savez_compressed(stream, **render))
        _write_json(directory / "rendering.json", self.rendering)
        if self.model_xml is not None:
            import mujoco
            assets = directory / "rendering_assets"
            assets.mkdir(exist_ok=True)
            (assets / "model.xml").write_bytes(self.model_xml)
            mujoco.mj_saveModel(self.task.dynamics.env.unwrapped.model, str(assets / "model.mjb"), None)
        _write_json(directory / "reference_calls.json", {"calls": self.reference_calls})
        from .runner import json_value, atomic_json_lines
        applied = {step["decision_index"] for step in self.steps if step["decision_index"] is not None}
        events = self.steps + [{**event, "unapplied": True} for event in self.decisions
                               if event["decision_index"] not in applied]
        atomic_json_lines(directory / "diagnostics.jsonl", [json_value(event) for event in events])
        with np.load(directory / "trajectory.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        shape = arrays["actions"].shape
        sent = np.asarray([step["sent_action"] for step in self.steps]).reshape(shape)
        np.testing.assert_array_equal(arrays["actions"], sent)
        np.testing.assert_array_equal(arrays["observations"], render["observations"])
        proposed = np.full(shape, np.nan)
        durations = np.full(len(self.steps), np.nan)
        for index, step in enumerate(self.steps):
            if step["policy_decision"]:
                proposed[index] = step["proposed_action"]
                durations[index] = step["decision_seconds"]
        arrays.update(
            raw_observations=render["raw_observations"], proposed_actions=proposed,
            sent_actions=sent.copy(),
            effective_actions=np.asarray([step["effective_action"] for step in self.steps]).reshape(shape),
            feedback_actions=np.asarray([step["feedback_action"] for step in self.steps]).reshape(shape),
            decision_durations=durations,
            policy_decision_mask=np.array([step["policy_decision"] for step in self.steps], dtype=bool),
            task_override_mask=np.array([step["task_override"] for step in self.steps], dtype=bool),
            rewards=np.array([step["reward"] for step in self.steps]),
            terminated=np.array([step["terminated"] for step in self.steps], dtype=bool),
            truncated=np.array([step["truncated"] for step in self.steps], dtype=bool),
            time=render["time"], timestamps=render["time"].copy(),
            cumulative_costs=np.r_[0.0, np.cumsum(arrays["state_costs"] + arrays["action_costs"])],
        )
        _atomic_write(directory / "trajectory.npz", lambda stream: np.savez_compressed(stream, **arrays))
