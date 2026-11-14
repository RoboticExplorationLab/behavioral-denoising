"""Replay saved Push-T actions at 100 Hz to draw the physical path.

No policy is called. Abort unless every 10 Hz state, velocity, contact count and
termination flag reproduces the original record exactly; polygon rewards may
differ by floating-point roundoff (absolute tolerance 1e-12).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    root = (arguments.repository or Path(__file__).resolve().parents[2]).resolve()
    record, output = arguments.record.resolve(), arguments.output.resolve()
    metadata = json.loads((record / "metadata.json").read_text())
    expected = json.loads((record / "record.json").read_text())["sha256"]
    for name in ("trajectory.npz", "metadata.json", "result.json"):
        if digest(record / name) != expected[name]:
            raise ValueError(f"Record hash mismatch: {name}")
    result = json.loads((record / "result.json").read_text())
    with np.load(record / "trajectory.npz", allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    vendored = root / "third_party/diffusion_policy"
    sys.path.insert(0, str(vendored))
    from diffusion_policy.env.pusht import pusht_env
    if not Path(pusht_env.__file__).resolve().is_relative_to(vendored):
        raise ImportError("Push-T must use the vendored physics model")
    source_paths = [record / name for name in ("trajectory.npz", "metadata.json", "result.json", "record.json")]
    source_paths += [Path(pusht_env.__file__), Path(pusht_env.__file__).with_name("pymunk_override.py")]
    input_hashes = {str(path): digest(path) for path in source_paths}
    env = pusht_env.PushTEnv(legacy=metadata["geometry"]["legacy"], render_size=512, render_action=False)

    def raw_state():
        return np.array([*env.agent.position, *env.block.position, env.block.angle])

    def velocities():
        return {"agent_velocity": np.asarray(env.agent.velocity),
                "block_velocity": np.asarray(env.block.velocity),
                "block_angular_velocity": env.block.angular_velocity,
                "agent_angle": env.agent.angle, "agent_angular_velocity": env.agent.angular_velocity}

    env.seed(metadata["environment_seed"])
    env.reset()
    if env.sim_hz != metadata["geometry"]["physics_hz"] or env.control_hz != metadata["geometry"]["control_hz"]:
        raise ValueError("Physics/control frequencies differ from the record")
    np.testing.assert_array_equal(raw_state(), saved["states"][0])
    for name, value in velocities().items():
        np.testing.assert_array_equal(value, saved[name][0], err_msg=f"initial {name}")
    samples = [raw_state()]
    physics_step = env.space.step

    def capture_physics_step(dt):
        # Record each physics step after integration.
        physics_step(dt)
        samples.append(raw_state())

    env.space.step = capture_physics_step
    maximum_reward_error = 0.
    try:
        for index, action in enumerate(saved["actions"]):
            _, reward, done, info = env.step(action)
            np.testing.assert_array_equal(raw_state(), saved["states"][index + 1], err_msg=f"state {index + 1}")
            for name, value in velocities().items():
                np.testing.assert_array_equal(value, saved[name][index + 1], err_msg=f"{name} {index + 1}")
            reward_error = abs(float(reward - saved["rewards"][index]))
            maximum_reward_error = max(maximum_reward_error, reward_error)
            if reward_error > 1e-12 or info["n_contacts"] != saved["n_contacts"][index + 1]:
                raise ValueError(f"Reward or contacts differ at step {index + 1}")
            if bool(done) != (bool(result["success"]) and index == len(saved["actions"]) - 1):
                raise ValueError(f"Termination differs at step {index + 1}")
    finally:
        env.close()
    samples = np.asarray(samples)
    ratio = env.sim_hz // env.control_hz
    if len(samples) != len(saved["actions"]) * ratio + 1:
        raise ValueError("Unexpected physics substep count")
    np.testing.assert_array_equal(samples[::ratio], saved["states"])
    if samples[::ratio].tobytes() != saved["states"].tobytes():
        raise ValueError("Control-boundary state bytes differ")
    times = np.arange(len(samples)) / env.sim_hz
    np.testing.assert_array_equal(times[::ratio], saved["time"])
    for path, expected_hash in input_hashes.items():
        if digest(Path(path)) != expected_hash:
            raise ValueError(f"Input changed during replay: {path}")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "trajectory.npz"
    np.savez_compressed(archive, states=samples, time=times)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "script_path": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__)),
              "record_directory": str(record), "seed": metadata["environment_seed"],
              "input_sha256": input_hashes, "physics_hz": env.sim_hz, "control_hz": env.control_hz,
              "recorded_samples": len(saved["states"]), "physics_samples": len(samples),
              "control_boundary_states_bitwise_equal": True,
              "velocities_contacts_and_termination_equal": True,
              "maximum_reward_error": maximum_reward_error, "reward_absolute_tolerance": 1e-12,
              "maximum_control_boundary_state_error": 0.,
              "controller_calls": 0, "replayed_control_steps": len(saved["actions"]),
              "replayed_physics_steps": len(samples) - 1, "smoothing": False, "interpolation": False,
              "software": {name: importlib.metadata.version(name)
                           for name in ("numpy", "pymunk", "gym", "Shapely")},
              "output_sha256": {archive.name: digest(archive)}}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("seed", "physics_samples", "maximum_control_boundary_state_error")}, indent=2))


if __name__ == "__main__":
    main()
