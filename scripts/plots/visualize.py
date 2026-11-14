"""Plot and animate a saved benchmark record without controllers or simulation.

Usage: python scripts/plots/visualize.py --record TASK/METHOD/seed_N --output OUT
Only saved arrays and model XML are read. MuJoCo and Box2D are not imported.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np


RENDERER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
METHOD_NAMES = {
    "bd_chi2": "BD-χ²", "bd_mean": "BD-mean", "bd_sample": "BD-sample",
    "bd_kl": "BD-KL", "bd_mean_chi2": "BD-mean-χ²", "select_dpc": "Select-DPC",
    "windowed_deepc": "Windowed DeePC", "deepc": "DeePC",
}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else (default or {})


def read_arrays(path):
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def load_record(directory):
    directory = Path(directory).resolve()
    result = read_json(directory / "result.json")
    metadata = read_json(directory / "metadata.json")
    arrays = read_arrays(directory / "trajectory.npz")
    render = read_arrays(directory / "render_state.npz")
    rendering = read_json(directory / "rendering.json")
    task = result.get("task", metadata.get("task"))
    if task not in {"cartpole", "reacher", "rocket"}:
        raise ValueError("Record must identify cartpole, reacher, or rocket")
    observations, actions = arrays["observations"], arrays["actions"]
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions) + 1:
        raise ValueError("Expected T+1 observations and T sent commands")
    references = arrays.get("references")
    if references is None or references.shape != observations[:-1].shape:
        raise ValueError("Expected one observation-sized reference per action")
    notes = []
    times = render.get("time", arrays.get("timestamps"))
    if times is None:
        dt = rendering.get("timestep")
        if dt is None and task == "cartpole":
            dt = .04
            notes.append("Legacy Cartpole time inferred from audited 0.02 s integration timestep and frame skip 2.")
        if dt is None:
            raise ValueError("No saved time axis or control timestep")
        times = np.arange(len(observations), dtype=float) * float(dt)
    times = np.asarray(times, dtype=float)
    if times.shape != (len(observations),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Time must contain one finite, strictly increasing timestamp per state")
    time_origin = float(times[0])
    times = times - time_origin
    if task == "cartpole" and "qpos" not in render:
        raw = arrays["states"]
        if raw.shape != (len(observations), 4):
            raise ValueError("Legacy Cartpole requires raw [x, theta, xdot, omega] states")
        render.update(qpos=raw[:, :2], qvel=raw[:, 2:])
        notes.append("Exact Cartpole qpos/qvel recovered from saved raw states.")
    if task in {"cartpole", "reacher"}:
        for key in ("qpos", "qvel"):
            if key in render and len(render[key]) != len(observations):
                raise ValueError(f"{key} length differs from observations")
    return dict(directory=directory, result=result, metadata=metadata, arrays=arrays,
                render=render, rendering=rendering, task=task, time=times,
                time_origin=time_origin, notes=notes)


def model_xml(record):
    candidate = record["rendering"].get("model_xml", "rendering_assets/model.xml")
    path = Path(candidate)
    if not path.is_absolute():
        path = record["directory"] / path
    if not path.is_file():
        return None
    expected_hash = record["rendering"].get("model_xml_sha256", record["rendering"].get("model_sha256"))
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("Archived model XML hash differs from rendering metadata")
    record["model_xml_used"] = {"path": str(path.resolve()), "sha256": actual_hash}
    return ET.parse(path).getroot()


def vector(attribute, default):
    return np.fromstring(attribute, sep=" ") if attribute is not None else np.array(default, dtype=float)


def geometry(record):
    xml = model_xml(record)
    if record["task"] == "cartpole":
        geom = xml.find(".//geom[@name='cpole']") if xml is not None else None
        if geom is not None and "fromto" in geom.attrib:
            endpoints = vector(geom.attrib["fromto"], []).reshape(2, 3)
        else:
            endpoints = np.array([[0, 0, 0], [.001, 0, .6]])
            record["notes"].append("Cartpole uses the audited schematic pole geometry; no original fromto XML was available.")
        return endpoints
    if record["task"] == "reacher":
        upper = xml.find(".//body[@name='body1']") if xml is not None else None
        tip = xml.find(".//body[@name='fingertip']") if xml is not None else None
        if upper is None or tip is None:
            record["notes"].append("Reacher uses standard model offsets: upper link 0.1 m, fingertip offset 0.11 m.")
        return (vector(upper.attrib.get("pos") if upper is not None else None, [.1, 0, 0]),
                vector(tip.attrib.get("pos") if tip is not None else None, [.11, 0, 0]))
    return None


def rotate2(vector, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]])


def reacher_points(qpos, offsets):
    elbow = rotate2(offsets[0][:2], qpos[0])
    fingertip = elbow + rotate2(offsets[1][:2], qpos[0] + qpos[1])
    return np.array([[0, 0], elbow, fingertip])


def reacher_world_points(record):
    """Use the archived body transforms when the recorder saved them."""
    positions = record["render"].get("body_xpos")
    if positions is None:
        return None
    xml = model_xml(record)
    world = xml.find("worldbody") if xml is not None else None
    if world is None:
        raise ValueError("Reacher world positions require archived XML body ordering")
    names = ["world"] + [body.attrib.get("name") for body in world.iter("body")]
    if positions.shape != (len(record["time"]), len(names), 3):
        raise ValueError("Reacher body positions do not align with archived XML bodies")
    indices = [names.index(name) for name in ["body0", "body1", "fingertip"]]
    points = positions[:, indices, :2]
    if not np.allclose(points[:, -1], record["arrays"]["observations"][:, 4:6], atol=1e-10, rtol=1e-10, equal_nan=True):
        raise ValueError("Saved Reacher fingertip world positions differ from task observations")
    return points


def record_title(record):
    result = record["result"]
    method = record["metadata"].get("public_method", result.get("method"))
    return f"{record['task'].title()} · {METHOD_NAMES.get(method, method)} · seed {result.get('seed')}"


def state_channels(record, wrap_cartpole=False):
    """Return saved physical coordinates and their recorded references."""
    arrays, task = record["arrays"], record["task"]
    observations, references = arrays["observations"], arrays["references"]
    if task == "cartpole":
        raw = arrays["states"]
        angle = np.arctan2(np.sin(raw[:, 1]), np.cos(raw[:, 1])) if wrap_cartpole else raw[:, 1]
        angle_label = "Wrapped pole angle (rad)" if wrap_cartpole else "Saved pole angle (unwrapped rad)"
        values = [raw[:, 0], angle, raw[:, 2], raw[:, 3]]
        labels = ["Cart position (m)", angle_label, "Cart velocity (m/s)", "Angular velocity (rad/s)"]
        targets = [references[:, 0], np.arctan2(references[:, 1], references[:, 2]), references[:, 3], references[:, 4]]
    elif task == "reacher":
        angles = record["render"].get("qpos")
        angles = angles[:, :2] if angles is not None else np.arctan2(observations[:, 2:4], observations[:, :2])
        values = [angles[:, 0], angles[:, 1], observations[:, 4], observations[:, 5], observations[:, 6], observations[:, 7]]
        labels = ["Joint 1 angle (rad)", "Joint 2 angle (rad)", "Fingertip x (m)", "Fingertip y (m)", "Joint 1 velocity (rad/s)", "Joint 2 velocity (rad/s)"]
        targets = [None, None, references[:, 4], references[:, 5], references[:, 6], references[:, 7]]
    else:
        values = [observations[:, index] for index in range(6)]
        targets = [references[:, index] for index in range(6)]
        labels = ["Rocket x (m)", "Rocket y (m)", "Horizontal velocity (m/s)", "Vertical velocity (m/s)", "Rocket angle (rad)", "Angular velocity (rad/s)"]
    return list(zip(values, targets, labels))


def draw_state(axis, times, values, reference, label):
    marker = "o" if len(times) == 1 else None
    axis.plot(times, values, color="#1769aa", linewidth=1.5, marker=marker, label="recorded state")
    if reference is not None and len(reference):
        reference_label = "reference (principal angle)" if "unwrapped" in label else "reference"
        axis.stairs(reference, times, baseline=None, color="#c55a11", linestyle="--", linewidth=1, label=reference_label)
    axis.set_ylabel(label.replace(" (", "\n("))
    axis.legend(fontsize=8, loc="best")


def finish_series(record, fig, axes, output, name, subtitle):
    times, result = record["time"], record["result"]
    for axis in np.asarray(axes).ravel():
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Time since episode start (s)")
        if times[-1] > 0:
            axis.set_xlim(0, times[-1])
        else:
            axis.set_xticks([0.])
    outcome = f"Success: {result.get('success')} · end: {result.get('terminal_reason')} · {len(record['arrays']['actions'])} actions"
    fig.suptitle(record_title(record) + "\n" + outcome, fontsize=13)
    fig.tight_layout(rect=(0, .045, 1, .91))
    if not len(record["arrays"]["actions"]):
        subtitle = "No controls executed; only the recorded initial state is available."
    fig.text(.5, .012, subtitle, ha="center", fontsize=9)
    paths = [output / f"{name}.png", output / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=170)
    plt.close(fig)
    return paths


def control_variants(record):
    """Keep sent commands distinct from saved actuator and feedback commands."""
    arrays = record["arrays"]
    sent = arrays["actions"]
    if "sent_actions" in arrays and not np.array_equal(sent, arrays["sent_actions"], equal_nan=True):
        raise ValueError("Saved sent_actions differ from the trajectory command trace")
    variants = [("sent_actions", "sent to simulator", sent, "#1769aa", "-")]
    for key, label, color, style in [
        ("effective_actions", "effective actuator command", "#2b8a3e", "--"),
        ("feedback_actions", "driver feedback command", "#c55a11", ":"),
    ]:
        if key not in arrays:
            continue
        values = arrays[key]
        if values.shape != sent.shape:
            raise ValueError(f"{key} does not align with the recorded commands")
        if not np.array_equal(values, sent, equal_nan=True):
            variants.append((key, label, values, color, style))
    return variants


def control_labels(task):
    if task == "cartpole":
        return ["Slider motor command (model control units)"]
    if task == "reacher":
        return ["Joint 1 motor command (model control units)", "Joint 2 motor command (model control units)"]
    return ["Main engine command (normalized units)", "Side engine command (normalized units)", "Nozzle command (normalized units)"]


def plot_states(record, output):
    channels = state_channels(record)
    fig, panels = plt.subplots(len(channels) // 2, 2, figsize=(12.5, 2.8 * (len(channels) // 2)), sharex=True)
    axes = panels.ravel()
    for axis, (values, reference, label) in zip(axes, channels):
        draw_state(axis, record["time"], values, reference, label)
    if record["task"] == "cartpole":
        angle = record["arrays"]["states"][:, 1]
        axes[1].plot(record["time"], np.arctan2(np.sin(angle), np.cos(angle)),
                     color="#7b5aa6", linestyle=":", linewidth=1, label="wrapped orientation")
        axes[1].legend(fontsize=8)
    if record["task"] == "reacher":
        axes[3].axhline(.1, color="#9e2828", linestyle=":", linewidth=1, label="height limit")
        axes[3].legend(fontsize=8)
    subtitle = "All saved states; step references act on the preceding state. No post-termination samples added."
    if record["task"] == "cartpole":
        subtitle = "Saved unwrapped pole angle and wrapped orientation are shown; the reference uses its principal branch."
    return finish_series(record, fig, axes, output, "state_vs_time", subtitle)


def plot_controls(record, output):
    labels = control_labels(record["task"])
    fig, panels = plt.subplots(len(labels), 1, figsize=(12.5, max(3.6, 2.7 * len(labels))), sharex=True, squeeze=False)
    axes = panels.ravel()
    variants = control_variants(record)
    for channel, (axis, label) in enumerate(zip(axes, labels)):
        if len(record["arrays"]["actions"]):
            for _, description, values, color, style in variants:
                axis.stairs(values[:, channel], record["time"], baseline=None, label=description, color=color, linestyle=style, linewidth=1.2)
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "No control executed", transform=axis.transAxes, ha="center", color="#9e2828")
        axis.set_ylabel(label.replace(" (", "\n("))
    subtitle = "Exact saved commands; effective/feedback overlays appear only when they differ from sent commands."
    return finish_series(record, fig, axes, output, "control_vs_time", subtitle)


def plot_series(record, output):
    arrays, times = record["arrays"], record["time"]
    channels = state_channels(record, wrap_cartpole=True)
    panel_count = len(channels) + 2
    fig, panels = plt.subplots(panel_count // 2, 2, figsize=(12.5, 2.6 * (panel_count // 2)), sharex=True)
    axes = panels.ravel()
    for axis, (values, reference, label) in zip(axes, channels):
        draw_state(axis, times, values, reference, label)
    if record["task"] == "reacher":
        axes[3].axhline(.1, color="#9e2828", linestyle=":", linewidth=1)
    input_axis = axes[-2]
    if len(arrays["actions"]):
        for index in range(arrays["actions"].shape[1]):
            input_axis.stairs(arrays["actions"][:, index], times, baseline=None, label=f"sent u{index + 1}", linewidth=1.2)
        input_axis.legend(fontsize=8)
    else:
        input_axis.text(.5, .5, "No control executed", transform=input_axis.transAxes, ha="center", color="#9e2828")
    input_axis.set_ylabel("Commands sent to simulator")
    state_costs, action_costs = arrays.get("state_costs"), arrays.get("action_costs")
    if state_costs is not None and action_costs is not None:
        total_label = "total raw cost" if record["task"] == "rocket" else "total trace cost"
        costs = [(state_costs, "state", "--"), (action_costs, "raw input", ":"), (state_costs + action_costs, total_label, "-")]
        if "hover_action_costs" in arrays:
            costs.append((state_costs + arrays["hover_action_costs"], "hover-relative cost", "-."))
        for values, label, style in costs:
            axes[-1].plot(times, np.r_[0., np.cumsum(values)], style, label=label, linewidth=1.5)
        axes[-1].legend(fontsize=8)
    else:
        axes[-1].text(.5, .5, "Per-step costs unavailable", transform=axes[-1].transAxes, ha="center")
    axes[-1].set_ylabel("Cumulative recorded cost")
    if record["result"].get("controller_failure"):
        axes[-1].set_title("Controller failure: reported cost is infinite; trace accrual shown", fontsize=9, color="#9e2828")
    return finish_series(record, fig, axes, output, "timeseries", "Saved trajectories only; commands/references/costs align with the pre-action state.")


def plot_targets(record, output):
    if record["task"] == "cartpole":
        return []
    arrays, times = record["arrays"], record["time"]
    xy = arrays["observations"][:, 4:6] if record["task"] == "reacher" else arrays["observations"][:, :2]
    reference_xy = arrays["references"][:, 4:6] if record["task"] == "reacher" else arrays["references"][:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(xy[:, 0], xy[:, 1], color="#1769aa", label="recorded path")
    axes[0].plot(xy[0, 0], xy[0, 1], "o", color="#1769aa", label="start")
    axes[0].plot(xy[-1, 0], xy[-1, 1], "s", color="#1769aa", label="last saved state")
    targets = arrays.get("targets", np.unique(reference_xy, axis=0))
    if len(targets):
        axes[0].scatter(targets[:, 0], targets[:, 1], marker="x", color="#c55a11", label="controller targets")
        for index, target in enumerate(targets):
            axes[0].annotate(str(index + 1), target, xytext=(4, 4), textcoords="offset points", fontsize=8)
    if record["task"] == "reacher":
        axes[0].axhline(.1, linestyle=":", color="#9e2828", label="height limit")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Recorded path and controller targets", aspect="equal")
    axes[0].legend(fontsize=8)
    if "target_indices" in arrays:
        axes[1].stairs(arrays["target_indices"] + 1, times, baseline=None, color="#c55a11")
        axes[1].set_ylabel("Active target number (cyclic)")
        axes[1].set_title(f"Final target completions: {record['result'].get('targets_completed', 'not recorded')}")
    else:
        for index, label in enumerate(["reference x", "reference y"]):
            axes[1].stairs(reference_xy[:, index], times, baseline=None, label=label)
        axes[1].legend(fontsize=8)
        axes[1].set_ylabel("Reference position (m)")
    axes[1].set_xlabel("Time since episode start (s)")
    for axis in axes:
        axis.grid(alpha=.18)
    fig.tight_layout()
    paths = [output / "targets.png", output / "targets.pdf"]
    for path in paths:
        fig.savefig(path, dpi=170)
    plt.close(fig)
    return paths


def fixture_patches(axis, fixtures, color):
    if not isinstance(color, str):
        color = np.asarray(color, dtype=float).copy()
        if np.max(color[:3]) > 1:
            color[:3] /= 255
        if len(color) == 4 and color[3] > 1:
            color[3] /= 255
    patches = []
    for fixture in fixtures:
        if fixture["type"] == "polygon":
            patch = Polygon(fixture["vertices"], closed=True, facecolor=color, edgecolor="#3a3a3a", linewidth=.7)
        elif fixture["type"] == "circle":
            patch = Circle(fixture.get("center", [0, 0]), fixture["radius"], facecolor=color, edgecolor="#3a3a3a", linewidth=.7)
        elif fixture["type"] in {"edge", "chain"}:
            patch = Polygon(fixture["vertices"], closed=False, fill=False, edgecolor=color, linewidth=1)
        else:
            continue
        axis.add_patch(patch)
        patches.append(patch)
    return patches


def motion_scene(record):
    task, saved, rendering = record["task"], record["render"], record["rendering"]
    fig, axis = plt.subplots(figsize=(8.5, 6.1))
    axis.set_aspect("equal")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("height (m)" if task != "reacher" else "y (m)")
    axis.grid(alpha=.12)
    axis.spines[["top", "right"]].set_visible(False)
    clock = axis.text(.5, 1.02, "", transform=axis.transAxes, ha="center", fontsize=11)
    if task == "cartpole":
        qpos = saved["qpos"]
        local_endpoints = geometry(record)
        limit = max(1.1, float(np.nanmax(np.abs(qpos[:, 0]))) + .8)
        axis.set(xlim=(-limit, limit), ylim=(-.8, .8))
        axis.axhline(0, color="#777777", linewidth=2)
        cart = Rectangle((0, -.06), .24, .12, color="#1769aa", zorder=3)
        axis.add_patch(cart)
        pole, = axis.plot([], [], color="#1769aa", linewidth=6, solid_capstyle="round")
        tip, = axis.plot([], [], "o", color="#1769aa", markersize=9)
        def draw(index):
            x, theta = qpos[index, :2]
            xs = x + local_endpoints[:, 0] * np.cos(theta) + local_endpoints[:, 2] * np.sin(theta)
            zs = -local_endpoints[:, 0] * np.sin(theta) + local_endpoints[:, 2] * np.cos(theta)
            cart.set_xy((x - .12, -.06)); pole.set_data(xs, zs); tip.set_data([xs[-1]], [zs[-1]])
        finite = np.isfinite(qpos[:, :2]).all(axis=1)
        fidelity = "Schematic Cartpole geometry from saved qpos; pole endpoints read from the archived model when available."
    elif task == "reacher":
        qpos = saved.get("qpos")
        if qpos is None:
            obs = record["arrays"]["observations"]
            qpos = np.arctan2(obs[:, 2:4], obs[:, :2])
            record["notes"].append("Reacher angles reconstructed modulo 2π from observations; original Gym target and full simulator state are unavailable.")
        offsets = geometry(record)
        world_points = reacher_world_points(record)
        axis.set(xlim=(-.26, .26), ylim=(-.24, .25))
        axis.axhspan(.1, .25, color="#b74e4e", alpha=.08)
        axis.axhline(.1, color="#9e2828", linestyle=":", linewidth=1)
        arm, = axis.plot([], [], "o-", color="#1769aa", linewidth=7, markersize=7, solid_capstyle="round")
        trail, = axis.plot([], [], color="#1769aa", linewidth=1, alpha=.3)
        target, = axis.plot([], [], "x", color="#c55a11", markersize=12, markeredgewidth=2, label="active controller target")
        targets = record["arrays"].get("targets", np.empty((0, 2)))
        if len(targets):
            axis.scatter(targets[:, 0], targets[:, 1], marker="x", color="#c55a11", alpha=.25)
        axis.legend(fontsize=9, loc="lower right")
        def draw(index):
            points = world_points[index] if world_points is not None else reacher_points(qpos[index], offsets)
            arm.set_data(points[:, 0], points[:, 1])
            obs = record["arrays"]["observations"][:index + 1]
            trail.set_data(obs[:, 4], obs[:, 5])
            if len(record["arrays"]["references"]):
                reference = record["arrays"]["references"][min(index, len(record["arrays"]["references"]) - 1), 4:6]
                if index == len(qpos) - 1 and "final_target_index" in record["arrays"] and len(targets):
                    reference = targets[int(record["arrays"]["final_target_index"])]
                target.set_data([reference[0]], [reference[1]])
            elif len(targets) and "final_target_index" in record["arrays"]:
                reference = targets[int(record["arrays"]["final_target_index"])]
                target.set_data([reference[0]], [reference[1]])
        if world_points is not None:
            finite = np.isfinite(world_points).all(axis=(1, 2))
            fidelity = "Schematic arm centerlines through archived MuJoCo body world positions; the fingertip matches the recorded task observation. Logical controller targets are overlaid."
        else:
            finite = np.isfinite(qpos[:, :2]).all(axis=1)
            fidelity = "Schematic arm centerlines reconstructed from recorded joint poses and model body offsets; world transforms were not saved. Logical controller targets are overlaid."
    else:
        required = {"body_names", "body_position", "body_angle"}
        if not required.issubset(saved) or "rocket_bodies" not in rendering:
            plt.close(fig)
            raise ValueError("Rocket motion requires saved body transforms and fixture geometry; no missing geometry is fabricated")
        names = [str(name) for name in saved["body_names"]]
        positions, angles = saved["body_position"], saved["body_angle"]
        axis.set_facecolor("#eef5fa")
        if len(positions) != len(record["time"]) or positions.shape[1] != len(names):
            raise ValueError("Rocket body transforms do not align with timestamps/body names")
        body_artists = []
        for body in rendering["rocket_bodies"]:
            index = names.index(body["name"])
            color = body.get("color1", "#1769aa")
            body_artists.append((index, fixture_patches(axis, body.get("fixtures", []), color)))
        for body in rendering.get("static_bodies", []):
            transform = Affine2D().rotate(float(body.get("angle", 0))).translate(*body.get("position", [0, 0])) + axis.transData
            for patch in fixture_patches(axis, body.get("fixtures", []), body.get("color1", "#aaaaaa")):
                patch.set_transform(transform)
        minimum, maximum = np.nanmin(positions, axis=(0, 1)), np.nanmax(positions, axis=(0, 1))
        axis.set(xlim=(minimum[0] - 3, maximum[0] + 3), ylim=(minimum[1] - 3, maximum[1] + 3))
        reference_marker, = axis.plot([], [], "x", color="#c55a11", markersize=10, markeredgewidth=2, label="controller reference")
        axis.legend(fontsize=9, loc="best")
        def draw(index):
            for body_index, patches in body_artists:
                transform = Affine2D().rotate(angles[index, body_index]).translate(*positions[index, body_index]) + axis.transData
                for patch in patches:
                    patch.set_transform(transform)
            if len(record["arrays"]["references"]):
                reference = record["arrays"]["references"][min(index, len(record["arrays"]["references"]) - 1)]
                reference_marker.set_data([reference[0]], [reference[1]])
        finite = np.isfinite(positions).all(axis=(1, 2)) & np.isfinite(angles).all(axis=1)
        fidelity = "Saved Box2D body transforms applied to archived fixture polygons/circles; schematic colors and camera, not the original renderer's pixels."
    valid = np.flatnonzero(finite)
    if not len(valid):
        plt.close(fig)
        raise ValueError("No finite recorded pose can be displayed")
    if not finite[0]:
        plt.close(fig)
        raise ValueError("Initial recorded pose is nonfinite; no earlier or replacement pose is fabricated")
    result = record["result"]
    fig.suptitle(record_title(record), fontsize=14)
    fig.text(.5, .045, f"Outcome: {result.get('success')} · {result.get('terminal_reason')} · {len(record['arrays']['actions'])} executed actions", ha="center", fontsize=10)
    fig.text(.5, .015, "Offline saved-pose replay; no controller calls or dynamics integration.", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.12, right=.97, bottom=.16, top=.86)
    def update(index):
        shown = valid[np.searchsorted(valid, index, side="right") - 1] if index >= valid[0] else valid[0]
        draw(shown)
        suffix = " · LAST FINITE POSE HELD; subsequent saved state is nonfinite" if shown != index else ""
        clock.set_text(f"Recorded t = {record['time'][shown]:.3f} s · state {shown}{suffix}")
    return fig, update, fidelity


def make_motion(record, output, fps):
    fig, update, fidelity = motion_scene(record)
    times = record["time"]
    update(len(times) - 1)
    poster = output / "motion.png"
    fig.savefig(poster, dpi=170)
    paths, frame_indices = [poster], []
    if len(times) > 1:
        native_rate = 1 / float(np.median(np.diff(times)))
        fps = min(float(fps), native_rate)
        presentation_times = np.arange(0, times[-1] + 1e-10, 1 / fps)
        frame_indices = np.searchsorted(times, presentation_times, side="right") - 1
        if frame_indices[-1] != len(times) - 1:
            frame_indices = np.r_[frame_indices, len(times) - 1]
        movie = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=False)
        if animation.writers.is_available("ffmpeg"):
            path = output / "motion.mp4"
            writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1600,
                                            extra_args=["-pix_fmt", "yuv420p", "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1"])
        else:
            path = output / "motion.gif"
            writer = animation.PillowWriter(fps=fps)
        movie.save(path, writer=writer, dpi=100)
        paths.append(path)
    else:
        record["notes"].append("Zero executed transitions: a static pose is saved; no motion video is fabricated.")
    plt.close(fig)
    return paths, fidelity, np.asarray(frame_indices, dtype=int).tolist()


def input_hashes(directory):
    names = ["trajectory.npz", "result.json", "metadata.json", "render_state.npz", "rendering.json"]
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names if (directory / name).is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plots-only", action="store_true")
    mode.add_argument("--motion-only", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs_before = input_hashes(args.record)
    record = load_record(args.record)
    options = {"fps": float(args.fps), "plot_dpi": 170, "video_dpi": 100}
    previous = read_json(args.output / "visualization.json")
    if (previous.get("inputs") != inputs_before or previous.get("renderer_sha256") != RENDERER_SHA256
            or previous.get("render_options") != options):
        previous = {}
    old_outputs = previous.get("outputs", [])
    if any(not Path(item["path"]).is_file() or hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]
           for item in old_outputs):
        previous, old_outputs = {}, []
    outputs = [Path(item["path"]) for item in old_outputs]
    components = set(previous.get("completed_components", []))
    fidelity = previous.get("motion_fidelity", "Motion not requested in this invocation")
    motion_status = previous.get("motion_status", "not_requested")
    frames = previous.get("video_saved_state_indices", [])
    if not args.motion_only:
        outputs += plot_states(record, args.output) + plot_controls(record, args.output)
        outputs += plot_series(record, args.output) + plot_targets(record, args.output)
        components.add("plots")
    if not args.plots_only:
        try:
            motion, fidelity, frames = make_motion(record, args.output, args.fps)
            outputs += motion
            motion_status = "video" if len(record["time"]) > 1 else "static_zero_actions"
            components.add("motion")
        except ValueError as error:
            fidelity, frames, motion_status = f"Motion unavailable: {error}", [], "unavailable"
            record["notes"].append(fidelity)
    if input_hashes(args.record) != inputs_before:
        raise RuntimeError("Input record changed during visualization")
    outputs = list(dict.fromkeys(outputs))
    manifest = {
        "record": str(record["directory"]), "task": record["task"],
        "seed": record["result"].get("seed"), "simulation_rerun": False,
        "renderer_sha256": RENDERER_SHA256, "render_options": options,
        "completed_components": sorted(components), "motion_status": motion_status,
        "motion_fidelity": fidelity, "notes": sorted(set(previous.get("notes", []) + record["notes"])),
        "time_origin": record["time_origin"], "duration_seconds": float(record["time"][-1]),
        "model_xml_used": record.get("model_xml_used", previous.get("model_xml_used")),
        "plotting_versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "video_saved_state_indices": frames,
        "state_channels": [label for _, _, label in state_channels(record)],
        "control_conventions": record["rendering"].get("action_conventions", {"sent_actions": "Recorded commands passed to the simulator; effective/feedback variants were not separately saved."}),
        "plotted_control_variants": [key for key, _, _, _, _ in control_variants(record)],
        "cost_scope": "Raw trace cost uses saved pre-action observations and commands sent to the simulator. Rocket's saved hover-relative cost is shown separately when available. Infinite failure penalties remain explicit in result.json.",
        "inputs": inputs_before,
        "outputs": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in outputs],
    }
    temporary = args.output / ".visualization.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output / "visualization.json")
    print(json.dumps({"outputs": [str(path) for path in outputs], "fidelity": fidelity, "notes": record["notes"]}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
