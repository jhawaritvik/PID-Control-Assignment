import os

import matplotlib.pyplot as plt
import numpy as np

from pid_controller import PID_controller
from plant import SpringDamper


os.makedirs("results", exist_ok=True)

plant_dt = 0.001
setpoint = 1.0

experiments = [
    {
        "name": "windup",
        "title": "Integral windup with a 30 N actuator limit",
        "duration": 5.0,
        "ki": 100.0,
        "force_limit": 30.0,
        "force_to_plot": "integral",
        "runs": [
            ("Clipping only", {}),
            ("Clipping + anti-windup", {"anti_windup": True}),
        ],
    },
    {
        "name": "sensor_noise",
        "title": "Derivative response to ±1 mm sensor noise",
        "duration": 4.0,
        "noise_amplitude": 0.001,
        "force_to_plot": "derivative",
        "runs": [
            ("Unfiltered derivative", {}),
            ("Filtered derivative", {"derivative_filter_tau": 0.02}),
        ],
    },
    {
        "name": "derivative_kick",
        "title": "Derivative kick after a setpoint change",
        "duration": 2.0,
        "setpoint_step_time": 0.5,
        "force_to_plot": "derivative",
        "runs": [
            ("Derivative on error", {"derivative_on_position": False}),
            ("Derivative on position", {}),
        ],
    },
    {
        "name": "slow_updates",
        "title": "Effect of slow controller updates",
        "duration": 4.0,
        "force_to_plot": "applied",
        "runs": [
            ("Update every 0.001 s", {"controller_dt": 0.001}),
            ("Update every 0.5 s", {"controller_dt": 0.5}),
        ],
    },
]

force_labels = {
    "integral": "Integral force (N)",
    "derivative": "Derivative force (N)",
    "applied": "Applied force (N)",
}

for experiment in experiments:
    steps = round(experiment["duration"] / plant_dt)
    times = np.arange(steps + 1) * plant_dt

    step_time = experiment.get("setpoint_step_time", 0.0)
    targets = np.where(times >= step_time, setpoint, 0.0)

    # Generate noise once so both runs receive the same noise samples.
    noise_amplitude = experiment.get("noise_amplitude", 0.0)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-noise_amplitude, noise_amplitude, size=steps)

    fig, (position_ax, force_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6)
    )

    force_key = experiment["force_to_plot"]

    for label, options in experiment["runs"]:
        plant = SpringDamper()
        controller_dt = options.get("controller_dt", plant_dt)
        update_every = round(controller_dt / plant_dt)

        controller = PID_controller(
            kp=50.0,
            ki=experiment.get("ki", 20.0),
            kd=10.0,
            dt=controller_dt,
            setpoint=targets[0],
            force_limit=experiment.get("force_limit"),
            anti_windup=options.get("anti_windup", False),
            derivative_filter_tau=options.get("derivative_filter_tau", 0.0),
            derivative_on_position=options.get(
                "derivative_on_position", True
            ),
        )

        positions = [plant.position]
        force_samples = []
        applied_force = 0.0

        for step in range(steps):
            if step % update_every == 0:
                controller.setpoint = targets[step]
                measured_position = plant.position + noise[step]
                applied_force = controller.update(measured_position)

            if force_key == "integral":
                force_samples.append(controller.integral_force)
            elif force_key == "derivative":
                force_samples.append(controller.derivative_force)
            else:
                force_samples.append(applied_force)

            position = plant.step(applied_force=applied_force, dt=plant_dt)
            positions.append(position)

        position_ax.plot(times, positions, label=label)
        force_ax.step(
            times, [*force_samples, force_samples[-1]], where="post", label=label
        )

        if experiment["name"] == "windup":
            overshoot = 100 * max(0.0, max(positions) - setpoint) / setpoint
            print(f"{label}: overshoot = {overshoot:.2f}%")
        elif experiment["name"] == "sensor_noise":
            # Use the final second to focus on noise after the initial motion.
            final_samples = np.asarray(force_samples[-round(1 / plant_dt):])
            print(f"{label}: derivative force std = {final_samples.std():.2f} N")
        elif experiment["name"] == "derivative_kick":
            step_index = round(step_time / plant_dt)
            print(f"{label}: D force at setpoint step = {force_samples[step_index]:.2f} N")

    position_ax.step(
        times, targets, where="post", color="black", linestyle="--", label="Target"
    )
    position_ax.set_title(experiment["title"])
    position_ax.set_ylabel("Position (m)")
    force_ax.set_ylabel(force_labels[force_key])
    force_ax.set_xlabel("Time (s)")

    for ax in (position_ax, force_ax):
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(f"results/section7_{experiment['name']}.png", dpi=300)
    plt.close(fig)