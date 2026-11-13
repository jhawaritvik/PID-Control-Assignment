from pathlib import Path

from pid_controller import PID_controller
from plant import SpringDamper
import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
duration = 30.0
target = 1.0
force_limit = 30.0
steps = round(duration / dt)

plant = SpringDamper()
controller = PID_controller(kp = 50.0, ki = 20.0, kd = 10.0, dt = dt, setpoint = target, force_limit = force_limit, anti_windup = True, derivative_filter_tau = 0.02, derivative_on_position = True)

rng = np.random.default_rng(42)
noise = rng.uniform(-0.001, 0.001, size=steps)

times = np.arange(steps + 1) * dt
positions = [plant.position]
forces = []

for step in range(steps):
    measured_position = plant.position + noise[step]
    applied_force = controller.update(measured_position)

    forces.append(applied_force)
    positions.append(plant.step(applied_force=applied_force, dt=dt))

positions = np.asarray(positions)
final_position = positions[times >= duration - 1.0].mean()
overshoot = 100 * max(0.0, positions.max() - target) / target

fig, (position_ax, force_ax) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

position_ax.plot(times, positions, label="Actual block position")
position_ax.axhline(target, color="black", linestyle="--", label="Target = 1 m")
position_ax.set_title("Spring–damper response with the final PID controller")
position_ax.set_ylabel("Position (m)")

force_ax.stairs(forces, times, baseline=None, label="Applied force")
force_ax.axhline(force_limit, color="black", linestyle="--", label="±30 N limits")
force_ax.axhline(-force_limit, color="black", linestyle="--")
force_ax.set_xlabel("Time (sec)")
force_ax.set_ylabel("Force (N)")

print(f"Mean position over the final second = {final_position:.6f} m")
print(f"Overshoot = {overshoot:.2f}%")
print(f"Peak force magnitude = {np.max(np.abs(forces)):.2f} N")

position_ax.legend()
force_ax.legend()
fig.tight_layout()
Path("results").mkdir(exist_ok=True)
fig.savefig("results/section9_final_run.png", dpi=300)
plt.show()