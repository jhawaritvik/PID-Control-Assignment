from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pid_controller import PID_controller

Path("results").mkdir(exist_ok=True)

dt = 0.001
duration = 30.0
steps = round(duration / dt)

times = np.arange(steps + 1) * dt
def simulate(kp, ki = 0.0, kd = 0.0): # Fn to simulate the plant 1/(s+1)^3
    stage_1 = 0.0
    stage_2 = 0.0
    output = 0.0  # Stage 3 is the plant output

    outputs = [output]

    target = 1.0

    controller = PID_controller(kp=kp, ki=ki, kd=kd, dt=dt, setpoint=target)

    for _ in range(steps):
        # Calculate all three new values using the values at the start of the step.
        command = controller.update(output)

        next_stage_1 = stage_1 + dt * (command - stage_1) # next stage is the current + how much increment will happen in that time step
        next_stage_2 = stage_2 + dt * (stage_1 - stage_2)
        next_output = output + dt * (stage_2 - output)

        stage_1 = next_stage_1
        stage_2 = next_stage_2
        output = next_output

        outputs.append(output)

    return np.array(outputs)


def find_peaks(outputs):
    left = outputs[:-2]
    middle = outputs[1:-1]
    right = outputs[2:]
    is_peak = (middle > left) & (middle >= right) # these being numpy arrays are compared one pair at a time
    return np.flatnonzero(is_peak) + 1 # gives indices where we have peaks

# Finding Ultimate Gain ku
lower_gain = 7.5  # Shrinking peaks
upper_gain = 8.5  # Growing peaks

while upper_gain - lower_gain > 0.0001:
    trial_gain = (lower_gain + upper_gain) / 2
    outputs = simulate(trial_gain)
    peaks = find_peaks(outputs)
    if len(peaks) < 5:
        raise ValueError("Simulation duration is too low, 5 peaks were not reached.")

    earlier_peak = outputs[peaks[-5]]
    latest_peak = outputs[peaks[-1]]

    if latest_peak > earlier_peak:
        upper_gain = trial_gain  # Peaks grew, so try lower gains.
    else:
        lower_gain = trial_gain  # Peaks shrank or were equal.

ku = (lower_gain + upper_gain) / 2

# Findin period at that gain
outputs = simulate(ku)
peaks = find_peaks(outputs)

peak_times = times[peaks[-5:]]
tu = np.diff(peak_times).mean() # Is the average time gap between the peaks. 5 peaks give 4 gaps, which are then averaged

print(f"Ultimate gain Ku = {ku:.4f}")
print(f"Ultimate period Tu = {tu:.3f} s")
print("Last five peak heights:", outputs[peaks[-5:]])

plt.plot(times, outputs, label=f"P only, Kp = {ku:.4f}")
plt.scatter(peak_times, outputs[peaks[-5:]], color="red", label="Last five peaks")
plt.title("Sustained oscillation at the ultimate gain")
plt.xlabel("Time (s)")
plt.ylabel("Output (normalized units)")
plt.legend()
plt.tight_layout()
plt.savefig("results/section8_ultimate_gain.png", dpi=300)


# Now that we have ultimate gain and period values we calculate kp, ki and kd using the recipe table
zn_kp = 0.6 * ku
zn_ki = 2 * zn_kp / tu
zn_kd = zn_kp * tu / 8

zn_response = simulate(zn_kp, zn_ki, zn_kd)

target = 1.0
peak = zn_response.max()
overshoot = 100 * max(0.0, peak - target) / target

print(f"Kp = {zn_kp:.4f}, Ki = {zn_ki:.4f}, Kd = {zn_kd:.4f}")
print(f"Peak output = {peak:.3f}")
print(f"Overshoot = {overshoot:.2f}%")

plt.figure()
plt.plot(times, zn_response, label="Ziegler–Nichols PID")
plt.axhline(target, color="black", linestyle="--", label="Target = 1")
plt.title("Step response with Ziegler–Nichols gains")
plt.xlabel("Time (s)")
plt.ylabel("Output (normalized units)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/section8_zn_response.png", dpi=300)

trial_response = simulate(4.0, 2.0, 2.5)

for label, response in [
    ("Ziegler–Nichols PID", zn_response),
    ("Adjusted PID", trial_response),
]:
    peak = response.max()
    overshoot = 100 * max(0.0, peak - target) / target

    outside_band = np.flatnonzero(np.abs(response - target) > 0.02 * target)
    settling_index = outside_band[-1] + 1 if outside_band.size else 0

    print(f"\n{label}")
    print(f"Peak output = {peak:.3f}")
    print(f"Overshoot = {overshoot:.2f}%")
    if settling_index < len(times):
        print(f"Settling time = {times[settling_index]:.3f} s")
    else:
        print("Settling time: did not settle during the run")

plt.figure()
plt.plot(times, zn_response, label="Ziegler–Nichols PID")
plt.plot(times, trial_response, label="Adjusted PID (Kp = 4, Ki = 2, Kd = 2.5)")
plt.axhline(target, color="black", linestyle="--", label="Target = 1")

plt.title("Ziegler–Nichols and adjusted PID step responses")
plt.xlabel("Time (s)")
plt.ylabel("Output (normalized units)")
plt.xlim(0, 15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/section8_pid_comparison.png", dpi=300)
plt.show()