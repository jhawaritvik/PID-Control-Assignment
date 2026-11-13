import numpy as np
import matplotlib.pyplot as plt

for experiment in ["P50", "PI", "PD", "PID", "PI_demo"]:
    with np.load(f"results/{experiment}.npz") as result:
        times = result["times"]
        positions = result["positions"]
        setpoint = result["setpoint"]

        # Calculating risetime
        # Finding indices where position reaches each threshold
        reached_10 = np.flatnonzero(positions >= 0.1 * setpoint) #stores indices where threshhold is true
        reached_90 = np.flatnonzero(positions >= 0.9 * setpoint)
        
        if reached_10.size > 0 and reached_90.size > 0:
            time_10 = times[reached_10[0]]
            time_90 = times[reached_90[0]]

            rise_time = time_90 - time_10
            print(f"{experiment}: rise time = {rise_time:.3f} s")
        else:
            rise_time = None
            print(f"{experiment}: rise time not reached")

        # Calculating overshoot
        peak_position = np.max(positions)
        overshoot = max(0.0, peak_position - setpoint)
        overshoot_percent = (overshoot / setpoint) * 100
        print(f"{experiment}: overshoot = {overshoot_percent:.2f}%")

        # Calculating settling time
        tolerance = 0.02 * setpoint
        outside_band = np.flatnonzero(np.abs(positions - setpoint) > tolerance) # stores indices where error is more than tolerance of +-2%
        if outside_band.size == 0: # error was always in the tolearance range
            settling_time = times[0]
        elif outside_band[-1] == len(times) - 1: # if the last sample was outside the band
            settling_time = None
        else:
            settling_time = times[outside_band[-1] + 1]
        
        if settling_time is None:
            print(f"{experiment}: did not settle within target range")
        else:
            print(f"{experiment}: settling time = {settling_time:.3f}s ")

        # Calculating steady state error
        last_second = positions[times >= times[-1] - 1.0] # postions in the last second of the run
        position_range = np.max(last_second) - np.min(last_second)

        if position_range <  0.0001: # if variation in the last second is less than 0.0001
            final_position = np.mean(last_second)
            steady_state_error = setpoint - final_position
            print(f"{experiment}: steady state error = {1000 * steady_state_error:.3f} mm")
        else:
            steady_state_error = None
            print(f"{experiment}: still not settled at the end of the run")     

        if experiment == "PI_demo":
            peak_index = np.argmax(positions)

            plt.figure(figsize=(9, 5))
            plt.plot(times, positions, label="PI response")
            plt.axhline(setpoint, color="black", linestyle="--",
                        label="Target: 1 m")

            # Rise time: shade the interval between the two crossings
            plt.axvspan(time_10, time_90, color="blue", alpha=0.12,
                        label=f"Rise time: {rise_time:.3f} s")

            # Overshoot: mark the highest recorded point
            plt.scatter(times[peak_index], peak_position, color="red",
                        label=f"Peak: {peak_position:.4f} m ({overshoot_percent:.2f}% overshoot)")

            # Settling time: show the band and when the response stays in it
            plt.axhspan(setpoint - tolerance, setpoint + tolerance,
                        color="green", alpha=0.12, label="Target band: ±2%")
            plt.axvline(settling_time, color="green", linestyle=":",
                        label=f"Settling time: {settling_time:.3f} s")

            plt.text(1.5, 0.15, "Estimated steady-state error ≈ 0 m\n"
                     "(from the final second of the 30 s run)")

            plt.xlim(0, 3)
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.title("Four performance measures of a PI step response")
            plt.legend(loc="center right")
            plt.tight_layout()
            plt.savefig("results/pi_demo_annotated.png", dpi=300)


plt.show()
