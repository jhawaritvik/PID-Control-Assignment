from pathlib import Path

from plant import SpringDamper
from pid_controller import PID_controller
import matplotlib.pyplot as plt
import numpy as np

def simulate(kp, ki, kd, dt = 0.001, duration = 30, setpoint = 1.0):

    plant = SpringDamper()
    controller  = PID_controller(kp = kp, ki = ki, kd = kd, dt = dt, setpoint = setpoint, force_limit = None)

    steps = round(duration / dt)

    times = [0.0]
    positions = [plant.position]
    forces = []

    for step in range(steps):
        applied_force = controller.update(current_position = plant.position) # force applied by the controller based on the position
        position = plant.step(applied_force = applied_force, dt = dt) # new position based on the force applied

        forces.append(applied_force)
        positions.append(position)
        times.append((step + 1) * dt)

    return {"kp":kp, "ki":ki, "kd":kd, "dt":dt, "duration":duration, "setpoint":setpoint, "times":np.array(times), "positions":np.array(positions), "forces":np.array(forces)}


def graphs(results, experiments, title, filename, end_time=None):
    plt.figure()
    for experiment in experiments:
        result = results[experiment]
        plt.plot(
            result["times"],
            result["positions"],
            label=f"{experiment}: Kp={result['kp']}, Ki={result['ki']}, Kd={result['kd']}",
        )
    
    plt.axhline(1.0, color = "black", linestyle = "--", label = "Setpoint = 1 Metre")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Position (metres)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if end_time is not None:
        plt.xlim(0, end_time)
    plt.tight_layout()
    plt.savefig(f"results/{filename}.png",dpi = 300)


def main():

    Path("results").mkdir(exist_ok=True)

    # Here experiments is a dict with key as the name of the experiment and value as the tuple of kp,ki,kd
    experiments = {"P20":(20.0,0.0,0.0), "P50":(50.0,0.0,0.0), "P80":(80.0,0.0,0.0), "P200":(200.0,0.0,0.0),
    "PI":(50.0,20.0,0.0), "PD":(50.0,0.0,10.0), "PID":(50.0,20.0,10.0), "PI_demo": (50.0, 100.0, 0.0)}

    results = {}

    for experiment,gains in experiments.items():
        kp, ki, kd = gains
        result = simulate(kp, ki, kd)

        results[experiment] = result

        # Saving the results
        np.savez_compressed(f"results/{experiment}.npz", **result)

    graphs(results, ["P20", "P50", "P80", "P200"], "Proportional Gain effects", "p_strengths")
    graphs(results, ["P50", "PI"], "Integral Gain effects", "i_strength")
    # Use a higher P gain to make the initial bounce easier to see in Section 5.3.
    damping_results = {
        "PI": simulate(kp=100.0, ki=20.0, kd=0.0),
        "PID": simulate(kp=100.0, ki=20.0, kd=10.0),
    }
    graphs(damping_results, ["PI", "PID"], "Effect of adding derivative action", "pi_vs_pid", end_time=2.0)
    graphs(results, ["P50", "PI", "PD", "PID"], "Comparison of P, PI, PD and PID controllers", "comparison")

    plt.show()

if __name__ == "__main__":
    main()
    

