# PID Control Assignment

Python simulations for the Modern MonkLabs PID control report. The PID logic is implemented in Python. NumPy handles sampled data and Matplotlib generates the figures.

## Reproduce the figures

Run these commands from the repository root with Python 3.10–3.13.

```sh
python -m pip install -r requirements.txt
python main.py
python analyse_results.py
python real_world.py
python tuning.py
python final_run.py
```

Run `main.py` before `analyse_results.py` because the latter reads its saved response data. Close any plot windows to continue to the next command. The scripts write figures and simulation data to `results/`. The two hand-drawn diagrams in Sections 1 and 3 are in `assets/`.

| Report section | Script | Figure files in results/ |
| --- | --- | --- |
| 4 | analyse_results.py | pi_demo_annotated.png |
| 5 | main.py | p_strengths.png, i_strength.png, pi_vs_pid.png |
| 6 | main.py | comparison.png |
| 7 | real_world.py | section7_windup.png, section7_sensor_noise.png, section7_derivative_kick.png, section7_slow_updates.png |
| 8 | tuning.py | section8_ultimate_gain.png, section8_zn_response.png, section8_pid_comparison.png |
| 9 | final_run.py | section9_final_run.png |

The analysis, real-world, tuning and final-run scripts print the measurements used in the report. The noisy-sensor experiments use a fixed NumPy random seed.

## Reusable code

- `pid_controller.py` implements the controller, optional force clipping, conditional integration and derivative filtering. It supports derivative on position and derivative on error.
- `plant.py` simulates the spring–damper block.
- `tuning.py` simulates the three-stage plant and estimates the ultimate gain and period with proportional control.
