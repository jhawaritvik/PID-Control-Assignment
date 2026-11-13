import numpy as np

class PID_controller:
    def __init__(self, kp, ki, kd, dt, setpoint, force_limit = None, anti_windup = False, derivative_filter_tau = 0, derivative_on_position = True):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.setpoint = setpoint
        self.force_limit = force_limit
        self.anti_windup = anti_windup

        self.integral_error = 0.0
        self.previous_position = None
        self.integral_force = 0.0

        self.derivative_force = 0.0

        self.filter_weight = 1.0

        if derivative_filter_tau > 0:
            self.filter_weight = 1 - np.exp(-dt / derivative_filter_tau) # 1 - e^(-dt/tau)

        self.derivative_on_position = derivative_on_position
        self.previous_error = None
        self.filtered_slope = 0.0 
    def update(self, current_position):
        error = self.setpoint - current_position
        p_force = self.kp * error

        if self.previous_position is None:
            slope = 0.0
        elif self.derivative_on_position:
            slope = (self.previous_position - current_position) / self.dt
        else:
            slope = (error - self.previous_error) / self.dt

        self.filtered_slope = (1 - self.filter_weight) * self.filtered_slope + self.filter_weight * slope # we take a weighted average of the previous slope with the new one
        d_force = self.kd * self.filtered_slope
        self.derivative_force = d_force

        self.previous_position = current_position
        self.previous_error = error


        integral_increment = error * self.dt
        # calculating a candidate for the force to be applied
        candidate_integral = self.integral_error + integral_increment
        candidate_force = p_force + self.ki * candidate_integral + d_force

        skip_integration = False
        if self.anti_windup and self.force_limit is not None:
            added_i_force = self.ki * integral_increment
            skip_integration = (candidate_force > self.force_limit and added_i_force > 0) or (candidate_force < -self.force_limit and added_i_force < 0) # True if candidate force is more than the force limit and integral component contributes to that

        if not skip_integration:
            self.integral_error = candidate_integral

        # We skip adding the integral increment to the integral error if skip_intergration condition is true 
        self.integral_force = self.ki * self.integral_error 
        desired_force = p_force + self.integral_force + d_force

        if self.force_limit is not None:
            return max(-self.force_limit, min(desired_force, self.force_limit))

        return desired_force
