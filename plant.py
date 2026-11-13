class SpringDamper:
    def __init__(self, mass = 1.0, damping = 10.0, spring = 20.0):
        self.mass = mass
        self.damping = damping # coefficient applied due to damping
        self.spring = spring # coefficent of spring stiffness applied by the spring

        self.position = 0 # initial position
        self.velocity = 0 # initital velocity

    def step(self, applied_force, dt):
        spring_force = self.spring * self.position # spring force is applied relative to postion hence unit is N/m
        damping_force = self.damping * self.velocity # damping force is applied to the velocity hence unit is N.s/m

        total_force = applied_force - spring_force - damping_force # total force is applied force -  resisiting forces of spring and damping
        acceleration = total_force / self.mass # F=ma Newton's 2nd law

        self.velocity = self.velocity + acceleration * dt # new velocity after dt time has passed
        self.position = self.position + self.velocity * dt # new postion after moving at the new velocity for dt time

        return self.position

