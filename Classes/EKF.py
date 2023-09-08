import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class EKF:
    def __init__(self, dim_x, dim_z, initial_state):
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf.x = initial_state
        self.ekf.P = np.diag([0.1**2, 0.1**2, 0.2**2])
        self.ekf.R = np.diag([0.3**2, 0.3**2, 0.2**2])
        self.ekf.Q = np.eye(3) * 0.1

    def _state_transition(self, x, u):
        """Predicts next state given state and control inputs."""
        delta_s = u[0]
        delta_theta = u[1]
        theta = x[2]
        x_next = x.copy()
        x_next[0] += delta_s * np.cos(theta + 0.5 * delta_theta)
        x_next[1] += delta_s * np.sin(theta + 0.5 * delta_theta)
        x_next[2] += delta_theta
        x_next[2] = x_next[2] % (2 * np.pi)  # Normalize theta to [0, 2*pi]
        return x_next

    def predict(self, u):
        self.ekf.x = self._state_transition(self.ekf.x, u)

    def update(self, z):
        def HJacobian(x):
            return np.eye(3)
            
        def Hx(x):
            return x  # Measurement function is identity

        self.ekf.update(z=z, HJacobian=HJacobian, Hx=Hx)