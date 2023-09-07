import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class AgentEKF(ExtendedKalmanFilter):
    def __init__(self, encoder_noise, uwb_noise):
        super(AgentEKF, self).__init__(dim_x=5, dim_z=5)  # Adjusted for theta

        # Initial state estimate
        self.x = np.array([0, 0, 0, 0, 0])  # [x, y, theta, v, omega]

        # Initialize the state covariance
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-1, 1e-1])  # Some example values, adjust as needed

        # Process noise matrix
        self.Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3])

        # Measurement noise matrix
        self.R = np.diag([uwb_noise, uwb_noise, 1e-3, encoder_noise, encoder_noise])  # Added noise for theta estimation, adjust as needed

        # State transition function
        self.f = self.state_transition

    def state_transition(self, x, dt, u):
        """Predict next state from current state and control inputs."""
        # Extract states
        x_pos, y_pos, theta, v, omega = x
        v_control, omega_control = u

        # Predict using differential drive motion model
        x_pos_new = x_pos + v * np.cos(theta) * dt
        y_pos_new = y_pos + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        v_new = v_control
        omega_new = omega_control
        
        return np.array([x_pos_new, y_pos_new, theta_new, v_new, omega_new])
