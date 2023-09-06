import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class AgentEKF(ExtendedKalmanFilter):
    def __init__(self, encoder_noise, uwb_noise):
        super(AgentEKF, self).__init__(dim_x=4, dim_z=4)  

        # Initial state estimate
        self.x = np.array([0, 0, 0, 0])  

        # Initialize the state covariance
        self.P *= 1e-4

        # Process noise matrix
        # As you might not have exact noise for the process, we make a guess
        # You might need to adjust this based on experience
        self.Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3])

        # Measurement noise matrix
        self.R = np.diag([uwb_noise, uwb_noise, encoder_noise, encoder_noise])

        # State transition function (from your motion model)
        self.f = self.state_transition

    def state_transition(self, x, dt, u):
        """Predict next state from current state and control inputs.
        
        Args:
        - x: Current state [x, y, v, omega]
        - dt: Time step
        - u: Control inputs [v, omega]
        
        Returns:
        - Next state estimate
        """
        # Extract states
        x_pos, y_pos, v, omega = x
        v_control, omega_control = u

        # Predict using differential drive motion model
        x_pos_new = x_pos + v * np.cos(omega) * dt
        y_pos_new = y_pos + v * np.sin(omega) * dt
        v_new = v_control  # Assuming v remains constant
        omega_new = omega_control  # Assuming omega remains constant
        
        return np.array([x_pos_new, y_pos_new, v_new, omega_new])