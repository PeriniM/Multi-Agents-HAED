import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class AgentEKF:
    def __init__(self, dim_x, dim_z, initial_state):
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # Initial state estimate (for example, [x, y, theta] = [0, 0, 0])
        self.ekf.x = initial_state
        
        # Initial state covariance. Represent your initial uncertainty about position and orientation.
        self.ekf.P = np.diag([0.1**2, 0.1**2, 0.2**2]) 
        
        # Measurement noise for [position_x, position_y, theta].
        # Since the encoders give you \(\Delta s\) and \(\Delta \theta\), if you convert these to position estimates, 
        # then the measurement noise from the encoders will impact your position noise.
        # Here, I'm assuming the 0.3 for encoders directly translates to position deviations. 
        # Depending on your model, this might need adjustments.
        self.ekf.R = np.diag([0.3**2, 0.3**2, 0.2**2]) 

        # Process noise: This might need tuning. Here's an initial guess.
        self.ekf.Q = np.eye(3) * 0.1

    def predict(self, u):
        delta_s = u[0]
        delta_theta = u[1]
        theta = self.ekf.x[2]
        
        self.ekf.x[0] += delta_s * np.cos(theta + 0.5 * delta_theta)
        self.ekf.x[1] += delta_s * np.sin(theta + 0.5 * delta_theta)
        self.ekf.x[2] += delta_theta
        # Normalize theta to [-pi, pi]
        self.ekf.x[2] = (self.ekf.x[2] + np.pi) % (2 * np.pi) - np.pi

    def update(self, z):
        # Define the Jacobian of the measurement function
        def HJacobian_at(x):
            return np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        # Define the measurement function
        def Hx_at(x):
            return x  # just return the state as the measurement

        self.ekf.update(z=z, HJacobian=HJacobian_at, Hx=Hx_at)