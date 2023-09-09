import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class EKF:
    def __init__(self, dim_x, dim_z, initial_state):
        """
        Initializes an Extended Kalman Filter object.

        Args:
        - dim_x: int, the dimensionality of the state vector.
        - dim_z: int, the dimensionality of the measurement vector.
        - initial_state: numpy array of shape (dim_x,), the initial state of the system.

        Returns:
        None.
        """
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf.x = initial_state
        self.ekf.P = np.diag([0.1**2, 0.1**2, 0.2**2])
        self.ekf.R = np.diag([0.3**2, 0.3**2, 0.2**2])
        self.ekf.Q = np.eye(3) * 0.1

    def _state_transition(self, x, u):
        """
        Predicts the next state given the current state and control inputs.

        Args:
        - x: numpy array of shape (dim_x,), the current state of the system.
        - u: numpy array of shape (2,), the control inputs.

        Returns:
        - x_next: numpy array of shape (dim_x,), the predicted next state of the system.
        """
        delta_s = u[0]
        delta_theta = u[1]
        theta = x[2]
        x_next = x.copy()
        x_next[0] += delta_s * np.cos(theta + 0.5 * delta_theta)
        x_next[1] += delta_s * np.sin(theta + 0.5 * delta_theta)
        x_next[2] += delta_theta
        return x_next
    
    def _compute_jacobian(self, x, u):
        """
        Computes the Jacobian matrix A based on the state and control inputs.
        """
        delta_s = u[0]
        theta = x[2]
        A = np.array([
            [1, 0, -delta_s * np.sin(theta)],
            [0, 1, delta_s * np.cos(theta)],
            [0, 0, 1]
        ])
        return A
    
    def predict(self, u):
        """
        Predicts the next state of the system given the control inputs.

        Args:
        - u: numpy array of shape (2,), the control inputs.

        Returns:
        None.
        """
        self.ekf.x = self._state_transition(self.ekf.x, u)
        A = self._compute_jacobian(self.ekf.x, u)
        
        # Propagate the covariance
        self.ekf.P = A @ self.ekf.P @ A.T + self.ekf.Q

    def update(self, z):
        """
        Updates the state estimate given the measurement.

        Args:
        - z: numpy array of shape (dim_z,), the measurement.

        Returns:
        None.
        """
        def HJacobian(x):
            return np.eye(3)
            
        def Hx(x):
            return x

        self.ekf.update(z=z, HJacobian=HJacobian, Hx=Hx)