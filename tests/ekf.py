import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter

class AgentEKF:
    def __init__(self, dim_x, dim_z, initial_state):
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # Initialize state estimate and covariance matrix
        self.ekf.x = initial_state
        self.ekf.P = np.eye(dim_x)  # For example, start with identity matrix
        self.ekf.Q = np.eye(dim_x)  # Process noise
        self.ekf.R = np.eye(dim_z)  # Measurement noise
        self.ekf.F = self.get_F()  # State transition matrix

    def get_F(self):
        # Define the state transition matrix based on agent dynamics
        return np.eye(self.ekf.x.shape[0])  # Just an identity matrix for now

    def predict(self):
        self.ekf.predict()

    def update(self, z):
        # Define the Jacobian of the measurement function
        def HJacobian_at(x):
            # This is just a dummy Jacobian, make sure to provide the correct one for your problem
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])

        # Define the measurement function
        def Hx_at(x):
            # Convert state x to a measurement
            # This is just an example, adjust according to your actual measurement model
            return x[:2]

        self.ekf.update(z=z, HJacobian=HJacobian_at, Hx=Hx_at)

# Example usage:

# Number of measurements
N = 50
# Define the true circle
radius = 5.0
theta_vals = np.linspace(0, 2*np.pi, N)
x_true = radius * np.cos(theta_vals)
y_true = radius * np.sin(theta_vals)

# Add Gaussian noise to generate measurements
std_dev = 0.2
x_noisy = x_true + np.random.normal(0, std_dev, N)
y_noisy = y_true + np.random.normal(0, std_dev, N)

# Use EKF to estimate the states given these measurements
agent = AgentEKF(dim_x=4, dim_z=2, initial_state=np.array([x_noisy[0], y_noisy[0], 0, 0]))
estimates = []

for i in range(N):
    agent.predict()
    measurement = np.array([x_noisy[i], y_noisy[i]])
    agent.update(measurement)
    estimates.append(agent.ekf.x_post[:2])

estimates = np.array(estimates)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_true, y_true, 'g-', label="True Circle")
plt.scatter(x_noisy, y_noisy, c='r', marker='o', label="Noisy Measurements")
plt.plot(estimates[:, 0], estimates[:, 1], 'b-', label="EKF Estimates")
plt.legend()
plt.title("EKF Estimation with Noisy Circular Measurements")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()