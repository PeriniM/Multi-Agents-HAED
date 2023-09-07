import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter

class AgentEKF:
    def __init__(self, dim_x, dim_z, initial_state):
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # Initial state estimate (for example, [x, y, theta] = [0, 0, 0])
        self.ekf.x = np.array([0, 0, 0])
        
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
        self.ekf.x[2] = self.ekf.x[2] % (2 * np.pi)  # Normalize theta to [0, 2*pi]


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

# Simulate Robot Movement
N = 20
true_states = np.zeros((N, 3))
estimated_states = np.zeros((N, 3))
for i in range(N):
    # True motion (move linearly with a slight rotation)
    if i != 0:
        delta_s = 0.1  # linear movement
        delta_theta = 0.01  # angular movement
        theta = true_states[i-1, 2]
        true_states[i, 0] = true_states[i-1, 0] + delta_s * np.cos(theta)
        true_states[i, 1] = true_states[i-1, 1] + delta_s * np.sin(theta)
        true_states[i, 2] = theta + delta_theta
        true_states[i, 2] = true_states[i, 2] % (2 * np.pi)  # Keep theta within [0, 2*pi]
    else:
        true_states[i] = np.array([0, 0, 0])

    # Add Gaussian noise to generate measurements
    measurement = true_states[i] + np.random.normal(0, 0.1, 3)

    if i == 0:
        agent = AgentEKF(dim_x=3, dim_z=3, initial_state=measurement)
      
    # Control inputs: linear and angular velocities
    u = [0.1, 0.01]
    
    agent.predict(u)
    agent.update(measurement)
    estimated_states[i] = agent.ekf.x_post

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(true_states[:, 0], true_states[:, 1], 'g-', label="True Path")
plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'b-', label="EKF Estimates")
plt.legend()
plt.title("EKF Estimation for Differential Drive Robot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()
