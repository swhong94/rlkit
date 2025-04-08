"""
Practice Problem: 2D Robot Localization with Kalman Filter

A robot moves in 2D space and receives noisy GPS measurements.
The robot's motion is modeled as a linear system with Gaussian noise. 
Your goal is to implement a Kalman Filter to estimate its current 2D position over time. 

Tasks
 Simulate 2D robot motion
 Simulate noisy measurements
 Implement a Kalman Filter that estimates the current position
 Plot: Ground truth trajectory, measurements, KF estimates"""

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt 

# Parameters
num_steps = 50
dt = 1.0
Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise covariance
R = np.array([[0.5, 0], [0, 0.5]])  # Measurement noise covariance

# Initial state
x_true = np.array([0, 0])  # True position
x_est = np.array([0, 0])   # Estimated position
P = np.eye(2)              # Initial covariance

# Storage for plotting
true_trajectory = []
measurements = []
estimates = []

def draw_covariance_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    ax = plt.gca()  # Get the current axis
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)  # Add the ellipse to the given axis

# Simulate motion and Kalman Filter
for t in range(num_steps):

    # Keyboard input
    key = input(f"\nStep {t + 1}/{num_steps} - Enter direction (w/a/s/d): ").lower()   

    # Simulate true motion
    # action = np.random.choice(['w', 'a', 's', 'd'])
    if key == 'w':
        a_t = np.array([0, 1])
    elif key == 'a':
        a_t = np.array([-1, 0])
    elif key == 's':
        a_t = np.array([0, -1])
    elif key == 'd':
        a_t = np.array([1, 0])
    else:
        a_t = np.array([0, 0])

    # dynamic model: x_t = x_t-1 + a_t + w_t
    w_t = np.random.multivariate_normal([0, 0], Q) # (mean, covariance)
    x_true = x_true + a_t + w_t
    true_trajectory.append(x_true)

    # Simulate noisy measurement
    # sensing model: z_t = x_t + v_t
    v_t = np.random.multivariate_normal([0, 0], R)
    z_t = x_true + v_t
    measurements.append(z_t)

    # Kalman Filter Prediction
    x_pred = x_est + a_t # state prediction
    P_pred = P + Q  # covariance prediction

    # Kalman Filter Update
    K = P_pred @ np.linalg.inv(P_pred + R)  # Kalman gain
    x_est = x_pred + K @ (z_t - x_pred)     # state update
    P = (np.eye(2) - K) @ P_pred            # covariance update
    estimates.append(x_est)

    # --- Plotting ---
    plt.clf()
    ax = plt.gca()
    ax.set_title(f"Step {t + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    true_arr = np.array(true_trajectory)
    est_arr = np.array(estimates)
    meas_arr = np.array(measurements)

    ax.plot(true_arr[:, 0], true_arr[:, 1], 'b-', label="True Trajectory")
    ax.plot(est_arr[:, 0], est_arr[:, 1], 'g--', label="KF Estimates")
    ax.plot(meas_arr[:, 0], meas_arr[:, 1], 'rx', label="Measurements")
    ax.plot(x_true[0], x_true[1], 'bo', label="Current True Position")
    ax.plot(x_est[0], x_est[1], 'go', label="Current Estimate")

    draw_covariance_ellipse(ax, x_est, P, alpha=0.3, color='green', label="Covariance")
    draw_covariance_ellipse(ax, x_pred, P_pred, alpha=0.2, color='blue', label="Covariance_pred")

    if t == 0:
        ax.legend()

    plt.pause(0.1)

