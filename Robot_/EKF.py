# Unlinear System Model - EKF
# Unicycle Model

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt 

""" EKF Parameters for Unicycle Model
 State x = [x, y, theta]
 Control input u = [v, w] (linear velocity, angular velocity)
 x_t = x_t-1 + v * cos(theta) * dt + w * dt
 y_t = y_t-1 + v * sin(theta) * dt + w * dt
 theta_t = theta_t-1 + w * dt
"""

# Parameters
num_steps = 50
dt = 1.0
# Process noise covariance (Q)
Q = np.diag([0.1, 0.1, np.deg2rad(5)])  # [x_noise, y_noise, theta_noise]
# Measurement noise covariance (R)
R = np.diag([0.5**2, 0.5**2])  # 대칭 행렬로 수정

# Initial state
x_true = np.array([0, 0, 0])  # [x, y, theta]
x_est = np.array([0, 0, 0])   # [x, y, theta]
#P = np.diag([1, 1, np.deg2rad(10)])  # Initial covariance
P = np.eye(3)  # Initial covariance

# Storage for plotting
true_trajectory = []
measurements = []
estimates = []

# EKF Prediction Step
# x_pred = f(x, u, dt) + w_t
# P_pred = F * P * F^T + Q
def ekf_predict(x, P, u, Q, dt):
    theta = x[2] # x vector의 3번째 요소
    v, w = u
    # State transition model - Jacobian
    F = np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1,  v * np.cos(theta) * dt],
        [0, 0, 1]
    ])
    # Control input model
    B = np.array([
        [np.cos(theta) * dt, 0],
        [np.sin(theta) * dt, 0],
        [0, dt]
    ])
    # Predict state
    x_pred = x + B @ u
    # Predict covariance
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

# EKF Update Step
# z = h(x) + R
def ekf_update(x_pred, P_pred, z, R):
    # Observation model (identity matrix)
    H = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    # Innovation covariance
    S = H @ P_pred @ H.T + R
    # Kalman gain
    K = P_pred @ H.T @ np.linalg.inv(S)
    # Update state
    x_updated = x_pred + K @ (z - H @ x_pred)
    # Update covariance
    P_updated = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_updated, P_updated


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
        u = np.array([1.0, 0.0]) # move forward
    elif key == 'a':
        u = np.array([0.5, np.deg2rad(30)]) # turn left, 제자리
    elif key == 's':
        u = np.array([-1.0, 0.0]) # move backward
    elif key == 'd':
        u = np.array([0.5, -np.deg2rad(30)]) # turn right, 제자리
    else:
        u = np.array([0.0, 0.0])

    # dynamic model: x_t = x_t-1 + a_t + w_t
    w_t = np.random.multivariate_normal(np.zeros(3), Q) # (mean, covariance)
    theta = x_true[2]
    dx = np.array([
        np.cos(theta) * u[0] * dt,
        np.sin(theta) * u[0] * dt,
        u[1] * dt
    ])
    x_true = x_true + dx + w_t
    true_trajectory.append(x_true)

    # Simulate noisy measurement
    # sensing model: z_t = x_t + v_t
    v_t = np.random.multivariate_normal([0, 0], R)  # v_t 생성
    z_t = x_true[:2] + v_t
    measurements.append(z_t)

    # Kalman Filter Prediction
    x_pred, P_pred = ekf_predict(x_est, P, u, Q, dt) # state prediction

    # Kalman Filter Update
    x_est, P = ekf_update(x_pred, P_pred, z_t, R)     # state update
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

    draw_covariance_ellipse(ax, x_est[:2], P[:2,:2], alpha=0.3, color='green', label="Covariance")
    draw_covariance_ellipse(ax, x_pred[:2], P_pred[:2,:2], alpha=0.2, color='blue', label="Covariance_pred")
    
    if t == 0:
        ax.legend()

    plt.pause(0.1)
