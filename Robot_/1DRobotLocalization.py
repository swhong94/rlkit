# 1D Robot Localization using Particle Filter
# This code simulates a simple 1D robot localization problem using a particle filter.

import numpy as np
import matplotlib.pyplot as plt

NUM_PARTICLES = 500
WORLD_SIZE = 100
LANDMARKS = [20, 50, 80]
MOTION_NOISE = 1.0
SENSOR_NOISE = 2.0

particles = np.random.uniform(0, WORLD_SIZE, NUM_PARTICLES)
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES


def move_particles(particles, movement):
    return (particles + np.random.normal(movement, MOTION_NOISE, len(particles))) % WORLD_SIZE


def sense(robot_pos):
    return min([abs(robot_pos - landmark) for landmark in LANDMARKS]) + np.random.normal(0, SENSOR_NOISE)


def compute_weights(particles, measurement):
    distances = np.array([min([abs(particle - landmark) for landmark in LANDMARKS]) for particle in particles])
    return np.exp(-0.5 * ((distances - measurement) ** 2) / (SENSOR_NOISE ** 2))


def resample(particles, weights):
    weights /= np.sum(weights)
    indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
    return particles[indices]


def plot_results(trajectory, estimate_trajectory, all_particles, landmarks):
    for t, (true_pos, est_pos, particles) in enumerate(zip(trajectory, estimate_trajectory, all_particles)):
        plt.figure(figsize=(10, 6))
        plt.hist(particles, bins=50, density=False, alpha=0.6, color='green', label="Particle Distribution")
        for landmark in landmarks:
            plt.axvline(x=landmark, color="black", linestyle=":", label="Landmark" if landmark == landmarks[0] else "")
        plt.axvline(x=true_pos, color="blue", linestyle="--", label="True Position")
        plt.axvline(x=est_pos, color="red", linestyle="-", label="Estimated Position")
        plt.xlabel("Position")
        plt.ylabel("Probability Distribution")
        plt.title(f"Particle Distribution at Time Step {t}")
        plt.xlim(0, WORLD_SIZE)  
        plt.yticks(np.arange(0, plt.gca().get_ylim()[1] + 1, 20))  # y축 간격을 20 단위로 설정
        plt.legend()
        plt.show() 

true_position = 30
trajectory = []
estimate_trajectory = []
all_particles = []

for t in range(10):
    true_position = (true_position + 5) % WORLD_SIZE
    trajectory.append(true_position)

    z = sense(true_position)

    particles = move_particles(particles, movement=5)
    weights = compute_weights(particles, z)
    weights += 1e-300
    weights /= np.sum(weights)

    particles = resample(particles, weights)
    estimated_position = np.mean(particles)
    estimate_trajectory.append(estimated_position)
    all_particles.append(particles.copy())

    #print(f"Time step {t}: True Position: {true_position}, Estimated Position: {estimated_position}")

plot_results(trajectory, estimate_trajectory, all_particles, LANDMARKS)

