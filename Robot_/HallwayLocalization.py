import keyboard
import time
import matplotlib.pyplot as plt
import random

STATE_SPACE = list(range(31))  # Discrete state space [0, 30]
DOOR_POSITIONS = [5, 10, 25]  # Example positions of doors
belief = [1 / len(STATE_SPACE)] * len(STATE_SPACE)  # Uniform initial belief, belief = 상태 확률 분포
robot_position = 0  # Initial robot position

def move(position, action):
    if action == 'right': # keyboard.is_pressed(right)
        probabilities = [0.2, 0.6, 0.2]  # Stay, Move 1 step, Move 2 steps
        steps = [0, 1, 2]
    elif action == 'left':
        probabilities = [0.2, 0.6, 0.2]  # Stay, Move 1 step, Move 2 steps
        steps = [0, -1, -2]
    else:
        return position

    step = random.choices(steps, probabilities)[0] #steps list에서 prob의 확률로 random choice
    new_position = position + step
    return max(0, min(new_position, len(STATE_SPACE) - 1))  # Keep within bounds


def sensing(position):
    if position in DOOR_POSITIONS: # position == Door_POSITION
        measurement = random.choices([1, 0], [0.95, 0.05])[0] # 1 - 0.95의 확률로, 0- 0.05의 확률로 리턴
    elif any(abs(position - door) == 1 for door in DOOR_POSITIONS): #[5,10,25] 중에 현재 위치-문위치 = 1이 하나라도 있으면 true
        measurement = random.choices([1, 0], [0.8, 0.2])[0]
    elif any(abs(position - door) == 2 for door in DOOR_POSITIONS):
        measurement = random.choices([1, 0], [0.6, 0.4])[0]
    else:
        measurement = random.choices([1, 0], [0.1, 0.9])[0]
    return measurement


def predict(prev_belief, action): # prev_belief: 이전 상태에 대한 확률 분포 list, x: index, y: 해당 index에 있을 확률 / action: 'right' or 'left'
    pred_belief = [0] * len(STATE_SPACE) # [0 0 0 0 ... 0]
    for i in range(len(STATE_SPACE)):
        if action == 'right':
            pred_belief[i] += 0.6 * prev_belief[i - 1] if i - 1 >= 0 else 0
            pred_belief[i] += 0.2 * prev_belief[i - 2] if i - 2 >= 0 else 0
            pred_belief[i] += 0.2 * prev_belief[i]
        elif action == 'left':
            pred_belief[i] += 0.6 * prev_belief[i + 1] if i + 1 < len(STATE_SPACE) else 0
            pred_belief[i] += 0.2 * prev_belief[i + 2] if i + 2 < len(STATE_SPACE) else 0
            pred_belief[i] += 0.2 * prev_belief[i]
    return pred_belief


def update(pred_belief, measurement):
    likelihood = []
    for i in range(len(STATE_SPACE)):
        if i in DOOR_POSITIONS:
            likelihood.append(0.95 if measurement == 1 else 0.05)
        elif any(abs(i - door) == 1 for door in DOOR_POSITIONS):
            likelihood.append(0.8 if measurement == 1 else 0.2)
        elif any(abs(i - door) == 2 for door in DOOR_POSITIONS):
            likelihood.append(0.6 if measurement == 1 else 0.4)
        else:
            likelihood.append(0.1 if measurement == 1 else 0.9)

    updated_belief = [pred_belief[i] * likelihood[i] for i in range(len(STATE_SPACE))]
    normalization_factor = sum(updated_belief)
    updated_belief = [b / normalization_factor for b in updated_belief]
    return updated_belief, likelihood
    
def plot_all(prev_belief, pred_belief, likelihood, updated_belief, action, robot_pos):
    plt.figure(figsize=(14, 8))
    # Belief plots
    plt.subplot(3, 2, 1)
    plt.bar(STATE_SPACE, prev_belief)
    plt.title("Previous Belief")
    plt.ylim(0, 1)

    plt.subplot(3, 2, 2)
    plt.bar(STATE_SPACE, pred_belief)
    plt.title(f"Predicted Belief after action: {action}")
    plt.ylim(0, 1)

    plt.subplot(3, 2, 3)
    plt.bar(STATE_SPACE, likelihood)
    plt.title(f"Likelihood (Sensor Model)\nTrue position: {robot_pos}")
    plt.ylim(0, 1)

    plt.subplot(3, 2, 4)
    plt.bar(STATE_SPACE, updated_belief)
    plt.title("Updated Belief (Posterior)")
    plt.ylim(0, 1)

    # Environment view
    plt.subplot(3, 1, 3)
    for i in STATE_SPACE:
        if i in DOOR_POSITIONS:
            plt.plot(i, 0.5, 'bs', label='Door' if i == DOOR_POSITIONS[0] else"")
        else:
            plt.plot(i, 0.5, 'ko', markersize=4)

    # Robot position
    plt.plot(robot_pos, 0.5, 'ro', markersize=12, label='Robot')
    plt.title("Environment View")
    plt.yticks([])
    plt.xlim(-1, 31)
    plt.ylim(0.3, 0.7)
    plt.xlabel("Position")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


while True:
    if keyboard.is_pressed('right'):
        action = 'right'
        prev_belief = belief.copy()
        robot_position = move(robot_position, action)
        measurement = sensing(robot_position)
        pred_belief = predict(prev_belief, action)
        updated_belief, likelihood = update(pred_belief,measurement)
        plot_all(prev_belief, pred_belief, likelihood, updated_belief, action, robot_position)
        belief = updated_belief
        time.sleep(0.3)
    elif keyboard.is_pressed('left'):
        action = 'left'
        prev_belief = belief.copy()
        robot_position = move(robot_position, action)
        measurement = sensing(robot_position)
        pred_belief = predict(prev_belief, action)
        updated_belief, likelihood = update(pred_belief, measurement)
        plot_all(prev_belief, pred_belief, likelihood, updated_belief, action, robot_position)
        belief = updated_belief
        time.sleep(0.3)
    elif keyboard.is_pressed('esc'):
        print("Exiting simulation.")
        break
