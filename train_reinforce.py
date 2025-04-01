import gymnasium as gym
from RL_algorithm.REINFORCE import REINFORCE
import matplotlib.pyplot as plt
import numpy as np

def train_reinforce():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0

    state_dim = env.observation_space.shape[0] #continuous state 
    action_dim = env.action_space.shape[0] #continuous action

    agent = REINFORCE(state_dim, action_dim)  

    episodes = 100  # 학습할 에피소드 수
    max_timestep = 500  # 최대 타임스텝 수 infinite: 999
    losses = []  # loss 저장
    moving_avg_losses =[]
    total_rewards = []  # score 저장

    for episode in range(episodes):
        state, _ = env.reset()  # Gymnasium의 reset()은 (state, info)를 반환
        log_probs = [] 
        rewards = [] 
        done = False

        for t in range(max_timestep):  # 한 에피소드 내에서 동작
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step([action])   # 연속 행동 공간 -> LIST로 넣어줌
            done = terminated or truncated

            log_probs.append(log_prob)  # log pi(a1|s1), log pi(a2|s2), ...
            rewards.append(reward)  # r1, r2, ...

            state = next_state  # state를 그대로 업데이트
            if done: break

        loss = agent.update(rewards, log_probs)
        total_reward = sum(rewards)

        total_rewards.append(total_reward)
        losses.append(loss)

        # 이동 평균 계산
        window_size = 10
        if len(losses) >= window_size:
            moving_avg_loss=np.mean(losses[-window_size:])      #-10~-1: 최근 10개 에피소드의 loss 평균 
        else:
            moving_avg_loss= np.mean(losses)
        moving_avg_losses.append(moving_avg_loss)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, loss: {moving_avg_loss:.4f}")

    env.close()

    plt.plot(total_rewards, label="total reward")
#    plt.plot(losses, label="loss")
    plt.plot(moving_avg_losses, label="moving average loss")
    plt.xlabel("Episode")
    plt.ylabel("total reward")
    plt.title("REINFORCE")
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    train_reinforce()
