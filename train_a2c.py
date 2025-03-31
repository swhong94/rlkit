import gymnasium as gym
from RL_algorithm.A2C import A2C
import matplotlib.pyplot as plt
import numpy as np

def train_a2c():
    env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1) 

    state_dim = env.observation_space.shape[0] #continuous state 
    action_dim = env.action_space.n 

    agent = A2C(state_dim, action_dim)  

    episodes = 100  # 학습할 에피소드 수
    a2c_loss = []  # loss 저장
    moving_avg_a2c_loss =[]
    total_rewards = []  # score 저장
    batch_size = 64

    for episode in range(episodes):
        state = env.reset()[0]   
        done = False
        total_reward = 0

        for _ in range(batch_size): #버퍼 사이즈만큼 반복
            action, log_prob = agent.select_action(state) # action 선택
            next_state, reward, terminated, truncated, _ = env.step(action) #action으로 다음 state, reward, done을 받아옴 
            done = terminated or truncated

            agent.store_transition(state, action, log_prob, reward, next_state, done) # 버퍼에 transition 저장
            total_reward+=reward
            state = next_state 
            if done: 
                break
            # 여기까진 버퍼에 하나씩 차곡차곡 쌓는 과정

        loss = agent.update() # 버퍼에 저장된 transition을 이용해 agent 업데이트
        total_rewards.append(total_reward)
        a2c_loss.append(loss)

        # 이동 평균 계산
        window_size = 10
        if len(a2c_loss) >= window_size:
            moving_avg_loss=np.mean(a2c_loss[-window_size:])      #-10~-1: 최근 10개 에피소드의 loss 평균 
        else:
            moving_avg_loss= np.mean(a2c_loss)
        moving_avg_a2c_loss.append(moving_avg_loss)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, loss: {loss:.4f}")

    env.close()

    plt.plot(total_rewards, label="total reward")
#    plt.plot(losses, label="loss")
    plt.plot(moving_avg_a2c_loss, label="moving average loss")
    plt.xlabel("Episode")
    plt.ylabel("total reward")
    plt.title("A2C")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_a2c()
