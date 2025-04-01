import gymnasium as gym
from RL_algorithm.A2C import A2C
import matplotlib.pyplot as plt
import numpy as np

def train_a2c():
    env = gym.make("MountainCar-v0", render_mode="human", goal_velocity=0.1) 

    state_dim = env.observation_space.shape[0] #continuous state 
    action_dim = env.action_space.n 

    agent = A2C(state_dim, action_dim)  

    episodes = 100  # 학습할 에피소드 수
    a2c_loss = []  # loss 저장
    moving_avg_a2c_loss =[]
    total_rewards = []  # score 저장
    mini_batch_size = 100

    for episode in range(episodes): 
        state = env.reset()[0]   
        # log_probs = [] # 한 에피소드 내에서 probs, rewards를 저장해놓고, mini_batch size를 주기로 update
        # rewards = []
        done = False
        total_reward = 0
        # step_count = 0 # timestep count

        while not done: 
            action, log_prob, _ = agent.select_action(state) 
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated

            #log_probs.append(log_prob) #log pi(a1|s1), ..., log pi(an|sn)
            #rewards.append(reward) # r1, r2, ..., rn
            
            # Store transition in the buffer
            agent.buffer.push(state, action, log_prob, reward, next_state, done) 
            loss = agent.update(mini_batch_size)
            
            total_reward+=reward
            state = next_state 
            #step_count+=1

            a2c_loss.append(loss)

        total_rewards.append(total_reward)

        # Calculate moving average loss
        window_size = 10
        if len(a2c_loss) >= window_size:
            moving_avg_loss=np.mean(a2c_loss[-window_size:])      #-10~-1: 최근 10개 에피소드의 loss 평균 
        else:
            moving_avg_loss= np.mean(a2c_loss)
        moving_avg_a2c_loss.append(moving_avg_loss)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, loss: {a2c_loss[-1]:.4f}")

    env.close()

    plt.plot(total_rewards, label="total reward")
    plt.plot(moving_avg_a2c_loss, label="moving average loss")
    plt.xlabel("Episode")
    plt.ylabel("total reward")
    plt.title("A2C")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_a2c()
