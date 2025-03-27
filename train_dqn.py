import gymnasium as gym
from RL_algorithm.DQN import DQN
import matplotlib.pyplot as plt

def train_dqn():
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    hard_update_agent = DQN(state_dim, action_dim, target_net_hard_update = True)  # DQN 에이전트 생성
    soft_update_agent = DQN(state_dim, action_dim, target_net_hard_update = False)  # DQN 에이전트 생성

    episodes = 200 # 학습할 에피소드 수
    batch_size = 64
    hard_update_avg_reward = [] #score 저장 
    soft_update_avg_reward = [] 

    for agent, reward_list in zip([hard_update_agent, soft_update_agent], [hard_update_avg_reward, soft_update_avg_reward]):
        avg_reward=0
        for episode in range(episodes):
        
            state = env.reset()[0]
            total_reward = 0
            done = False

            while not done: # 한 에피소드 내에서 동작
                action = agent.sample_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.update(batch_size)

                state = next_state
                total_reward += reward  # total reward = timestep

            avg_reward=0.9*avg_reward+0.1*total_reward if avg_reward !=0 else total_reward  # q(s,a)는 평균 reward = 0.9*avg_reward+0.1*total_reward
            reward_list.append(avg_reward)
            print(f"Episode {episode+1}, timestep:{total_reward:.0f}, avg_reward: {avg_reward:.3f}, Epsilon: {agent.eps:.3f}")
            
    env.close()

    plt.plot(hard_update_avg_reward, label="Hard Update")
    plt.plot(soft_update_avg_reward, label="Soft Update")
    plt.xlabel("Episode")
    plt.ylabel("avg reward")
    plt.title("Hard Update vs Soft Update")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_dqn()
