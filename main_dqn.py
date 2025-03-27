import gymnasium as gym
from RL_algorithm.DQN import DQN

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)  # DQN 에이전트 생성

    episodes = 70 # 학습할 에피소드 수
    batch_size = 64

    for episode in range(episodes):
        state = env.reset()[0] # Gymnasium의 reset()은 (state, info)를 반환
        total_reward = 0
        done = False

        while not done:
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            agent.update(batch_size)

            state = next_state
            total_reward += reward

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.eps:.3f}")

    env.close()

if __name__ == "__main__":
    main()
