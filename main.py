import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Set up the CartPole environment
env = gym.make("CartPole-v1")

# Define and train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Function to evaluate the agent
def evaluate_agent(env, model, num_episodes=10):
    all_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            # Only take the first four values from the step output
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if done or truncated:  # Ensure we stop the episode when either flag is True
                break
        all_rewards.append(episode_reward)
    
    average_reward = np.mean(all_rewards)
    return average_reward

# Evaluate the trained agent and record the average reward
average_reward = evaluate_agent(env, model, num_episodes=10)
print("Baseline Average Reward:", average_reward)

# Close the environment after evaluation
env.close()
