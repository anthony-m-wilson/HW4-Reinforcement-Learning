import gymnasium as gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import torch as th
from torch import nn


class CustomCartPoleReward(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomCartPoleReward, self).__init__(env)

    def reward(self, reward):
        # Increase reward for keeping the pole upright and penalize for moving away from the center
        x, x_dot, theta, theta_dot = self.env.state
        new_reward = reward - np.abs(theta)  # Penalize for angle from upright
        return new_reward


class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.custom_mlp = CustomMLP()
        self._features_dim = features_dim

    def forward(self, observations):
        return self.custom_mlp(observations)


# Set up the CartPole environment
env = gym.make("CartPole-v1")

# Initialize the custom environment
# render_mode="human"
# custom_env = CustomCartPoleReward(gym.make("CartPole-v1"))

# Set up CartPole as a vectorized environment
vec_env = make_vec_env("CartPole-v1")

# Define and train the PPO model
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)

# Define and train the custom PPO model
# custom_model = PPO("MlpPolicy", custom_env, verbose=1)
# custom_model.learn(total_timesteps=10000)

# Modify the policy architecture
custom_model_architecture = PPO(
    "MlpPolicy",
    vec_env,
    policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[128, 128]),
    verbose=1,
)
custom_model_architecture.learn(total_timesteps=10000)


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
# average_reward = evaluate_agent(env, model, num_episodes=10)
# print("Baseline Average Reward:", average_reward)

# avg_custom_reward = evaluate_agent(custom_env, custom_model, num_episodes=10)
# print(f"Average reward with custom reward function: {avg_custom_reward}")

# Evaluate the custom model with the custom network architecture
avg_custom_network = evaluate_agent(env, custom_model_architecture, num_episodes=10)
print(f"Average reward with custom network architecture: {avg_custom_network}")

# Close the environment after evaluation
env.close()
