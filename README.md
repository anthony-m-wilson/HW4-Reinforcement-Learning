# HW4-Reinforcement-Learning

Group Members:
- Michael Grajera
- Chase Mortensen
- Anthony Wilson

## Running the program

In order to run the program, install the following packages: `gym`, `gymnasium`, `pandas`, and `stable-baselines3`.

Then, run

```sh
python main.py
```

## Record the average reward as a baseline for later comparison.

```
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 23.2     |
|    ep_rew_mean     | 23.2     |
| time/              |          |
|    fps             | 1552     |
|    iterations      | 1        |
|    time_elapsed    | 1        |
|    total_timesteps | 2048     |
---------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 28.2          |
|    ep_rew_mean          | 28.2          |
| time/                   |               |
|    fps                  | 1039          |
|    iterations           | 2             |
|    time_elapsed         | 3             |
|    total_timesteps      | 4096          |
| train/                  |               |
|    approx_kl            | 0.008430149   |
|    clip_fraction        | 0.088         |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.687        |
|    explained_variance   | -0.0048627853 |
|    learning_rate        | 0.0003        |
|    loss                 | 7.79          |
|    n_updates            | 10            |
|    policy_gradient_loss | -0.0131       |
|    value_loss           | 57.5          |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 34.8         |
|    ep_rew_mean          | 34.8         |
| time/                   |              |
|    fps                  | 944          |
|    iterations           | 3            |
|    time_elapsed         | 6            |
|    total_timesteps      | 6144         |
| train/                  |              |
|    approx_kl            | 0.0096760895 |
|    clip_fraction        | 0.087        |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.668       |
|    explained_variance   | 0.11428839   |
|    learning_rate        | 0.0003       |
|    loss                 | 14.9         |
|    n_updates            | 20           |
|    policy_gradient_loss | -0.0207      |
|    value_loss           | 34.8         |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 46.2        |
|    ep_rew_mean          | 46.2        |
| time/                   |             |
|    fps                  | 915         |
|    iterations           | 4           |
|    time_elapsed         | 8           |
|    total_timesteps      | 8192        |
| train/                  |             |
|    approx_kl            | 0.008846871 |
|    clip_fraction        | 0.0831      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.641      |
|    explained_variance   | 0.21914816  |
|    learning_rate        | 0.0003      |
|    loss                 | 27          |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0198     |
|    value_loss           | 56.1        |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 62           |
|    ep_rew_mean          | 62           |
| time/                   |              |
|    fps                  | 896          |
|    iterations           | 5            |
|    time_elapsed         | 11           |
|    total_timesteps      | 10240        |
| train/                  |              |
|    approx_kl            | 0.0076143593 |
|    clip_fraction        | 0.0706       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.615       |
|    explained_variance   | 0.27840477   |
|    learning_rate        | 0.0003       |
|    loss                 | 19.3         |
|    n_updates            | 40           |
|    policy_gradient_loss | -0.0179      |
|    value_loss           | 63.3         |
------------------------------------------
Baseline Average Reward: 174.8
```

## Compare the baseline and modified agent’s performance. How did changing the reward function affect behavior?

Baseline Average Reward: 156.7
Average reward with custom reward function: 165.9

The average rewards have been pretty similar. Both seem to vary quite a bit each time we run the program. The custom reward function seems to end up with a higher reward more often than the baseline reward function, but it depends on the run and they are often pretty close.

Here's the full output from one of the modified agent's runs:

```
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 23.4     |
|    ep_rew_mean     | 21.6     |
| time/              |          |
|    fps             | 5078     |
|    iterations      | 1        |
|    time_elapsed    | 0        |
|    total_timesteps | 2048     |
---------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 28.1         |
|    ep_rew_mean          | 25.9         |
| time/                   |              |
|    fps                  | 3552         |
|    iterations           | 2            |
|    time_elapsed         | 1            |
|    total_timesteps      | 4096         |
| train/                  |              |
|    approx_kl            | 0.009991797  |
|    clip_fraction        | 0.122        |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.685       |
|    explained_variance   | -0.009401798 |
|    learning_rate        | 0.0003       |
|    loss                 | 5.63         |
|    n_updates            | 10           |
|    policy_gradient_loss | -0.02        |
|    value_loss           | 44.8         |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 35.4         |
|    ep_rew_mean          | 32.6         |
| time/                   |              |
|    fps                  | 3266         |
|    iterations           | 3            |
|    time_elapsed         | 1            |
|    total_timesteps      | 6144         |
| train/                  |              |
|    approx_kl            | 0.0096580675 |
|    clip_fraction        | 0.0647       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.659       |
|    explained_variance   | 0.085132     |
|    learning_rate        | 0.0003       |
|    loss                 | 12.8         |
|    n_updates            | 20           |
|    policy_gradient_loss | -0.0169      |
|    value_loss           | 32.6         |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 46.5        |
|    ep_rew_mean          | 42.9        |
| time/                   |             |
|    fps                  | 3141        |
|    iterations           | 4           |
|    time_elapsed         | 2           |
|    total_timesteps      | 8192        |
| train/                  |             |
|    approx_kl            | 0.008621851 |
|    clip_fraction        | 0.0807      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.632      |
|    explained_variance   | 0.20792335  |
|    learning_rate        | 0.0003      |
|    loss                 | 17.6        |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0153     |
|    value_loss           | 46.4        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 59.1        |
|    ep_rew_mean          | 54.3        |
| time/                   |             |
|    fps                  | 3068        |
|    iterations           | 5           |
|    time_elapsed         | 3           |
|    total_timesteps      | 10240       |
| train/                  |             |
|    approx_kl            | 0.006489607 |
|    clip_fraction        | 0.0608      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.611      |
|    explained_variance   | 0.35760826  |
|    learning_rate        | 0.0003      |
|    loss                 | 18.6        |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0147     |
|    value_loss           | 51.7        |
-----------------------------------------
Average reward with custom reward function: 173.52622567371105
```
## Compare the baseline and custom nueral network architecture's performance. How did changing the architecture affect behavior?

Average reward with custom nueral network architecture: 200.1

The modle with a custom nueral network had a higher average reward.

```
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 23.3     |
|    ep_rew_mean     | 23.3     |
| time/              |          |
|    fps             | 1554     |
|    iterations      | 1        |
|    time_elapsed    | 1        |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 27.2        |
|    ep_rew_mean          | 27.2        |
| time/                   |             |
|    fps                  | 1070        |
|    iterations           | 2           |
|    time_elapsed         | 3           |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.011969291 |
|    clip_fraction        | 0.102       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.686      |
|    explained_variance   | 0.0182      |
|    learning_rate        | 0.0003      |
|    loss                 | 8.16        |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0176     |
|    value_loss           | 51          |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 37          |
|    ep_rew_mean          | 37          |
| time/                   |             |
|    fps                  | 1076        |
|    iterations           | 3           |
|    time_elapsed         | 5           |
|    total_timesteps      | 6144        |
| train/                  |             |
|    approx_kl            | 0.009763462 |
|    clip_fraction        | 0.0549      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.663      |
|    explained_variance   | 0.0606      |
|    learning_rate        | 0.0003      |
|    loss                 | 12.5        |
|    n_updates            | 20          |
|    policy_gradient_loss | -0.0149     |
|    value_loss           | 36.2        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 51.8        |
|    ep_rew_mean          | 51.8        |
| time/                   |             |
|    fps                  | 1067        |
|    iterations           | 4           |
|    time_elapsed         | 7           |
|    total_timesteps      | 8192        |
| train/                  |             |
|    approx_kl            | 0.008619969 |
|    clip_fraction        | 0.0931      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.63       |
|    explained_variance   | 0.195       |
|    learning_rate        | 0.0003      |
|    loss                 | 21.8        |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0222     |
|    value_loss           | 57.3        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 67.2        |
|    ep_rew_mean          | 67.2        |
| time/                   |             |
|    fps                  | 1069        |
|    iterations           | 5           |
|    time_elapsed         | 9           |
|    total_timesteps      | 10240       |
| train/                  |             |
|    approx_kl            | 0.007016697 |
|    clip_fraction        | 0.0771      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.604      |
|    explained_variance   | 0.269       |
|    learning_rate        | 0.0003      |
|    loss                 | 17          |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0158     |
|    value_loss           | 55.7        |
-----------------------------------------
Average reward with custom network architecture: 200.1
```