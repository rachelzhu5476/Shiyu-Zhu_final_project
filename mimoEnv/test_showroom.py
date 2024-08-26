import gymnasium as gym
from stable_baselines3 import PPO

# 假设MIMoStandupEnv已经正确实现在mimoEnv.envs.mimo_env
from mimoEnv.envs.standup import MIMoStandupEnv

import gymnasium as gym
import time
import numpy as np
import mimoEnv
from stable_baselines3 import PPO  # 导入PPO模型

def main():
    """ Creates the environment and uses the trained model to take actions.
    The environment is rendered to an interactive window.
    """

    env = gym.make("MIMoStandup-v0", render_mode = 'human' )

    # 加载训练好的模型，确保替换为你的模型路径
    model_path = "/Users/rachelzhu/Desktop/MIMo/mimoEnv/models/standup/model_1.zip"
    model = PPO.load(model_path)

    max_steps = 500

    obs = env.reset()

    start = time.time()
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)  # 使用模型预测动作
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        if done or trunc:
            obs = env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()

if __name__ == "__main__":
    main()

