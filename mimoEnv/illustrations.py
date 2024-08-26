import os
import gymnasium as gym
import time
import argparse
import cv2
import csv
import numpy as np

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

class SaveTrainingDataCallback(BaseCallback):
    def __init__(self, save_every: int, log_dir: str, model_save_dir: str, verbose=1):
        super(SaveTrainingDataCallback, self).__init__(verbose)
        self.save_every = save_every
        self.log_dir = log_dir
        self.model_save_dir = model_save_dir
        self.counter = 0
        self.data = []

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        ep_info = [info for info in self.locals['infos'] if 'episode' in info]
        if len(ep_info) > 0:
            ep_rew_mean = np.mean([info['episode']['r'] for info in ep_info])
            timesteps = self.num_timesteps
            self.data.append([timesteps, ep_rew_mean])

        if self.num_timesteps % self.save_every == 0:
            self.counter += 1
            # 确保目录存在
            os.makedirs(self.model_save_dir, exist_ok=True)
            file_path = os.path.join(self.model_save_dir, f"training_data_{self.counter}.csv")
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timesteps", "Ep_Rew_Mean"])
                writer.writerows(self.data)

            if self.verbose > 0:
                print(f"Data saved to {file_path}")

            # 清空数据以记录下一个周期
            self.data = []

        return True

def test(env, save_dir, test_for=1000, model=None, render_video=False):
    obs, _ = env.reset()
    images = []
    im_counter = 0

    for idx in range(test_for):
        if model is None:
            print("No model, taking random actions")
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        obs, _, done, trunc, _ = env.step(action)
        if render_video:
            img = env.mujoco_renderer.render(render_mode="rgb_array")
            images.append(img)
        if done or trunc:
            time.sleep(1)
            obs, _ = env.reset()
            if render_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(os.path.join(save_dir, f'episode_{im_counter}.avi'), fourcc, 50,
                                        (500, 500))
                for img in images:
                    video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.destroyAllWindows()
                video.release()

                images = []
                im_counter += 1

    env.reset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True,
                        choices=['reach', 'standup', 'selfbody', 'catch'],
                        help='The demonstration environment to use. Must be one of "reach", "standup", "selfbody", '
                             '"catch"')
    parser.add_argument('--train_for', default=20000, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str, required=True,
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    parser.add_argument('--render_video', action='store_true',
                        help='Renders a video for each episode during the test run.')
    parser.add_argument('--use_muscle', action='store_true',
                        help='Use the muscle actuation model instead of spring-damper model if provided.')

    args = parser.parse_args()
    env_name = args.env
    algorithm = args.algorithm
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for
    render = args.render_video
    use_muscle = args.use_muscle

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("models", env_name, f"{save_model}_{timestamp}")
    log_dir = os.path.join("logs", env_name, f"{save_model}_{timestamp}")

    actuation_model = MuscleModel if use_muscle else SpringDamperModel

    env_names = {"reach": "MIMoReach-v0",
                 "standup": "MIMoStandup-v0",
                 "selfbody": "MIMoSelfBody-v0",
                 "catch": "MIMoCatch-v0"}

    env = gym.make(env_names[env_name], actuation_model=actuation_model)
    env.reset()

    if algorithm == 'PPO':
        from stable_baselines3 import PPO as RL
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC as RL
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3 as RL
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG as RL
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C as RL

    # Load pretrained model or create new one
    if load_model:
        model = RL.load(load_model, env)
    else:
        model = RL("MultiInputPolicy", env,
                   tensorboard_log=os.path.join("models", "tensorboard_logs", env_name, save_model),
                   verbose=1)

    # Create the callback
    callback = SaveTrainingDataCallback(save_every=save_every, log_dir=log_dir, model_save_dir=save_dir)

    # Train the model
    model.learn(total_timesteps=train_for, reset_num_timesteps=False, callback=callback)
    model.save(os.path.join(save_dir, "final_model.zip"))

    # Test the model
    test(env, save_dir, model=model, test_for=test_for, render_video=render)

if __name__ == '__main__':
    main()

