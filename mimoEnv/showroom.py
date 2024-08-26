""" Simple script to view the showroom. We perform no training and MIMo takes no actions.
"""

import gymnasium as gym
import time
import numpy as np
import mimoEnv


def main():
    """ Creates the environment and takes 200 time steps. MIMo takes no actions.
    The environment is rendered to an interactive window.
    """

    env = gym.make("MIMoStandup-v0", render_mode = 'human')
    #env = gym.make("MIMoStandup-v0")
    max_steps = 200

    _ = env.reset()

    start = time.time()
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        if done or trunc:
            print("success", step)
            env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()


if __name__ == "__main__":
    main()
