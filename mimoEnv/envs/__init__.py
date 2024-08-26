from mimoEnv.envs.dummy import MIMoDummyEnv
from mimoEnv.envs.dummy import MIMoV2DummyEnv
from mimoEnv.envs.reach import MIMoReachEnv
from mimoEnv.envs.standup import MIMoStandupEnv
from mimoEnv.envs.selfbody import MIMoSelfBodyEnv
from mimoEnv.envs.catch import MIMoCatchEnv
from mimoEnv.envs.dummy import MIMoMuscleDummyEnv
from mimoEnv.envs.muscle_test import MIMoStaticMuscleTestEnv
from mimoEnv.envs.muscle_test import MIMoVelocityMuscleTestEnv
from mimoEnv.envs.muscle_test import MIMoStaticMuscleTestV2Env
from mimoEnv.envs.muscle_test import MIMoVelocityMuscleTestV2Env
from mimoEnv.envs.muscle_test import MIMoComplianceEnv, MIMoComplianceMuscleEnv

import time
import numpy as np
import gymnasium as gym
import mimoEnv
def main():

    env = gym.make('MIMoStandup-v0')

    max_steps = 500

    _ = env.reset()

    start = time.time()
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape)
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        if done or trunc:
            env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()



if __name__ == "__main__":
    main()

# import time
# import numpy as np
# import gymnasium as gym
#
#
# def main():
#     # 初始化环境
#     env = MIMoStandupEnv()
#
#     # 可以选择渲染模式，'human' 是常用的模式，它会打开一个窗口显示仿真
#     env.render(mode='human')
#
#     # 重置环境，开始新的一回合
#     observation = env.reset()
#
#     # 总的仿真步骤
#     num_steps = 200  # 示例步数，可根据需要调整
#
#     for step in range(num_steps):
#         # 使用随机策略作为示例
#         action = env.action_space.sample()
#
#         # 执行动作，返回新的观测值、奖励、是否结束以及其他信息
#         observation, reward, done, info = env.step(action)
#
#         # 更新渲染窗口
#         env.render(mode='human')
#
#         # 打印信息，例如奖励
#         print(f"Step: {step}, Reward: {reward}")
#
#         # 可以添加额外的逻辑来决定是否结束仿真
#         if done:
#             break
#
#         # 为了看到仿真效果，可以在每步之间暂停一段时间
#         time.sleep(0.01)
#
#     # 关闭环境
#     env.close()
#
#
# if __name__ == "__main__":
#     main()
