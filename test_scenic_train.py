from scenic.gym import ScenicGymEnv, ScenicOAIGymEnv, ScenicOAI15GymEnv
import scenic
from scenic.simulators.crowd_sim.simulator import CrowdSimSimulator 
import numpy as np

scenario = scenic.scenarioFromFile("/home/kxu/ScenicGym/src/scenic/simulators/crowd_sim/training_scenario.scenic",
                                   model="scenic.simulators.crowd_sim.model") # shouldn't use 2D mode?

# env = ScenicGymEnv(scenario, CrowdSimSimulator(), max_steps=50)
env = ScenicOAI15GymEnv(scenario, CrowdSimSimulator(), max_steps=50)

observation, info = env.reset()

print(f"first reset obs: {observation['detected_human_num']}")
# print(f"OBS {obs}, info {info}")
render = True

episode_over = False

for i in range(3):
    while not episode_over:
        # action = env.action_space.sample_action() # dummy here ok this is not right
         
        action = env.action_space.sample() # dummy here ok this is not right
        # if i == 1:
            # print(f"Sampled action: {action}")
        # observation, reward, terminated, truncated, info = env.step(action)
        observation, reward, terminated, info = env.step(action)
        print(f"REWARD: {reward}")
        # print(f"obs: {observation['detected_human_num']} episode {i}")
        # print(observation[0]['position'])
        # episode_over = terminated or truncated
        episode_over = terminated

        if render:
            env.render()

    obs, info = env.reset()
    episode_over = False
    # print(f"OBS {obs}, info {info}")
    # print("finsih reset")


env.close()
