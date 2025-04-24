from scenic.gym import ScenicGymEnv, ScenicOAIGymEnv, ScenicOAI15GymEnv
import scenic
from scenic.simulators.crowd_sim.simulator import CrowdSimSimulator 
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

scenario = scenic.scenarioFromFile("/home/kxu/ScenicGym/src/scenic/simulators/crowd_sim/training_scenario.scenic",
                                   model="scenic.simulators.crowd_sim.model") # shouldn't use 2D mode?

env = ScenicGymEnv(scenario, CrowdSimSimulator(), max_steps=50)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("baseline_nav")
# envs = gym.vector.AsyncVectorEnv([])
# env = ScenicOAI15GymEnv(scenario, CrowdSimSimulator(), max_steps=50)
# env = ScenicOAI15GymEnv(scenario, CrowdSimSimulator(), max_steps=50)

# observation, info = env.reset()


# env.close()
