from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.crowd_sim.simulator import CrowdSimSimulator
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class CrowdSimEnv(ScenicGymEnv):

    def __init__(self, max_steps=50):

        scenario = scenic.scenarioFromFile("/home/kxu/ScenicGym/src/scenic/simulators/crowd_sim/training_scenario.scenic",
                                   model="scenic.simulators.crowd_sim.model")
        # simulator = CrowdSimSimulator()

        super().__init__(scenario, CrowdSimSimulator(), max_steps=50)


vec_env = make_vec_env(CrowdSimEnv)

model = PPO("MultiInputPolicy", vec_env, verbose=1)
callback = CheckpointCallback(1e6, "trained_models/baseline_train/")


model.learn(total_timesteps=1e7, callback=callback, progress_bar=True)
model.save("baseline_ppo_crowd_nav")


