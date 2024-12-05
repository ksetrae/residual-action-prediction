import pathlib

import numpy as np
import gym
from stable_baselines3 import A2C

env = gym.make('Pendulum-v1', render_mode='human')
model = A2C.load("rl-baselines3-zoo/rl-trained-agents/a2c/Pendulum-v1_1/Pendulum-v1.zip")
dataset_dir_name = 'pendulum_expert_data'

observations = []
actions = []

obs, info = env.reset()
done = False
max_steps = 1000
step_counter = 0

while not done and step_counter < max_steps:
    action, _states = model.predict(obs)
    observations.append(obs)
    actions.append(action)
    obs, reward, done, terminated, info = env.step(action)
    env.render()
    step_counter += 1
    if step_counter % 10 == 0:
        print(f'step: {step_counter}')

dataset_path = pathlib.Path(f'datasets/{dataset_dir_name}')
dataset_path.mkdir(parents=True, exist_ok=True)

with open(dataset_path / 'observations.npy', 'wb') as f:
    np.save(f, np.array(observations), allow_pickle=False)

with open(dataset_path / 'actions.npy', 'wb') as f:
    np.save(f, np.array(actions), allow_pickle=False)

env.close()
