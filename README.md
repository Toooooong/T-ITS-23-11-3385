
# TTLO Environment & PDIML Algorithm

## Overview

This repository contains two key components:

1. **TTLO Environment**: A custom Gym environment designed to address the high-speed rail （HSR）Train Trajectory Optimization (TTO) problem using a novel reconceptualization approach.
2. **PDIML Algorithm**: A novel first-order gradient-based meta-reinforcement learning (meta-RL) algorithm for HSR operation tasks, combining Proximal Policy Optimization (PPO) with Policy Deviation Integre (PDI) and Reptile meta-learning.

## Dependent Environment

Ensure that the following dependencies are installed in your Python environment before running the code:

- `python: 3.8.16`
- `numpy: 1.24.2`
- `torch: 2.0.0`
- `gym: 0.26.2`
- `matplotlib: 3.7.1`
- `pandas: 1.5.3`
- `scipy: 1.10.1`
- `pygame: 2.3.0`

You can install the required packages using the following command:

```bash
pip install numpy==1.24.2 torch==2.0.0 gym==0.26.2 matplotlib==3.7.1 pandas==1.5.3 scipy==1.10.1 pygame==2.3.0
```

## TTLO Environment

### Overview

The **TTLO Environment** redefines the traditional TTO (Task-To-Optimization) issue. In this environment, instead of directly seeking the optimal solution, the agent refines a suboptimal PMP (Policy Meta-Parameters). The action space is compressed from a unidimensional scale into a compact three-dimensional domain, encapsulating both retrospective and prospective state horizons. The reward function is designed for on-time evaluation, avoiding delayed feedback.

### Structure

- **Action Space**: A three-dimensional domain.
- **State Space**: Contains both retrospective and prospective elements.
- **Reward Function**: Designed for immediate feedback, evaluated on-time to avoid delays.

### Files & Directories

- The environment code is located in the `./TTLO Env/` directory.
- Environment parameter data is stored in `./TTLO Env/TTLO/data/`.
- You can modify the files under `./TTLO Env/TTLO/data/parameter/` to create different meta-environment tasks.
- The data generation script is available at `./TTLO Env/TTLO/get_data.py`, which allows you to generate new environment data.
  
### Registration

To use the environment, you need to place the files in `TTLO Env/` under the `./gym.envs/` directory and register the environment in `./gym.envs/__init__.py`:

### Render Modes

The environment supports two render modes:

1. **Plt (matplotlib)**: For visualizing data plots.
2. **Pygame**: For graphical interaction.

You can switch between these modes based on your needs during testing and debugging.

---

## PDIML Algorithm

### Overview

The **PDIML Algorithm** (Policy Deviation Integre with Meta-Learning) is a first-order gradient-based meta-RL algorithm that is designed for efficient adaptation across a distribution of tasks with complex dynamic constraints, specifically in high-speed rail (HSR) operations.

Key components:
- **Outer Meta-Learning Loop**: Integrates Policy Deviation Integre (PDI) with Reptile meta-learning to initialize neural network parameters for fast adaptation.
- **Inner DRL Loop**: Utilizes Proximal Policy Optimization (PPO) to iteratively refine policy parameters.
- **Goal**: Accelerates learning by initializing neural network models efficiently, enabling adaptation with limited training data.

### Structure

- **Main Program**: Located in `PDIML algorithm/train.py`.
- **PPO and PDI**: Found in `PDIML algorithm/PPO.py`.
  - The `policy_deviation_integral` method implements the PDI algorithm.
  - The `update_init_params` method in `train.py` implements the Reptile meta-learning update.
- **Control Loop**: The control loop and interactions with the TTLO environment are implemented in `train.py`.

### Running the PDIML Algorithm

To run the algorithm, execute the following:

```bash
python PDIML algorithm/train.py
```

This will train the agent in the TTLO environment using the PDIML meta-learning approach.

---

## How to Modify and Customize

1. **Generate Data**: Use the `get_data.py` script under `./TTLO Env/TTLO/` to generate new environment parameter data.
2. **Modify Meta-Tasks**: Customize the environment by editing files under `./TTLO Env/TTLO/data/parameter/` to create different meta-environment tasks.
3. **Train the Agent**: Train the agent by running the main training script at `PDIML algorithm/train.py`.

### Example Code Snippets

Here is an example of how to register and use the TTLO environment:

```python
import gym

env = gym.make('TTLO-v0')
observation = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode='plt')
```

---

## Conclusion

This repository provides a framework for solving dynamic, meta-learning based optimization tasks in complex environments using both the TTLO environment and the PDIML algorithm. By modifying environment parameters and training the model using meta-RL techniques, you can explore various optimization challenges and tasks efficiently.

