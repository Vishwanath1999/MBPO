# Reinforcement Learning with Model-Based Policy Optimization (MBPO)

This repository contains the implementation of Model-Based Policy Optimization (MBPO) algorithm for reinforcement learning. MBPO is an off-policy model-based reinforcement learning algorithm that uses an ensemble of models to learn dynamics and a policy to optimize control. It combines the benefits of model-based RL (sample efficiency) and model-free RL (policy optimization). The algorithm achieves state-of-the-art performance on several continuous control benchmark tasks.

## Installation

To run the code in this repository, you need to have Python 3.6 or later installed on your system. You also need to install the following dependencies:

- PyTorch: The deep learning library used for implementing the neural network models.
- NumPy: A library for numerical computing in Python.
- OpenAI Gym: A toolkit for developing and comparing reinforcement learning algorithms.
- Matplotlib: A plotting library for creating visualizations in Python.

You can install the dependencies using the following command:

```
pip install torch numpy gym matplotlib
```

## Usage

To use the MBPO algorithm, you need to import the necessary modules and classes from the code. Here is an example of how to use the `EnsembleModel` class for training and testing an agent:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mbpo import EnsembleModel, ReplayBuffer

# Create an instance of the model
input_dims = (state_dim,)
n_actions = action_dim
hidden_dims = 200
alpha = 0.001
weight_decay = 1e-4
n_models = 5
n_elites = 2
batch_size = 256
model = EnsembleModel(alpha, input_dims, n_actions, weight_decay, n_models, n_elites, hidden_dims, batch_size)

# Create an instance of the replay buffer
buffer_size = int(1e6)
replay_buffer = ReplayBuffer(input_dims, n_actions, buffer_size)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Collect data using the current policy
        action = model.get_action(state)
        next_state, reward, done, _ = env.step(action)

        # Store the transition in the replay buffer
        replay_buffer.store_transition(state, action, reward, next_state, done)

        # Update the model using MBPO
        model.update(replay_buffer)

        state = next_state

# Testing loop
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Get action from the model
        action = model.get_action(state)

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)

        state = next_state
```

You can customize the hyperparameters and other settings according to your needs. The above code provides a basic example of how to use the MBPO algorithm with the provided classes.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## References
- [Janner M, Fu J, Zhang M, Levine S. When to trust your model: Model-based policy optimization. Advances in neural information processing systems. 2019;32.](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html)
