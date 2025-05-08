# Reinforcement Learning Algorithms in GridWorld

This repository contains hands-on practice code for reinforcement learning (RL) algorithms, designed to support educational lectures and tutorials. It provides implementations and experiments for key RL methods in a discrete GridWorld environment.

Implemented algorithms include:

- **Policy Iteration**
- **Value Iteration**
- **Monte Carlo Control**
- **Temporal-Difference Learning (TD(0))**
- **SARSA**
- **Q-Learning**

---

## 🌍 GridWorld

The GridWorld environment is a 2D grid where each cell can be one of the following types:

- 🟩 **Normal**: The agent can move to a normal cell with a reward of -1.
- 🧱 **Wall**: The agent cannot move into a wall cell. The agent stays in its current position and receives a reward of -1.
- ☠️ **Trap**: If the agent moves into a trap cell, it receives a reward of -100, and the episode ends.
- 🎯 **Goal**: If the agent reaches the goal cell, it receives a reward of 100, and the episode ends.

### 📏 Grid Dimensions

- The grid size can range from **5x5** to **10x10**.

### 🏃‍♂️ Actions

- The agent has 4 possible actions:  
  - ⬆️ **Move Up**  
  - ⬇️ **Move Down**  
  - ⬅️ **Move Left**  
  - ➡️ **Move Right**

### 🔄 Transitions and Rewards

- **Normal**: Transitioning into a normal cell results in a reward of -1.
- **Wall**: The agent cannot move into a wall, and stays in the same position with a reward of -1.
- **Trap**: If the agent lands on a trap, it receives a reward of -100, and the episode ends.
- **Goal**: If the agent reaches the goal, it receives a reward of 100, and the episode ends.

---

## 🛠️ Installation

We recommend using a virtual environment for package management. This project has been tested on **Ubuntu 20.04** with **Python 3.10**.

```bash
# Clone the repository and navigate to the project folder
cd {project_folder}

# Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install required packages
pip install gymnasium pygame
```
---

## 🚀 Training
To train an RL agent, run the train.py script with the desired algorithm and optional arguments.
```bash
python train.py --algo {algorithm} [--width WIDTH] [--height HEIGHT] [--render]
```
**Arguments**
- --algo (str, required): Choose the learning algorithm.
  - Options: vi, pi, mc, td0, sarsa, q_learning
- --width (int, optional): Width of the GridWorld. Default is 6.
- --height (int, optional): Height of the GridWorld. Default is 6.
- --render (flag, optional): Render the environment during training.

The trained policy will be saved in the checkpoints/ directory as a .pkl file.

---

## 🖼️ Rendering a Trained Policy
You can visualize a learned policy using the render.py script:
```bash
python render.py --policy {path_to_policy.pkl}
```
This will render the agent's behavior following the trained policy in the GridWorld environment.
---

## 📁 Folder Structure
```bash
.
├── train.py               # Main training script
├── render.py              # Visualization script
├── checkpoints/           # Saved policy files
├── env/                   # GridWorld environment
├── venv/                  # Virtual environment folder
├── alogs/                 # Folder containing algorithm-related files
└── assets/                # Folder for environment assets (e.g., graphics)

```
## ✉️ Contact
For questions or suggestions, please open an issue or contact the repository maintainer.
