# 2025_DS_RL

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

---

## 🚀 Training
To train an RL agent, run the train.py script with the desired algorithm and optional arguments.
```bash
python train.py --algo {algorithm} [--width WIDTH] [--height HEIGHT] [--render]

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

This will render the agent's behavior following the trained policy in the GridWorld environment.
---

## 📁 Folder Structure
```bash
.
├── train.py               # Main training script
├── render.py              # Visualization script
├── checkpoints/           # Saved policy files
├── env/                   # GridWorld environment
├── venv/
├── alogs/ 
└── assets/

## ✉️ Contact
For questions or suggestions, please open an issue or contact the repository maintainer.
