# Practice 3 & 4
This repository contains implementations and experiments for three core RL algorithms in both discrete and continuous GridWorld environments.

Implemented algorithms include:

- **Deep SARSA** (on-policy value-based)  
- **DQN** (off-policy value-based with target network and replay)  
- **REINFORCE** (Monte-Carlo policy gradient for continuous actions)
 
---

## 🚀 Training
To train an RL agent, run the `train.py` script with the desired algorithm and optional arguments.
```bash
python train.py --algo {algorithm} [--map MAP_NAME] [--render]
```
**Arguments**
- --algo (str, required): Choose the learning algorithm.
  - Options: deepsarsa, dqn, reinforce
- --map (str, optional): Name of a custom map JSON file (e.g. custom_map.json).
- --render (flag, optional): Render the environment during training.

The trained policy will be saved in the `checkpoints/{algo}.pth` directory as a .pkl file.

**To run the TensorBoard:**
```bash
tensorboard --logdir runs/
```

TensorBoard logs include:
- **Reward Curve**
- **Epsilon Decay** (for DeepSARSA & DQN)  
- **Loss**
- **StateValueHeatmap**
- **PolicyArrows


## 🔍 Testing / Rendering
valuate or visualize a trained policy:
```bash
python test.py \
  --algo {deepsarsa|dqn|reinforce} \
  --map MAP_NAME.yaml \
  [--render]
```

- Loads `checkpoints/{algo}.pth
- Runs for a fixed number of episodes or until goal
- `--render` uses pygame to display each step

## 🌍 GridWorld Environments

There are two variants:

### Discrete-Action GridWorld (`gridworld_c1.py`)
- **State:** continuous \((row, col)\) in meters  
- **Actions:** 8 directions (N, NE, E, SE, S, SW, W, NW), fixed step length  
- **Rewards:**  
  - Move: **–1**  
  - Trap: **–100**, episode ends  
  - Goal: **+100**, episode ends  

### Continuous-Action GridWorld (`gridworld_c2.py`)
- **State:** continuous \((row, col)\) in meters  
- **Action:** 2D vector \(\in[-1,1]^2\) (clamped internally)  
- **Rewards:** same as discrete  

Each cell in the map can be one of:
- 🟩 **Normal** (0): free to move, reward –1  
- 🧱 **Wall** (1): blocks movement, reward –1  
- ☠️ **Trap** (2): ends episode, reward –100  
- 🎯 **Goal** (3): ends episode, reward +100  

---

## 📁 Folder Structure

```bash
Practice3_4/
├── algos/                # Algorithm implementations
│   ├── deepsarsa.py
│   ├── dqn.py
│   └── reinforce.py
├── configs/              # Map configuration files (YAML)
│   ├── map1.yaml
│   └── map2.yaml
├── env/                  # GridWorld environments
│   ├── gridworld_c1.py
│   └── gridworld_c2.py
├── checkpoints/          # Saved models (*.pth)
├── outputs/              # Generated plots (heatmaps, policy arrows)
├── runs/                 # TensorBoard logs
├── train.py              # Unified training script
├── test.py               # Policy evaluation / rendering script
└── README.md             # This file

```
