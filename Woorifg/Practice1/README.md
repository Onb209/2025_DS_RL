# Practice 1
It provides implementations and experiments for key RL methods in a discrete GridWorld environment.

Implemented algorithms include:

- **Monte Carlo Control**
- **Temporal-Difference Learning (TD(0))**
- **SARSA**
- **Q-Learning**

---

## 🚀 Training
To train an RL agent, run the `train.py` script with the desired algorithm and optional arguments.
```bash
python train.py --algo {algorithm} [--size SIZE] [--map MAP_NAME] [--render]
```
**Arguments**
- --algo (str, required): Choose the learning algorithm.
  - Options: pi, vi, mf_pred, mc, sarsa, q_learning
- --size (int, optional): Width & Height of the GridWorld. Default is 6. (Ignored if --map is specified.)
- --map (str, optional): Name of a custom map JSON file (e.g. custom_map.json).
- --render (flag, optional): Render the environment during training.

The trained policy will be saved in the checkpoints/ directory as a .pkl file.


![Output](assets/_img/animation.gif)

**Cliff Walking**

The map for the Cliff Walk environment is located at:  
`env/maps/cliff_walk.json`

To use this map, make the following modification in the `gridworld_env.py` file:

In both the `__init__()` and `reset()` functions, change:

```python
self.agent_pos = [0, 0]
```
to
```python
self.agent_pos = [3, 0]
```
Then run the following commands:
```bash
# Train with SARSA
python train.py --algo sarsa --map cliff_walk.json

# Train with Q-Learning
python train.py --algo q_learning --map cliff_walk.json
```
**SARSA:**

![Output](assets/_img/sarsa.gif)

**Q-Learing:**

![Output](assets/_img/q_learning.gif)

---

## 🖼️ Rendering a Trained Policy
You can visualize a learned policy using the `render.py` script:
```bash
python render.py --policy {path_to_policy.pkl} [--size SIZE] [--map MAP_NAME] [--random]
```
**Arguments**
- --policy (str, optional): Path to a .pkl policy file. If omitted, agent acts randomly
- --size (int, optional): Grid size (default 6). Ignored if --map is provided.
- --map (str, optional): Path to a map JSON file (e.g. custom_map.json).
- --random (flag): Use a randomly generated map instead of a fixed one.

This will render the agent's behavior following the trained policy in the GridWorld environment.

![ex](assets/_img/render_img.png)

**Cliff Walking**

The map for the Cliff Walk environment is located at:  
`env/maps/cliff_walk.json`

Run the following commands:
```bash
# Test with SARSA
python render.py --policy checkpoints/policy_sarsa.pkl --map cliff_walk.json

# Test with Q-Learning
python render.py --policy checkpoints/policy_q_learning.pkl --map cliff_walk.json
```
**SARSA:**

![Output](assets/_img/sarsa_test.gif)

**Q-Learing:**

![Output](assets/_img/q_learning_test.gif)

---

## Algorithms
<details><summary>Monte Carlo Control & TD(0)</summary>

![algo](assets/_img/mc_td.png)

</details>

<details><summary>SARSA</summary>

![algo](assets/_img/sarsa.png)
</details>

<details><summary>Q-Learning</summary>

![algo](assets/_img/q_learning.png)
</details>


---

## 🌍 GridWorld

The GridWorld environment is a 2D grid-based world where each cell can be one of the following types:

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

---

## 📁 Folder Structure

```bash
Practice1
├── train.py               # Main training script
├── render.py              # Visualization script
├── checkpoints/           # Saved policy files
├── env/                   # GridWorld environment
│   └── maps/              # Predefined map configurations
├── outputs/               # Plotted value tables and action maps
├── venv/                  # Virtual environment folder
├── algos/                 # Folder containing algorithm-related files
└── assets/                # Folder for environment assets (e.g., graphics)

```
