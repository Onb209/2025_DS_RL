# Practice 5

## 🚀 Training
```bash
python main.py -t -n {folder_name}
```
**Arguments**
- -t, --train: Enables training mode. If not specified, the script runs in test mode.
- -n, --name: Name of the experiment. The model and logs will be saved in logs/<name>/.

**To run the TensorBoard:**
```bash
tensorboard --logdir tb_logs/
```
## 🔍 Testing / Rendering
```bash
python main.py -l {path_to_saved_model} [--eval]
```
**Arguments**
- -l, --load: Path to a saved model file (e.g., .zip) to load and run in test mode.
- --eval: Evaluate the model and print task-specific success rates.

---

## 📁 Folder Structure

```bash
Practice5/
├── pbd_simulator/        # PBD simulator
├── logs/                 # Saved models (*.zip)
├── tb_logs/              # TensorBoard logs
├── config.yaml           # PPO training config
├── main.py
└── README.md            

```