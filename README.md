# Safe Driving with DQN in MetaDrive

Implementation for the "Safe Driving with Deep Q-Networks in MetaDrive"
homework (Practical Neural Networks). All 8 tasks (environment setup
through evaluation/visualization) are implemented and verified end to end.

## Project structure

```
metadrive_dqn/
├── README.md
├── requirements.txt
├── train.py                    # Task 6 + 7: training loop + full training run
├── evaluate.py                 # Task 8: evaluation + visualization
├── replay_buffer.py            # Task 4: ReplayBuffer
├── dqn_agent.py                # Task 5: DQNAgent
├── configs/
│   ├── __init__.py
│   └── env_config.py           # Task 1: MetaDriveEnv factory + sanity check
├── utils/
│   ├── __init__.py
│   └── processing.py           # Task 2: flatten_obs, discrete_to_continuous_action
├── models/
│   ├── __init__.py
│   ├── dqn_network.py          # Task 3: DQNNetwork (state -> hidden -> hidden -> 6 Q-values)
│   ├── dqn_trained.pt          # Final trained model weights (Task 7 output)
│   ├── training_history.json   # Full per-episode training history (Task 7 output)
│   └── training_curves.png     # Reward / episode length / crash-rate plots (Task 7/8 output)
└── tests/
    ├── __init__.py
    ├── test_tasks_1_2_3.py     # Integration test for Task 1-3
    ├── test_tasks_4_5.py       # Integration test for Task 4-5
    └── test_tasks_6_7.py       # Integration test for Task 6-7
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> MetaDrive needs a working OpenGL / Panda3D setup. If you only need
> headless training (no rendering), `use_render=False` (the default in
> `configs/env_config.py`) avoids most graphics-driver issues.

## Training on GPU

`DQNAgent` already auto-selects a GPU whenever one is visible to torch
(`device = "cuda" if torch.cuda.is_available() else "cpu"`) -- there's
nothing to turn on in the code itself. What actually determines whether
you train on GPU or CPU is which torch build is installed:

1. Install torch with an explicit CUDA wheel -- see the commented
   instructions at the top of `requirements.txt` (a plain
   `pip install torch` can silently give you a CPU-only build).
2. Verify it before training:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   This must print `True`.
3. `train.py`, `evaluate.py`, `inspect_dead_relu.py`, and
   `inspect_policy.py` all now print a `CUDA available: ...` line on
   startup, and `train.py`/`evaluate.py`/`inspect_*.py` accept a
   `--device {cuda,cpu}` flag if you want to force one explicitly instead
   of relying on auto-detect (`--device cuda` will raise a clear error
   instead of silently training on CPU if no GPU is actually visible):
   ```bash
   python -m train --episodes 800 --device cuda
   python -m evaluate --eval-episodes 20 --device cuda
   ```

Note on expected speedup: MetaDrive's environment stepping (physics +
scenario generation) runs on CPU regardless -- the GPU only accelerates
the network forward/backward passes in `agent.learn()`. With the default
small state vector and a 2-hidden-layer, `hidden_size=256` network, the
env-stepping CPU cost dominates wall-clock time, so GPU vs CPU training
speed may be closer than you'd expect for this particular assignment size.
GPU helps more if you increase `--hidden-size` substantially or raise
`batch_size` in `dqn_agent.py`.

### Windows install issue: "Long Path" error

Installing `metadrive-simulator` on Windows can fail with:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: '...events.out.tfevents...'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

This happens because some files bundled inside the MetaDrive package have
paths longer than Windows' default 260-character limit. Fix it by opening
PowerShell **as Administrator** and running:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then **fully restart** the machine (not just the terminal). If a
half-installed `metadrive` folder is left over from a previous failed
attempt, remove it manually before retrying:

```powershell
Remove-Item -Recurse -Force ".\.venv\Lib\site-packages\metadrive" -ErrorAction SilentlyContinue
pip install metadrive-simulator==0.4.3
```

### Windows runtime issue: torch DLL load failure

If running `train.py` or `evaluate.py` raises:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...torch\lib\c10.dll" or one of its dependencies.
```

this is a known conflict between panda3d's native DLLs (used by MetaDrive)
and torch's native DLLs on Windows. The fix already applied in this code:
`torch` (via `dqn_agent`) is imported **before** `metadrive` (via
`configs.env_config`) in both `train.py` and `evaluate.py`. If it still
happens on your machine, also try:

- Making sure conda's `base` environment is not active at the same time
  as your venv (`conda deactivate` first).
- `set KMP_DUPLICATE_LIB_OK=TRUE` before running the script.
- Repairing/reinstalling the Visual C++ Redistributable, then rebooting.

## How each task is implemented

### Task 1 — `configs/env_config.py`
- `create_metadrive_env(config_overrides=None, render=False)` builds a
  `MetaDriveEnv` with a reproducible config: fixed `start_seed`,
  `num_scenarios` (number of maps), `horizon` (max episode length),
  `traffic_density`, and explicit crash/termination flags
  (`crash_vehicle_done`, `crash_object_done`, `out_of_route_done`).
  `use_render` toggles between headless training and 3D visualization.
  Also includes safety-shaped reward settings by default
  (`use_lateral_reward=True`, `out_of_road_penalty` / `crash_vehicle_penalty`
  / `crash_object_penalty` = 20.0 instead of MetaDrive's own default of
  5.0) -- see `--legacy-reward` in `train.py` to disable this for comparison.
- `sanity_check_env(env, n_steps=20)` resets the env and takes random
  steps, asserting the observation/reward/info all look sane.

### Task 2 — `utils/processing.py`
- `flatten_obs(obs)` converts the raw MetaDrive observation (numpy array,
  or dict if extra sensors are enabled) into a flat `float32` vector.
- `discrete_to_continuous_action(action_idx)` maps action indices `0-9` to
  a `[steering, throttle_brake]` continuous action in `[-1, 1]²`.
  **Updated from the original 6-action table** (Left/Straight/Right ×
  Brake/Forward, steering only at full-lock -1.0/0.0/+1.0): diagnosed via
  `inspect_policy.py` that even a correctly-trained, non-collapsed network
  still ran off-road most of the time, because ordinary curves need a
  moderate steering angle that simply didn't exist as an option -- full
  lock oversteers, straight understeers off the outside of the curve.
  Added a "slight" steering level (±`SLIGHT_STEER`, default 0.4) on each
  side: 5 steering positions (hard-left/slight-left/straight/slight-right/
  hard-right) × 2 throttle/brake = 10 actions. **Breaking change**: any
  checkpoint trained under the old 6-action table is incompatible and
  must be retrained.

### Task 3 — `models/dqn_network.py`
- `DQNNetwork(state_size, action_size, hidden_size=256)`: two hidden
  `Linear + ReLU` layers, `state_size -> hidden -> hidden -> action_size`,
  outputting one raw Q-value per action (no output activation).

### Task 4 — `replay_buffer.py`
- `ReplayBuffer(capacity, seed=None)` backed by a `collections.deque`
  (auto-discards oldest transitions once full).
- `push(state, action, reward, next_state, done)` stores one transition.
- `sample(batch_size, device="cpu")` returns `(states, actions, rewards,
  next_states, dones)` as correctly-shaped PyTorch tensors.

### Task 5 — `dqn_agent.py`
- `DQNAgent(state_size, action_size, ...)` owns an online `DQNNetwork`, a
  target `DQNNetwork` (synced periodically, never trained directly), an
  `Adam` optimizer, and a `ReplayBuffer`.
- `select_action(state, greedy=False)`: epsilon-greedy action selection.
  `greedy=True` always takes the argmax (used at evaluation time).
- `decay_epsilon()`: multiplies epsilon by `epsilon_decay`, floored at
  `epsilon_end`. Called once per episode.
- `learn()`: samples a mini-batch, computes the Bellman target
  `y = r + (1 - d) * gamma * max_a' Q_target(s', a')` using the **target**
  network, computes `Q(s,a)` from the **online** network, backprops the
  MSE loss, and every `target_update_freq` calls hard-copies the online
  weights into the target network.
- `save(path)` / `load(path)`: persist/restore the online network's
  weights (`models/dqn_trained.pt`).

### Task 6 — `train.py` → `train_dqn(...)`
Per episode: reset env, epsilon-greedy action → continuous action →
`env.step()` → store transition → learn once `min_buffer_size` transitions
are collected. Episode ends on `terminated` or `truncated`. Records, per
episode: total reward, episode length, whether a crash occurred (from
`info["crash_vehicle"]` / `info["crash_object"]`), average loss, and
epsilon.

### Task 7 — `train.py` → `main()`
CLI-driven training run. Builds the env + agent, runs `train_dqn`, then
saves the trained model, the full training history (JSON), and the
training-curve plot.

```bash
python -m train --smoke-test          # quick 10-episode sanity run
python -m train --episodes 800        # full training run
```

### Task 8 — `evaluate.py`
- `evaluate_policy(...)`: runs the trained agent fully greedy (epsilon=0)
  for N episodes and reports average reward, average episode length, and
  crash rate.
- `analyze_training_curves(history)`: re-plots the training curves and
  numerically compares the first vs. last 10% of training episodes
  (reward / length / crash rate trend).
- `record_topdown_video(...)`: records a top-down view video of one greedy
  episode (headless, no on-screen window required).
- `run_3d_visualization(...)`: opens MetaDrive's interactive 3D window and
  drives with the trained agent, taking periodic screenshots.

```bash
python -m evaluate --eval-episodes 20                                     # numeric evaluation only
python -m evaluate --eval-episodes 0 --plot-only                          # training-curve analysis only
python -m evaluate --eval-episodes 0 --topdown-video --video-steps 500    # top-down video only
python -m evaluate --eval-episodes 0 --show-3d                            # live 3D view only
python -m evaluate --eval-episodes 20 --plot-only --topdown-video --show-3d   # everything at once
```

> `imageio` and `imageio-ffmpeg` (already in `requirements.txt`) are
> required for `--topdown-video`.

## Running the sanity tests

From the `metadrive_dqn/` directory:

```bash
python -m tests.test_tasks_1_2_3
python -m tests.test_tasks_4_5
python -m tests.test_tasks_6_7
```

Each prints `ALL CHECKS PASSED for Task N, M.` on success.

## Training run summary (reference results)

```
Training finished in 27.9 min (2.09 s/episode)
Saved trained model -> models/dqn_trained.pt
Saved training history -> models/training_history.json
Saved training curves -> models/training_curves.png
Summary:
  mean reward (last 10 eps): 91.13
  mean length (last 10 eps): 83.7
  crash rate  (last 10 eps): 0.30
```

Over the full 800-episode run, reward improved substantially (early vs.
late 10% of episodes: ~17 → ~87), while episode length dropped sharply
(~744 → ~84 steps) and crash rate rose (0% → ~38%). This reflects
MetaDrive's default reward function rewarding speed/progress more than it
penalizes collisions — a classic speed-vs-safety tradeoff, not a bug in
the implementation.

> **Update: this reference run used the OLD, safety-blind reward.** The
> rising-crash-rate pattern above is exactly why the safety-shaped reward
> (`use_lateral_reward=True` + higher `*_penalty` values) is now the
> **default** in `configs/env_config.py` (previously it was opt-in via a
> `--reward-shaping` flag you had to remember to pass). If you're seeing
> this same "reward goes up, crashes go up too" pattern, make sure you're
> not passing `--legacy-reward` to `train.py` -- that flag intentionally
> reproduces the old behavior above for comparison. Re-run without it (or
> re-run and compare both) and check `crashRate` / `offRoadRate` in the
> new `training_history.json` against this table. See the full report PDF
> for the detailed discussion.

## Notes / assumptions
- Default MetaDrive observation is the built-in state-vector observation
  (no camera), so `flatten_obs` mainly casts dtype/shape.
- Crash info is read from MetaDrive's `info` dict (`crash_vehicle`,
  `out_of_road`, etc.) — enabled by default, no extra config flag needed.
- `hidden_size=256`, `gamma=0.99`, `lr=1e-3`, `epsilon_decay=0.995`,
  `target_update_freq=1000` are the defaults used for the reference run;
  feel free to tune further.
- **Reward**: `use_lateral_reward=True`, `out_of_road_penalty` /
  `crash_vehicle_penalty` / `crash_object_penalty` = 50.0 (MetaDrive's own
  default is 5.0), `speed_reward=0.0` (default 0.1), and
  `success_reward=50.0` (default ~10.0) are baked into `DEFAULT_CONFIG` as
  of this version -- not opt-in anymore. Pass `--legacy-reward` to
  `train.py` only if you deliberately want to reproduce the old
  crash-prone, arrival-blind behavior for an ablation comparison.
- **Reward clipping is ASYMMETRIC**: `dqn_agent.py`'s `reward_clip_min`
  (default -10.0) / `reward_clip_max` (default 50.0). This guards against
  Q-value divergence (a real failure mode hit during development -- see
  git history / conversation notes if you have them) while still letting
  the large one-time `success_reward` register as clearly bigger than
  routine per-step reward instead of being squashed to the same size as a
  crash penalty.
- **Horizon raised to 2000** (was 1000) -- evaluation was showing average
  episode lengths of 660-820 steps out of a 1000-step horizon, meaning a
  meaningful fraction of episodes may have been cut off by the horizon
  before the car could reach the destination even when driving well.
- **Arrival is now tracked explicitly.** `train.py`'s history and
  `evaluate.py`'s results both include an `"arrived"` field
  (`info["arrive_dest"]`), plotted as a third line on `training_curves.png`
  and reported as `arrival rate` by `evaluate.py`. This matters because
  "didn't crash and didn't leave the road" is NOT the same as "reached the
  destination" -- an episode can survive to the horizon by, e.g., driving
  in circles or getting stuck, and none of the older metrics would have
  caught that. `evaluate.py` also reports a `timed-out-without-arriving
  rate` for exactly that case.
