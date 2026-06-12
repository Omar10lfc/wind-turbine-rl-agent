"""Train the PPO pitch-control agent.

Supports single-seed training (default) and reproducible multi-seed training
for the N-seed evaluation. VecNormalize is used so observation/return scaling
is learned and saved alongside the model — required for correct inference.

Usage:
    python src/train.py                 # single run, seed from config
    python src/train.py --seed 3        # specific seed
    python src/train.py --all-seeds     # train every seed in config.train.seeds
"""
import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, get_paths
from env import WindTurbineEnv


def make_model_paths(models_dir, seed):
    """Model + VecNormalize stats paths for a given seed."""
    model_path = os.path.join(models_dir, f"ppo_wind_turbine_seed{seed}.zip")
    vecnorm_path = os.path.join(models_dir, f"vecnormalize_seed{seed}.pkl")
    return model_path, vecnorm_path


def train_one(seed, cfg=CONFIG):
    _, models_dir, logs_dir, _, _ = get_paths(cfg)
    t = cfg["train"]
    seed_log_dir = os.path.join(logs_dir, f"seed{seed}")
    os.makedirs(seed_log_dir, exist_ok=True)

    def _make_env():
        env = WindTurbineEnv(cfg, randomize=True)  # domain randomisation on
        return Monitor(env, seed_log_dir)

    venv = DummyVecEnv([_make_env])
    if t["use_vecnormalize"]:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        t["policy"],
        venv,
        verbose=1,
        learning_rate=t["learning_rate"],
        ent_coef=t["ent_coef"],
        n_steps=t["n_steps"],
        batch_size=t["batch_size"],
        gamma=t["gamma"],
        tensorboard_log=logs_dir,
        seed=seed,
    )

    print(f"=== Training seed {seed} for {t['total_timesteps']} timesteps ===")
    model.learn(total_timesteps=t["total_timesteps"])

    model_path, vecnorm_path = make_model_paths(models_dir, seed)
    model.save(model_path)
    if t["use_vecnormalize"]:
        venv.save(vecnorm_path)
    print(f"Saved model -> {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    seeds = CONFIG["train"]["seeds"]
    if args.all_seeds:
        for s in seeds:
            train_one(s)
    else:
        train_one(args.seed if args.seed is not None else seeds[0])
    print("Training complete.")


if __name__ == "__main__":
    main()
