import os
import random
import torch
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from evnrionment code here import CryoAblationEnv 





class PrintDiceScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PrintDiceScoreCallback, self).__init__(verbose)
        self.dice_scores = []
        self.rewards = []

    def _on_step(self) -> bool:
       
        infos = self.locals.get("infos", [])
        for info in infos:
           
            
           
            
            if "cancer_dice" in info:
                self.dice_scores.append(info["cancer_dice"])
        
        
        
        
        rewards = self.locals.get("rewards", [])
        self.rewards.extend(rewards)
        return True







    def _on_rollout_end(self):
        if self.dice_scores:
            mean_dice = sum(self.dice_scores) /len(self.dice_scores)
            mean_reward = sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
            print(f"Mean score: {mean_dice:.4f},  Reward: {mean_reward:.4f}")
            self.dice_scores = []
            self.rewards = []

def get_valid_patient_dirs(base_dir):
    
    
    
    patient_dirs = [
        os.path.join(base_dir, p) for p in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, p)) and
           os.path.exists(os.path.join(base_dir, p, "l_a1.nii.gz"))
    ]
    return patient_dirs








def make_env(patient_dir):
    return lambda: CryoAblationEnv(patient_dirs=[patient_dir])
    
if __name__ == "__main__":
    multiprocessing.freeze_support()

    base_dir = r"C:\Users\user\OneDrive\Documents\RESEARCH PROJECT FINAL\10_patients"
    patient_dirs = get_valid_patient_dirs(base_dir)

    random.shuffle(patient_dirs)

    num_envs = min(4, len(patient_dirs))
    env_fns = [make_env(patient_dirs[i]) for i in range(num_envs)]

    if num_envs == 0:
        raise ValueError("NO")

    env = SubprocVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy", env, gamma=0.99, ent_coef=0.005, learning_rate=0.0001,
        verbose=0, tensorboard_log=None, n_steps=256, batch_size=64, n_epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    callback = PrintDiceScoreCallback()
    print("Started")
    try:
        model.learn(total_timesteps=500000, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        model.save("ppo_cryo_final")
        env.close()
