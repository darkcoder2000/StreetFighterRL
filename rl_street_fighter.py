import json
import os
import glob
import sys
import gym
from ray import tune
from ray.rllib import agents
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A3CTrainer

import json
import os
import glob
import sys
import gym
from ray import tune
from ray.rllib import agents
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A3CTrainer

from env.streetfighter_env import StreetFighter

env_id = "StreetFighter"

# Where the trained agents and the logs will end up.
local_dir = f"tune_runs_{env_id}"

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": StreetFighter,
    "num_gpus": 1,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [128, 128, 128, 128],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}

def main():
    if "train" in sys.argv:
        train()
    elif "run" in sys.argv:
        run()
    else:
        print("Please specify either 'train' or 'run'.")


def train():
    # How many time steps to run the experiment for.
    time_steps_total = 1_000_000

    # Run the experiment.
    results = tune.run(
        PPOTrainer,
        #A3CTrainer,
        config=config,
        metric="episode_reward_mean",
        mode="max",
        stop={"timesteps_total": time_steps_total},
        checkpoint_at_end=True,
        checkpoint_freq=10,
        local_dir=local_dir,
    )

    # Get the checkpoints.
    checkpoints = results.get_trial_checkpoints_paths(
        trial=results.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean")
    for checkpoint in checkpoints:
        checkpoint_path = checkpoint[0]
        print("Checkpoint path:", checkpoint_path)

    # Clean up.
    subfolders = glob.glob(os.path.join(local_dir, "*"), recursive=True)
    subfolders = [subfolder for subfolder in subfolders if os.path.isdir(subfolder)]
    for subfolder in subfolders:
        subsubfolders = glob.glob(os.path.join(subfolder, "*"), recursive=True)
        subsubfolders = [subsubfolder for subsubfolder in subsubfolders if os.path.isdir(subsubfolder)]
        for subsubfolder in subsubfolders:
            subsubfolder_cleaned = subsubfolder.replace("[", "").replace("]", "").replace(", ", "-")
            if subsubfolder_cleaned != subsubfolder:
                os.rename(subsubfolder, subsubfolder_cleaned)
                print(f"Renamed {subsubfolder} to {subsubfolder_cleaned}")

    # Sound an alarm using unix escape sequence.
    print("Done.")
    print("\a")
    return

def run():
    # Find all the occurences of params.json in the directory tune_runs.
    params_paths = glob.glob(os.path.join(local_dir, "**", "params.json"), recursive=True)
    params_paths = sorted(params_paths)

    # Find all the checkpoints per params.json.
    pairs = []
    for params_path in params_paths:
        search_path = os.path.join(os.path.dirname(params_path), "**")
        checkpoint_paths = glob.glob(search_path, recursive=True)
        checkpoint_paths = [checkpoint_path for checkpoint_path in checkpoint_paths if
                            not os.path.isdir(checkpoint_path)]
        checkpoint_paths = [checkpoint_path for checkpoint_path in checkpoint_paths if
                            os.path.basename(checkpoint_path).startswith(
                                "checkpoint-") and not checkpoint_path.endswith(".tune_metadata")]
        checkpoint_paths = sorted(checkpoint_paths)
        pairs += [(params_path, checkpoint_path) for checkpoint_path in checkpoint_paths]

    # Get the user input.
    if len(pairs) == 0:
        print("No checkpoints found.")
    elif len(pairs) == 1:
        user_index = 0
    else:
        print("Select a checkpoint:")
        for index, (_, subfolder) in enumerate(pairs):
            print(f"{index: >2}: {subfolder}")
        user_index = int(input("Enter the index of the checkpoint: "))

    # Get config path and checkpoint path.
    config_path, checkpoint_path = pairs[user_index]
    assert os.path.exists(config_path)
    assert os.path.exists(checkpoint_path)
    print("Config path:", config_path)
    print("Checkpoint path:", checkpoint_path)

    # Read the config.
    with open(config_path, "r") as file:
        enjoy_config = json.load(file)
        enjoy_config = {key: value for key, value in enjoy_config.items() if key not in ["num_gpus", "num_workers"]}

    print(enjoy_config, config_path)
    enjoy_config["env"] = StreetFighter

    # Load the agent.
    print("Loading agent...")
    agent = agents.ppo.PPOTrainer(config=enjoy_config)
    agent.restore(checkpoint_path)
    print("Agent loaded.")

    # Create the environment.
    print("Creating the environment.")
    environment = StreetFighter()
    observation = environment.reset()
    done = False
    while not done:
        action = agent.compute_action(observation)
            
        # Execute the action.
        observation, reward, done, info = environment.step(action)
        
        # Render the environment and print the data.
        environment.render()

        if done:
            observation = environment.reset()
            done = False

if __name__ == "__main__":
    main()