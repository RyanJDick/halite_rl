import argparse
import glob
import os
import random

def write_episode_list(episodes, filename):
    with open(filename, 'w') as f:
        for episode in episodes:
            f.write(episode + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_dir")
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()

    #episode_files = os.listdir(args.episodes_dir)
    episode_files = glob.glob(os.path.join(args.episodes_dir, "*.json"))
    episode_files = [os.path.basename(f) for f in episode_files]
    random.Random(123).shuffle(episode_files) # Seed for consistent behaviour.

    split_idx = int(len(episode_files) * args.train_frac)
    train_files = sorted(episode_files[:split_idx])
    val_files = sorted(episode_files[split_idx:])

    write_episode_list(train_files, os.path.join(args.episodes_dir, "train_episodes.txt"))
    write_episode_list(val_files, os.path.join(args.episodes_dir, "val_episodes.txt"))
