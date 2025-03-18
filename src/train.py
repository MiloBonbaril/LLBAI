#!/usr/bin/env python3
"""
Train the Lethal League Blaze Agent using DQN.
This is a convenience script to start training from the project root.
"""

import os
import sys
import time
import argparse
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.learning.dqn_runner import DQNRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN Agent for Lethal League Blaze")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume training")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to train")
    parser.add_argument("--steps", type=int, default=5000,
                       help="Maximum steps per episode")
    parser.add_argument("--window", type=str, default="Lethal League Blaze",
                       help="Window name of the game")
    parser.add_argument("--manual-rewards", action="store_true",
                       help="Use manual rewards via keyboard input")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Disable visualization")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage instead of GPU")
    parser.add_argument("--double-dqn", action="store_true",
                       help="Use double DQN algorithm")
    parser.add_argument("--prioritized", action="store_true",
                       help="Use prioritized experience replay")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                       help="Starting exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                       help="Final exploration rate")
    parser.add_argument("--epsilon-decay", type=int, default=100000,
                       help="Steps over which to decay epsilon")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Determine device
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration dictionary
    config = {
        # Game interface
        "window_name": args.window,
        "fps_limit": 30,
        
        # Frame processing
        "frame_height": 144,
        "frame_width": 256,
        "stack_size": 4,
        
        # Action space
        "num_actions": 10,
        
        # Training
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": 0.99,
        "tau": 0.005,  # Soft update parameter
        "target_update_freq": 10,  # Hard update frequency (if tau=0)
        "memory_capacity": 100000,
        "initial_epsilon": args.epsilon_start,
        "final_epsilon": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay,
        "use_double_dqn": args.double_dqn,
        "use_prioritized_replay": args.prioritized,
        "clip_grad_norm": 10.0,
        
        # Rewards
        "use_manual_rewards": args.manual_rewards,
        "ball_detection_enabled": True,
        "score_detection_enabled": True,
        "survival_reward": 0.01,
        "hit_ball_reward": 1.0,
        "miss_ball_penalty": -0.5,
        "score_point_reward": 5.0,
        "lost_point_penalty": -3.0,
        "game_win_reward": 10.0,
        "game_loss_penalty": -5.0,
        
        # Manual rewards
        "manual_hit_reward": 1.0,
        "manual_miss_penalty": -0.5,
        "manual_score_reward": 5.0,
        "manual_lost_penalty": -3.0,
        "manual_win_reward": 10.0,
        "manual_lose_penalty": -5.0,
        
        # Training loop
        "max_episodes": args.episodes,
        "max_steps_per_episode": args.steps,
        "train_every_n_steps": 4,
        "updates_per_step": 1,
        "save_every_n_episodes": 10,
        "eval_every_n_episodes": 20,
        
        # Visualization
        "visualize": not args.no_visualize,
        "render_every_n_steps": 10,
        
        # Paths
        "checkpoint_dir": "data/models",
        "log_dir": "data/logs",
        
        # Device
        "device": device
    }
    
    print("=== Lethal League Blaze DQN Training ===")
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {args.steps}")
    print(f"Window name: {args.window}")
    print(f"Device: {device}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using Double DQN: {args.double_dqn}")
    print(f"Using Prioritized Replay: {args.prioritized}")
    print(f"Manual rewards: {args.manual_rewards}")
    print(f"Epsilon: {args.epsilon_start} → {args.epsilon_end} over {args.epsilon_decay} steps")
    
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
    else:
        print("Starting new training run")
    
    # Create and run the DQN trainer
    runner = DQNRunner(config, checkpoint_path=args.checkpoint)
    
    print("\nStarting DQN training. Press CTRL+C to stop.")
    print("Giving you 3 seconds to focus on the game window...")
    time.sleep(3)
    
    # Start training
    runner.train()


if __name__ == "__main__":
    main()