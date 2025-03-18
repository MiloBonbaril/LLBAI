#!/usr/bin/env python3
"""
Run the Threaded Lethal League Blaze Agent.
This version uses multi-threading for improved performance.
"""

import os
import sys
import time
import argparse
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.learning.threaded_agent_runner import ThreadedAgentRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Threaded Lethal League Blaze Agent")
    parser.add_argument("--model", type=str, default=None,
                      help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=3,
                      help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000,
                      help="Maximum steps per episode")
    parser.add_argument("--time", type=int, default=300,
                      help="Maximum time per episode in seconds")
    parser.add_argument("--window", type=str, default="Lethal League Blaze",
                      help="Window name of the game")
    parser.add_argument("--exploration", type=float, default=1.0,
                      help="Exploration rate (0.0 to 1.0)")
    parser.add_argument("--fps", type=int, default=60,
                      help="FPS limit for game capture")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage instead of GPU")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Determine device
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration dictionary
    config = {
        "window_name": args.window,
        "fps_limit": args.fps,
        "frame_height": 144,
        "frame_width": 256,
        "stack_size": 4,
        "num_actions": 10,
        "device": device
    }
    
    print(f"Running on device: {device}")
    
    # Create and run agent
    agent = ThreadedAgentRunner(
        config=config,
        model_path=args.model,
        exploration_rate=args.exploration
    )
    
    print("Starting threaded agent runner. Press Ctrl+C to stop.")
    print("Giving you 3 seconds to focus on the game window...")
    time.sleep(3)
    
    try:
        # Run episodes
        agent.run(num_episodes=args.episodes, max_steps=args.steps, max_time=args.time)
    finally:
        # Always clean up, even if an error occurs
        agent.close()


if __name__ == "__main__":
    main()