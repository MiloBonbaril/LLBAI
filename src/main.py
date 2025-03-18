#!/usr/bin/env python3
"""
Run the Lethal League Blaze Agent.
This is a convenience script to run the agent from the project root.
"""

import os
import sys
import time
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.learning.agent_runner import AgentRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Lethal League Blaze Agent")
    parser.add_argument("--model", type=str, default=None,
                      help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=3,
                      help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000,
                      help="Maximum steps per episode")
    parser.add_argument("--window", type=str, default="Lethal League Blaze",
                      help="Window name of the game")
    parser.add_argument("--exploration", type=float, default=1.0,
                      help="Exploration rate (0.0 to 1.0)")
    parser.add_argument("--fps", type=int, default=30,
                      help="FPS limit for game capture")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Configuration dictionary
    config = {
        "window_name": args.window,
        "fps_limit": args.fps,
        "frame_height": 144,
        "frame_width": 256,
        "stack_size": 4,
        "num_actions": 10,
        "device": "cuda"  # Use "cuda" for GPU
    }
    
    # Create and run agent
    agent = AgentRunner(
        config=config,
        model_path=args.model,
        exploration_rate=args.exploration
    )
    
    print("Starting agent runner. Press Ctrl+C to stop.")
    print("Giving you 3 seconds to focus on the game window...")
    time.sleep(3)
    
    try:
        # Run episodes
        agent.run(num_episodes=args.episodes, max_steps=args.steps)
    finally:
        # Always clean up, even if an error occurs
        agent.close()


if __name__ == "__main__":
    main()