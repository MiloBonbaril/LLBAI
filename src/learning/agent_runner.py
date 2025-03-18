import os
import time
import numpy as np
import torch
import random
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

# Import local modules
from src.game_interface.game_interface import GameInterface
from src.preprocessing.frame_processor import FrameProcessor
from src.models.cnn_model import LLBAgent
from src.models.model_factory import ModelFactory


class AgentRunner:
    """
    Runs a trained or random model to play Lethal League Blaze.
    
    This class connects all components of the system:
    - Game interface for screen capture and input
    - Frame processor for preprocessing
    - Neural network model for action selection
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 model_path: Optional[str] = None,
                 exploration_rate: float = 1.0):
        """
        Initialize the agent runner.
        
        Args:
            config: Configuration dictionary with parameters
            model_path: Path to a saved model (if None, will use random actions)
            exploration_rate: Probability of taking random actions (epsilon)
        """
        self.config = config
        self.exploration_rate = exploration_rate
        
        # Set up the game interface
        window_name = config.get("window_name", "Lethal League Blaze")
        fps_limit = config.get("fps_limit", 30)
        self.game_interface = GameInterface(window_name=window_name, fps_limit=fps_limit)
        
        # Set up the frame processor
        frame_height = config.get("frame_height", 144)
        frame_width = config.get("frame_width", 256)
        stack_size = config.get("stack_size", 4)
        self.frame_processor = FrameProcessor(
            resize_shape=(frame_width, frame_height),
            stack_size=stack_size
        )
        
        # Set up the model
        input_shape = (stack_size, frame_height, frame_width)
        num_actions = config.get("num_actions", 10)
        device = config.get("device", "cpu")
        
        print(f"is cuda available: {torch.cuda.is_available()}")
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"AgentRunner Using device: {self.device}")
        
        # Create the model
        self.model_factory = ModelFactory(config)
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model, _, _, _, _ = self.model_factory.load_checkpoint(model_path)
        else:
            print("No model provided or model not found. Using random actions.")
            self.model = self.model_factory.create_model()
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Define action space
        self.action_map = {
            0: [],                # No action
            1: ["LEFT"],          # Move left
            2: ["RIGHT"],         # Move right
            3: ["JUMP"],          # Jump
            4: ["ATTACK"],        # Attack
            5: ["JUMP", "ATTACK"], # Jump + Attack
            6: ["JUMP", "LEFT"],   # Jump + Left
            7: ["JUMP", "RIGHT"],  # Jump + Right
            8: ["GRAB"],          # Grab
            9: ["BUNT"]           # Bunt
        }
        
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.fps_history = []
        
        # Create necessary directories
        os.makedirs("data/recordings", exist_ok=True)
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Process a raw frame and convert to PyTorch tensor.
        
        Args:
            frame: Raw BGR frame from screen capture
            
        Returns:
            torch.Tensor: Processed frame ready for the model
        """
        # Add to frame stack
        processed_stack = self.frame_processor.add_to_stack(frame)
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(processed_stack, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def select_action(self, state_tensor: torch.Tensor) -> int:
        """
        Select an action based on current state.
        
        Args:
            state_tensor: Processed state tensor
            
        Returns:
            int: Action index
        """
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Take random action
            return random.randint(0, len(self.action_map) - 1)
        
        # Use model to select action
        with torch.no_grad():
            action_probs = self.model.get_action_probs(state_tensor)
            
        # Select action with highest probability
        action = torch.argmax(action_probs, dim=1).item()
        return action
    
    def execute_action(self, action_idx: int, duration: float = 0.1) -> None:
        """
        Execute the selected action in the game.
        
        Args:
            action_idx: Index of the action to execute
            duration: Duration to hold the action
        """
        if action_idx in self.action_map:
            action = self.action_map[action_idx]
            self.game_interface.take_action(action, duration)
    
    def run_episode(self, max_steps: int = 1000, render: bool = True) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            max_steps: Maximum number of steps per episode
            render: Whether to display the game frames
            
        Returns:
            dict: Episode statistics
        """
        self.episode_steps = 0
        self.episode_start_time = time.time()
        frame_times = []
        
        print(f"Starting episode {self.episode_count + 1}")
        
        try:
            # Reset frame processor
            self.frame_processor.reset()
            
            # Initialize frame stack with initial frames
            for _ in range(self.frame_processor.stack_size):
                frame = self.game_interface.get_frame()
                self.frame_processor.add_to_stack(frame)
                time.sleep(0.05)  # Small delay between initial frames
            
            # Main episode loop
            for step in range(max_steps):
                step_start_time = time.time()
                
                # Get current frame
                frame = self.game_interface.get_frame()
                
                # Save frame occasionally for debugging
                if step % 100 == 0:
                    self.game_interface.save_frame(frame)
                
                # Process frame and select action
                state_tensor = self.preprocess_frame(frame)
                action_idx = self.select_action(state_tensor)
                
                # Execute action
                self.execute_action(action_idx)
                
                # Increment counters
                self.step_count += 1
                self.episode_steps += 1
                
                # Calculate step time
                step_time = time.time() - step_start_time
                frame_times.append(step_time)
                
                # Print progress
                if step % 10 == 0:
                    avg_fps = 1.0 / (sum(frame_times[-10:]) / len(frame_times[-10:]))
                    print(f"Step: {step}, Action: {action_idx}, FPS: {avg_fps:.1f}")
                    
        except KeyboardInterrupt:
            print("Episode interrupted by user")
            
        except Exception as e:
            print(f"Error during episode: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Calculate episode statistics
            episode_time = time.time() - self.episode_start_time
            avg_fps = self.episode_steps / episode_time if episode_time > 0 else 0
            self.fps_history.append(avg_fps)
            
            episode_stats = {
                "episode": self.episode_count,
                "steps": self.episode_steps,
                "time": episode_time,
                "avg_fps": avg_fps
            }
            
            print(f"Episode {self.episode_count + 1} finished:")
            print(f"  Steps: {self.episode_steps}")
            print(f"  Time: {episode_time:.1f} seconds")
            print(f"  Average FPS: {avg_fps:.1f}")
            
            self.episode_count += 1
            return episode_stats
    
    def run(self, num_episodes: int = 5, max_steps: int = 1000, render: bool = True) -> List[Dict[str, Any]]:
        """
        Run multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            render: Whether to display the game
            
        Returns:
            list: Statistics for each episode
        """
        episode_stats = []
        
        try:
            for i in range(num_episodes):
                stats = self.run_episode(max_steps, render)
                episode_stats.append(stats)
                
                # Short delay between episodes
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("Run interrupted by user")
            
        finally:
            # Display overall statistics
            total_steps = sum(stat["steps"] for stat in episode_stats)
            total_time = sum(stat["time"] for stat in episode_stats)
            avg_fps = total_steps / total_time if total_time > 0 else 0
            
            print("\nRun Summary:")
            print(f"  Episodes: {len(episode_stats)}")
            print(f"  Total Steps: {total_steps}")
            print(f"  Total Time: {total_time:.1f} seconds")
            print(f"  Average FPS: {avg_fps:.1f}")
            
            return episode_stats
    
    def close(self):
        """Clean up resources."""
        self.game_interface.close()
        print("Agent runner closed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Lethal League Blaze Agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
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


if __name__ == "__main__":
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
    
    # Run episodes
    agent.run(num_episodes=args.episodes, max_steps=args.steps)
    
    # Clean up
    agent.close()