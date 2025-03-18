import os
import time
import numpy as np
import torch
import random
import argparse
import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
import queue
import threading
import keyboard

# Import local modules
from ..game_interface.game_interface import GameInterface
from ..preprocessing.frame_processor import FrameProcessor
from ..models.cnn_model import LLBAgent
from ..models.model_factory import ModelFactory
from .dqn import DQNTrainer
from .reward import RewardSystem, ManualRewardSystem


class DQNRunner:
    """
    DQN training system for Lethal League Blaze.
    
    Combines all components for end-to-end reinforcement learning:
    - Game interface
    - Frame processor
    - DQN trainer
    - Reward system
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: Optional[str] = None):
        """
        Initialize the DQN runner.
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to a checkpoint to load (optional)
        """
        self.config = config
        self.start_time = time.time()
        
        # Create directories
        os.makedirs(config.get("checkpoint_dir", "data/models"), exist_ok=True)
        os.makedirs(config.get("log_dir", "data/logs"), exist_ok=True)
        
        # Initialize game interface
        window_name = config.get("window_name", "Lethal League Blaze")
        fps_limit = config.get("fps_limit", 30)
        self.game_interface = GameInterface(window_name=window_name, fps_limit=fps_limit)
        
        # Initialize frame processor
        frame_height = config.get("frame_height", 144)
        frame_width = config.get("frame_width", 256)
        stack_size = config.get("stack_size", 4)
        self.frame_processor = FrameProcessor(
            resize_shape=(frame_width, frame_height),
            stack_size=stack_size
        )
        
        # Define model parameters
        self.input_shape = (stack_size, frame_height, frame_width)
        self.num_actions = config.get("num_actions", 10)
        config["input_shape"] = self.input_shape
        config["num_actions"] = self.num_actions
        
        # Initialize model factory
        self.model_factory = ModelFactory(config)
        
        # Initialize DQN trainer
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Load model from checkpoint
            print(f"Loading DQN from checkpoint: {checkpoint_path}")
            self.model, _, _, _, loaded_config = self.model_factory.load_checkpoint(checkpoint_path)
            
            # Update config with loaded values
            for key, value in loaded_config.items():
                if key in config:
                    config[key] = value
            
            # Create trainer with loaded model
            self.trainer = DQNTrainer(config, model=self.model)
            
            # Load trainer state
            self.trainer.load_checkpoint(checkpoint_path)
        else:
            # Create new model and trainer
            print("Initializing new DQN trainer")
            self.model = self.model_factory.create_model()
            self.trainer = DQNTrainer(config, model=self.model)
        
        # Initialize reward system
        use_manual_rewards = config.get("use_manual_rewards", False)
        if use_manual_rewards:
            self.reward_system = ManualRewardSystem(config)
            print("Using manual reward system")
        else:
            self.reward_system = RewardSystem(config)
            print("Using automatic reward system")
        
        # Define action mapping
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
        
        # Training parameters
        self.max_episodes = config.get("max_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 10000)
        self.train_every_n_steps = config.get("train_every_n_steps", 4)
        self.save_every_n_episodes = config.get("save_every_n_episodes", 10)
        self.eval_every_n_episodes = config.get("eval_every_n_episodes", 10)
        self.updates_per_step = config.get("updates_per_step", 1)
        
        # Visualization parameters
        self.visualize = config.get("visualize", True)
        self.render_every_n_steps = config.get("render_every_n_steps", 10)
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.current_episode = 0
        
        # For manual rewards
        self.reward_queue = queue.Queue()
        if use_manual_rewards:
            # Start keyboard listener thread
            self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
            self.keyboard_thread.start()
    
    def _keyboard_listener(self):
        """Thread function for listening to keyboard input for manual rewards."""
        # Define key mappings for rewards
        reward_keys = {
            'h': 'hit',      # Ball hit
            'm': 'miss',     # Ball miss
            's': 'score',    # Scored a point
            'l': 'lost',     # Lost a point
            'w': 'win',      # Won the game
            'd': 'lose'      # Lost the game
        }
        
        # Set up keyboard hooks
        for key in reward_keys:
            keyboard.add_hotkey(key, lambda k=key: self.reward_queue.put(k))
        
        # Keep thread alive
        while True:
            time.sleep(0.1)
    
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
        tensor = torch.tensor(processed_stack, dtype=torch.float32, device=self.trainer.device)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def run_episode(self, train: bool = True, render: bool = True, 
                    max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single training or evaluation episode.
        
        Args:
            train: Whether to train the model during this episode
            render: Whether to display the game frames
            max_steps: Maximum steps for this episode
            
        Returns:
            dict: Episode statistics
        """
        if max_steps is None:
            max_steps = self.max_steps_per_episode
            
        # Initialize episode
        self.frame_processor.reset()
        self.reward_system.reset()
        
        episode_reward = 0.0
        episode_start_time = time.time()
        
        # Initialize state with initial frames
        state = None
        for _ in range(self.frame_processor.stack_size):
            frame = self.game_interface.get_frame()
            state_tensor = self.preprocess_frame(frame)
            state = state_tensor.cpu().numpy()
            time.sleep(0.05)  # Small delay between initial frames
        
        # Initialize statistics
        steps = 0
        losses = []
        rewards = []
        actions_taken = [0] * self.num_actions
        manual_rewards_applied = 0
        training_updates = 0
        training_time = 0.0
        
        try:
            # Main episode loop
            for step in range(max_steps):
                step_start_time = time.time()
                
                # Select action
                state_tensor = torch.tensor(state, device=self.trainer.device)
                action = self.trainer.select_action(state_tensor, evaluate=not train)
                
                # Execute action in game
                action_keys = self.action_map[action]
                self.game_interface.take_action(action_keys, duration=0.1)
                
                # Update action statistics
                actions_taken[action] += 1
                
                # Get next frame and state
                frame = self.game_interface.get_frame()
                next_state_tensor = self.preprocess_frame(frame)
                next_state = next_state_tensor.cpu().numpy()
                
                # Calculate reward
                reward = self.reward_system.calculate_reward(frame, action)
                
                # Check for manual rewards
                if isinstance(self.reward_system, ManualRewardSystem):
                    while not self.reward_queue.empty():
                        key = self.reward_queue.get()
                        manual_reward = self.reward_system.process_key(key)
                        reward += manual_reward
                        manual_rewards_applied += 1
                
                # Placeholder for done and win signals (would be detected from game state)
                done = False
                win = False
                
                # Add experience to replay memory
                if train:
                    self.trainer.add_experience(state, action, reward, next_state, done)
                
                # Update state and accumulate reward
                state = next_state
                episode_reward += reward
                rewards.append(reward)
                
                # Perform training updates
                if train and step % self.train_every_n_steps == 0 and self.trainer.memory.is_ready(self.trainer.batch_size):
                    train_start = time.time()
                    
                    # Perform multiple updates per step if configured
                    for _ in range(self.updates_per_step):
                        loss = self.trainer.train_batch()
                        losses.append(loss)
                        training_updates += 1
                    
                    training_time += time.time() - train_start
                
                # Increment counters
                steps += 1
                self.total_steps += 1
                
                # Render occasionally
                if render and step % self.render_every_n_steps == 0:
                    # Display the current frame with some statistics
                    display_frame = frame.copy()
                    
                    # Add text with stats
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(display_frame, f"Episode: {self.current_episode}", (10, 20), font, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"Step: {step}", (10, 40), font, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"Reward: {episode_reward:.2f}", (10, 60), font, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"Action: {action}", (10, 80), font, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"Epsilon: {self.trainer.epsilon:.2f}", (10, 100), font, 0.5, (0, 255, 0), 1)
                    
                    # Display the frame
                    cv2.imshow("LLB Agent Training", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Training interrupted by user (q pressed)")
                        break
                
                # Print progress occasionally
                if step % 100 == 0:
                    elapsed = time.time() - episode_start_time
                    print(f"Step: {step}/{max_steps}, "
                          f"Reward: {episode_reward:.2f}, "
                          f"Epsilon: {self.trainer.epsilon:.2f}, "
                          f"Steps/sec: {step/elapsed:.1f}")
                
                # Check for manual termination
                if keyboard.is_pressed('esc'):
                    print("Training interrupted by user (ESC pressed)")
                    break
                    
                # Check for done condition (would come from game state in a full implementation)
                if done:
                    break
                    
        except KeyboardInterrupt:
            print("Episode interrupted by user")
            
        except Exception as e:
            print(f"Error during episode: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Calculate episode statistics
            episode_time = time.time() - episode_start_time
            fps = steps / episode_time if episode_time > 0 else 0
            avg_loss = np.mean(losses) if losses else 0.0
            
            # Save episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            
            # Print episode summary
            print(f"\nEpisode {self.current_episode} summary:")
            print(f"  Steps: {steps}")
            print(f"  Time: {episode_time:.1f} seconds")
            print(f"  Average FPS: {fps:.1f}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Training updates: {training_updates}")
            print(f"  Training time: {training_time:.1f} seconds")
            print(f"  Manual rewards applied: {manual_rewards_applied}")
            
            # Most frequent actions
            action_counts = [(i, count) for i, count in enumerate(actions_taken)]
            action_counts.sort(key=lambda x: x[1], reverse=True)
            print("  Action distribution:")
            for action_idx, count in action_counts[:3]:
                percent = (count / steps) * 100 if steps > 0 else 0
                print(f"    {action_idx}: {count} ({percent:.1f}%)")
            
            # Close visualization window
            if render:
                cv2.destroyAllWindows()
            
            # Return episode statistics
            return {
                "episode": self.current_episode,
                "steps": steps,
                "time": episode_time,
                "fps": fps,
                "reward": episode_reward,
                "loss": avg_loss,
                "actions": actions_taken,
                "training_updates": training_updates,
                "training_time": training_time
            }
    
    def evaluate(self, num_episodes: int = 3, max_steps: int = None) -> Dict[str, Any]:
        """
        Evaluate the current model without training.
        
        Args:
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            
        Returns:
            dict: Evaluation statistics
        """
        print(f"\nEvaluating model for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_steps = []
        
        for i in range(num_episodes):
            print(f"Evaluation episode {i+1}/{num_episodes}")
            stats = self.run_episode(train=False, render=True, max_steps=max_steps)
            eval_rewards.append(stats["reward"])
            eval_steps.append(stats["steps"])
        
        avg_reward = np.mean(eval_rewards)
        avg_steps = np.mean(eval_steps)
        
        print(f"\nEvaluation results:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average steps: {avg_steps:.1f}")
        
        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "rewards": eval_rewards,
            "steps": eval_steps
        }
    
    def visualize_training(self):
        """Visualize training statistics with plots."""
        if not self.episode_rewards:
            print("No episodes completed yet. Nothing to visualize.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Statistics", fontsize=16)
        
        # Plot episode rewards
        ax1 = axes[0, 0]
        ax1.plot(self.episode_rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.grid(True)
        
        # Plot episode lengths
        ax2 = axes[0, 1]
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.grid(True)
        
        # Plot loss history
        ax3 = axes[1, 0]
        if self.trainer.loss_history:
            # Smooth loss for better visualization
            window_size = min(100, len(self.trainer.loss_history))
            smoothed_loss = np.convolve(
                self.trainer.loss_history, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            ax3.plot(smoothed_loss)
            ax3.set_title(f"Training Loss (Smoothed, Window={window_size})")
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Loss")
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, "No loss data yet", ha='center', va='center')
            ax3.set_title("Training Loss")
        
        # Plot epsilon decay
        ax4 = axes[1, 1]
        steps = np.linspace(0, self.trainer.train_step, 100)
        epsilon_initial = self.trainer.initial_epsilon
        epsilon_final = self.trainer.final_epsilon
        epsilon_decay_steps = self.trainer.epsilon_decay_steps
        
        epsilon_values = [
            max(epsilon_final, epsilon_initial - (s / epsilon_decay_steps) * (epsilon_initial - epsilon_final))
            for s in steps
        ]
        
        ax4.plot(steps, epsilon_values)
        ax4.axvline(x=self.trainer.train_step, color='r', linestyle='--', label='Current Step')
        ax4.axhline(y=self.trainer.epsilon, color='g', linestyle='--', label='Current Epsilon')
        ax4.set_title("Epsilon Decay")
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Epsilon")
        ax4.legend()
        ax4.grid(True)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plot_dir = self.config.get("log_dir", "data/logs")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"training_stats_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Training statistics plot saved to {plot_path}")
        
        # Show the figure
        plt.show()
    
    def train(self):
        """Run the full training process."""
        print(f"Starting training for {self.max_episodes} episodes")
        print(f"Max steps per episode: {self.max_steps_per_episode}")
        print(f"Training device: {self.trainer.device}")
        
        try:
            # Training loop
            for episode in range(self.max_episodes):
                self.current_episode = episode
                
                # Run training episode
                print(f"\nStarting episode {episode+1}/{self.max_episodes}")
                stats = self.run_episode(train=True, render=self.visualize)
                
                # Save checkpoint periodically
                if (episode + 1) % self.save_every_n_episodes == 0:
                    checkpoint_path = self.trainer.save_checkpoint(
                        episode=episode+1,
                        rewards=self.episode_rewards
                    )
                    print(f"Checkpoint saved to {checkpoint_path}")
                
                # Evaluate periodically
                if (episode + 1) % self.eval_every_n_episodes == 0:
                    self.evaluate(num_episodes=3)
                    
                    # Visualize training progress
                    if self.visualize:
                        self.visualize_training()
                
                # Check for training completion
                if keyboard.is_pressed('q'):
                    print("Training stopped by user")
                    break
                
        except KeyboardInterrupt:
            print("Training interrupted by user")
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Final evaluation
            print("\nTraining completed, running final evaluation...")
            self.evaluate(num_episodes=5)
            
            # Final checkpoint
            final_checkpoint_path = self.trainer.save_checkpoint(
                episode=self.current_episode+1,
                rewards=self.episode_rewards
            )
            print(f"Final checkpoint saved to {final_checkpoint_path}")
            
            # Visualize final training statistics
            if self.visualize:
                self.visualize_training()
            
            # Print training summary
            total_time = time.time() - self.start_time
            print("\nTraining summary:")
            print(f"  Episodes completed: {self.current_episode+1}")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Total training time: {total_time/3600:.1f} hours")
            print(f"  Average reward per episode: {np.mean(self.episode_rewards):.2f}")
            print(f"  Best episode reward: {np.max(self.episode_rewards):.2f}")
            print(f"  Final epsilon: {self.trainer.epsilon:.4f}")
            
            # Clean up resources
            self.close()
    
    def close(self):
        """Clean up resources."""
        self.game_interface.close()
        print("DQN runner closed")


if __name__ == "__main__":
    # This allows running this module directly for testing
    
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
        return parser.parse_args()
    
    args = parse_args()
    
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
        "batch_size": 32,
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "tau": 0.005,  # Soft update parameter
        "target_update_freq": 10,  # Hard update frequency (if tau=0)
        "memory_capacity": 100000,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_decay_steps": 100000,
        "use_double_dqn": True,
        "use_prioritized_replay": False,
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
        "device": "cpu" if args.cpu else "cuda"
    }
    
    # Create and run the DQN trainer
    runner = DQNRunner(config, checkpoint_path=args.checkpoint)
    
    print("Starting DQN training. Press CTRL+C to stop.")
    print("Giving you 3 seconds to focus on the game window...")
    time.sleep(3)
    
    # Start training
    runner.train()