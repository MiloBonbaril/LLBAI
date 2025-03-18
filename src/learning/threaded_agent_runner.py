import os
import time
import numpy as np
import torch
import random
import threading
import queue
from typing import Dict, List, Tuple, Optional, Union, Any

# Import local modules
from src.game_interface.game_interface import GameInterface
from src.preprocessing.frame_processor import FrameProcessor
from src.models.cnn_model import LLBAgent
from src.models.model_factory import ModelFactory


class ThreadedAgentRunner:
    """
    Multi-threaded implementation of the agent runner.
    Uses separate threads for:
    1. Frame capture
    2. Frame processing
    3. Model inference
    4. Action execution
    
    This reduces bottlenecks and improves FPS by parallelizing operations.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 model_path: Optional[str] = None,
                 exploration_rate: float = 1.0,
                 queue_size: int = 10):
        """
        Initialize the threaded agent runner.
        
        Args:
            config: Configuration dictionary
            model_path: Path to a saved model
            exploration_rate: Probability of taking random actions
            queue_size: Size of the queues between threads
        """
        self.config = config
        self.exploration_rate = exploration_rate
        self.queue_size = queue_size
        
        # Set up components
        window_name = config.get("window_name", "Lethal League Blaze")
        fps_limit = config.get("fps_limit", 30)
        self.game_interface = GameInterface(window_name=window_name, fps_limit=fps_limit)
        
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
        
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"Using device: {self.device}")
        
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
        
        # Set up queues for inter-thread communication
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.state_queue = queue.Queue(maxsize=queue_size)
        self.action_queue = queue.Queue(maxsize=queue_size)
        
        # Set up threading control
        self.running = False
        self.threads = []
        
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.frame_capture_fps = 0
        self.frame_processing_fps = 0
        self.action_selection_fps = 0
        
        # Create necessary directories
        os.makedirs("data/recordings", exist_ok=True)
    
    def _frame_capture_thread(self):
        """Thread function for capturing frames from the game."""
        last_time = time.time()
        frames_captured = 0
        
        # Create a dedicated game interface for this thread
        window_name = self.config.get("window_name", "Lethal League Blaze")
        fps_limit = self.config.get("fps_limit", 30)
        thread_game_interface = GameInterface(window_name=window_name, fps_limit=fps_limit)
        
        print("Initialized dedicated screen capture for capture thread")
        
        try:
            while self.running:
                try:
                    # Capture frame using the thread-local game interface
                    frame = thread_game_interface.get_frame()
                    
                    # Put frame in queue, with a timeout to prevent blocking forever
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                        frames_captured += 1
                        
                        # Calculate and update FPS every second
                        if time.time() - last_time >= 1.0:
                            self.frame_capture_fps = frames_captured / (time.time() - last_time)
                            frames_captured = 0
                            last_time = time.time()
                    except queue.Full:
                        # Queue is full, skip this frame
                        pass
                        
                except Exception as e:
                    print(f"Error in frame capture thread: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        finally:
            # Clean up the thread-local resources
            thread_game_interface.close()
            print("Closed thread-local game interface")
    
    def _frame_processing_thread(self):
        """Thread function for processing frames."""
        last_time = time.time()
        frames_processed = 0
        
        while self.running:
            try:
                # Get frame from queue with timeout
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    # No frame available, try again
                    continue
                
                # Process frame
                state_tensor = self.preprocess_frame(frame)
                
                # Put processed state in queue
                try:
                    self.state_queue.put(state_tensor, timeout=0.1)
                    frames_processed += 1
                    
                    # Calculate and update FPS every second
                    if time.time() - last_time >= 1.0:
                        self.frame_processing_fps = frames_processed / (time.time() - last_time)
                        frames_processed = 0
                        last_time = time.time()
                except queue.Full:
                    # Queue is full, skip this state
                    pass
                    
                # Mark frame as processed
                self.frame_queue.task_done()
                
            except Exception as e:
                print(f"Error in frame processing thread: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def _action_selection_thread(self):
        """Thread function for selecting actions based on processed states."""
        last_time = time.time()
        actions_selected = 0
        
        while self.running:
            try:
                # Get processed state from queue with timeout
                try:
                    state_tensor = self.state_queue.get(timeout=0.1)
                except queue.Empty:
                    # No state available, try again
                    continue
                
                # Select action
                action_idx = self.select_action(state_tensor)
                
                # Put action in queue
                try:
                    self.action_queue.put(action_idx, timeout=0.1)
                    actions_selected += 1
                    
                    # Calculate and update FPS every second
                    if time.time() - last_time >= 1.0:
                        self.action_selection_fps = actions_selected / (time.time() - last_time)
                        actions_selected = 0
                        last_time = time.time()
                except queue.Full:
                    # Queue is full, skip this action
                    pass
                    
                # Mark state as processed
                self.state_queue.task_done()
                
            except Exception as e:
                print(f"Error in action selection thread: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def _action_execution_thread(self):
        """Thread function for executing actions in the game."""
        while self.running:
            try:
                # Get action from queue with timeout
                try:
                    action_idx = self.action_queue.get(timeout=0.1)
                except queue.Empty:
                    # No action available, try again
                    continue
                
                # Execute action
                self.execute_action(action_idx)
                
                # Update statistics
                self.step_count += 1
                self.episode_steps += 1
                
                # Mark action as executed
                self.action_queue.task_done()
                
            except Exception as e:
                print(f"Error in action execution thread: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def _stats_thread(self):
        """Thread function for printing statistics."""
        while self.running:
            try:
                # Print stats every second
                time.sleep(1.0)
                
                # Calculate overall FPS (limited by the slowest component)
                fps = min(self.frame_capture_fps, self.frame_processing_fps, self.action_selection_fps)
                
                # Print current statistics
                print(f"Step: {self.step_count}, FPS: {fps:.1f}")
                print(f"  Capture: {self.frame_capture_fps:.1f}, Process: {self.frame_processing_fps:.1f}, Action: {self.action_selection_fps:.1f}")
                print(f"  Queue sizes - Frames: {self.frame_queue.qsize()}, States: {self.state_queue.qsize()}, Actions: {self.action_queue.qsize()}")
                
            except Exception as e:
                print(f"Error in stats thread: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a raw frame and convert to PyTorch tensor."""
        # Add to frame stack
        processed_stack = self.frame_processor.add_to_stack(frame)
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(processed_stack, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def select_action(self, state_tensor: torch.Tensor) -> int:
        """Select an action based on current state."""
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Take random action
            return random.randint(0, len(self.action_map) - 1)
        
        # Use model to select action
        with torch.no_grad():
            try:
                # Try the specific get_action_probs method first
                if hasattr(self.model, 'get_action_probs'):
                    action_probs = self.model.get_action_probs(state_tensor)
                else:
                    # Fall back to standard model forward pass
                    output = self.model(state_tensor)
                    
                    # Handle different output types
                    if isinstance(output, tuple):
                        # Some models return (logits, _) or similar
                        action_probs = torch.softmax(output[0], dim=1)
                    else:
                        # Assume output is logits
                        action_probs = torch.softmax(output, dim=1)
            except Exception as e:
                print(f"Error during model inference: {e}")
                # If model inference fails, take a random action
                return random.randint(0, len(self.action_map) - 1)
        
        # Select action with highest probability
        action = torch.argmax(action_probs, dim=1).item()
        return action
    
    def execute_action(self, action_idx: int, duration: float = 0.1) -> None:
        """Execute the selected action in the game."""
        if action_idx in self.action_map:
            action = self.action_map[action_idx]
            self.game_interface.take_action(action, duration)
    
    def start_threads(self):
        """Start all worker threads."""
        self.running = True
        
        # Create threads
        self.threads = [
            threading.Thread(target=self._frame_capture_thread, name="FrameCapture"),
            threading.Thread(target=self._frame_processing_thread, name="FrameProcessing"),
            threading.Thread(target=self._action_selection_thread, name="ActionSelection"),
            threading.Thread(target=self._action_execution_thread, name="ActionExecution"),
            threading.Thread(target=self._stats_thread, name="Statistics")
        ]
        
        # Start threads
        for thread in self.threads:
            thread.daemon = True  # Threads will exit when main thread exits
            thread.start()
        
        print(f"Started {len(self.threads)} worker threads")
    
    def stop_threads(self):
        """Stop all worker threads."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)  # Wait up to 2 seconds for each thread
        
        print("Stopped all worker threads")
    
    def run_episode(self, max_steps: int = 1000, max_time: int = 300) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            max_steps: Maximum number of steps per episode
            max_time: Maximum episode time in seconds
            
        Returns:
            dict: Episode statistics
        """
        self.episode_steps = 0
        self.episode_start_time = time.time()
        
        print(f"Starting episode {self.episode_count + 1}")
        
        try:
            # Reset frame processor
            self.frame_processor.reset()
            
            # Start threads if not already running
            if not self.running:
                self.start_threads()
            
            # Wait until max steps or time limit
            while self.episode_steps < max_steps and time.time() - self.episode_start_time < max_time:
                time.sleep(0.1)  # Check periodically
                
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
    
    def run(self, num_episodes: int = 5, max_steps: int = 1000, max_time: int = 300) -> List[Dict[str, Any]]:
        """
        Run multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            max_time: Maximum episode time in seconds
            
        Returns:
            list: Statistics for each episode
        """
        episode_stats = []
        
        try:
            for i in range(num_episodes):
                stats = self.run_episode(max_steps, max_time)
                episode_stats.append(stats)
                
                # Short delay between episodes
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("Run interrupted by user")
            
        finally:
            # Stop threads
            self.stop_threads()
            
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
        # Stop threads
        self.stop_threads()
        
        # Close game interface
        self.game_interface.close()
        print("Threaded agent runner closed")


if __name__ == "__main__":
    import argparse
    
    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Run Threaded Lethal League Blaze Agent")
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
    
    args = parse_args()
    
    # Configuration dictionary
    config = {
        "window_name": args.window,
        "fps_limit": args.fps,
        "frame_height": 144,
        "frame_width": 256,
        "stack_size": 4,
        "num_actions": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Create and run agent
    agent = ThreadedAgentRunner(
        config=config,
        model_path=args.model,
        exploration_rate=args.exploration
    )
    
    print("Starting threaded agent runner. Press Ctrl+C to stop.")
    print("Giving you 3 seconds to focus on the game window...")
    time.sleep(3)
    
    # Run episodes
    agent.run(num_episodes=args.episodes, max_steps=args.steps)
    
    # Clean up
    agent.close()