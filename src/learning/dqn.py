import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import copy
import math

# Import local modules
from ..models.cnn_model import LLBAgent
from .replay_memory import ReplayMemory, PrioritizedReplayMemory


class DQNTrainer:
    """
    Deep Q-Network (DQN) implementation for training Lethal League Blaze agent.
    
    Implements key DQN features:
    - Experience replay
    - Target network
    - Double DQN (optional)
    - Dueling DQN (via model architecture)
    - Prioritized Experience Replay (optional)
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 model: LLBAgent = None,
                 target_model: LLBAgent = None,
                 optimizer: torch.optim.Optimizer = None):
        """
        Initialize the DQN trainer.
        
        Args:
            config: Configuration dictionary
            model: Main policy network (if None, will be created)
            target_model: Target network for stable training (if None, will be created)
            optimizer: Optimizer (if None, will be created)
        """
        self.config = config
        
        # Get parameters from config
        self.batch_size = config.get("batch_size", 32)
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.tau = config.get("tau", 0.005)  # Soft update parameter
        self.target_update_freq = config.get("target_update_freq", 10)  # Hard update frequency
        self.learning_rate = config.get("learning_rate", 0.0001)
        self.use_double_dqn = config.get("use_double_dqn", True)
        self.use_prioritized_replay = config.get("use_prioritized_replay", False)
        self.clip_grad_norm = config.get("clip_grad_norm", 10.0)
        
        # Exploration parameters
        self.initial_epsilon = config.get("initial_epsilon", 1.0)
        self.final_epsilon = config.get("final_epsilon", 0.01)
        self.epsilon_decay_steps = config.get("epsilon_decay_steps", 100000)
        self.epsilon = self.initial_epsilon
        
        # Device
        self.device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_name)
        print(f"Training on device: {self.device}")
        
        # Model input shape
        self.input_shape = config.get("input_shape", (4, 144, 256))
        self.num_actions = config.get("num_actions", 10)
        
        # Set up models
        if model is None:
            self.model = LLBAgent(self.input_shape, self.num_actions)
            self.model.to(self.device)
        else:
            self.model = model
        
        if target_model is None:
            self.target_model = copy.deepcopy(self.model)
            self.target_model.to(self.device)
        else:
            self.target_model = target_model
        
        # Set target model to evaluation mode
        self.target_model.eval()
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer
        
        # Set up replay buffer
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayMemory(
                capacity=config.get("memory_capacity", 100000),
                alpha=config.get("priority_alpha", 0.6),
                beta=config.get("priority_beta", 0.4)
            )
        else:
            self.memory = ReplayMemory(capacity=config.get("memory_capacity", 100000))
        
        # Training metrics
        self.train_step = 0
        self.update_target_step = 0
        self.loss_history = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = config.get("checkpoint_dir", "data/models")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def select_action(self, state_tensor: torch.Tensor, evaluate: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state_tensor: Current state tensor
            evaluate: If True, use greedy policy (epsilon=0)
            
        Returns:
            int: Selected action index
        """
        # Use greedy policy during evaluation
        if evaluate:
            epsilon = 0.0
        else:
            epsilon = self.epsilon
        
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            # Random action
            return random.randint(0, self.num_actions - 1)
        
        # Greedy action
        with torch.no_grad():
            q_values = self.model.get_q_values(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            
        return action
    
    def update_epsilon(self, step: int = None) -> None:
        """
        Update exploration rate (epsilon) using linear decay.
        
        Args:
            step: Current training step (if None, uses internal counter)
        """
        if step is None:
            step = self.train_step
            
        # Linear decay
        decay_progress = min(1.0, step / self.epsilon_decay_steps)
        self.epsilon = self.initial_epsilon - decay_progress * (self.initial_epsilon - self.final_epsilon)
    
    def add_experience(self, state, action, reward, next_state, done):
        """
        Add experience to replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.add(state, action, reward, next_state, done)
        
        # Update episode reward
        self.current_episode_reward += reward
        
        # Reset episode reward if done
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
    
    def update_target_network(self) -> None:
        """Update target network using either soft or hard update."""
        if self.tau > 0:
            # Soft update (Polyak averaging)
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
        else:
            # Hard update
            self.update_target_step += 1
            if self.update_target_step % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
    
    def optimize_model(self) -> float:
        """
        Perform one optimization step using a batch of experiences.
        
        Returns:
            float: Loss value
        """
        # Check if we have enough samples
        if not self.memory.is_ready(self.batch_size):
            return 0.0
            
        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            batch = self.memory.sample_tensors(self.batch_size, self.device)
            # Extract importance sampling weights
            weights = batch['weights']
        else:
            batch = self.memory.sample_tensors(self.batch_size, self.device)
            # No importance sampling weights for regular replay
            weights = torch.ones(self.batch_size, device=self.device)
        
        # Extract batch data
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        # Remove extra dimension if present
        if states.dim() == 5:
            states = states.squeeze(1)
        if next_states.dim() == 5:
            next_states = next_states.squeeze(1)
        
        # Compute current Q values
        q_values = self.model.get_q_values(states)
        # Select Q values for taken actions
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select actions
                next_actions = self.model.get_q_values(next_states).argmax(1, keepdim=True)
                # Use target network to evaluate actions
                next_q_values = self.target_model.get_q_values(next_states)
                next_q_values = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_model.get_q_values(next_states).max(1)[0]
            
            # Set Q value of terminal states to 0
            next_q_values[dones] = 0.0
            
            # Compute target Q values using Bellman equation
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss
        td_errors = q_values - target_q_values
        
        # Weighted MSE loss for prioritized replay
        loss = (weights * td_errors.pow(2)).mean()
        
        # Update priorities in prioritized replay
        if self.use_prioritized_replay:
            priorities = torch.abs(td_errors).detach().cpu().numpy()
            self.memory.update_priorities(batch['indices'], priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
        self.optimizer.step()
        
        # Update target network
        self.update_target_network()
        
        # Increment step counter
        self.train_step += 1
        
        # Update exploration rate
        self.update_epsilon()
        
        # Save loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def train_batch(self, batch_size: Optional[int] = None) -> float:
        """
        Train on a single batch of experiences.
        
        Args:
            batch_size: Batch size (if None, use default)
            
        Returns:
            float: Loss value
        """
        if batch_size is not None:
            old_batch_size = self.batch_size
            self.batch_size = batch_size
            loss = self.optimize_model()
            self.batch_size = old_batch_size
        else:
            loss = self.optimize_model()
            
        return loss
    
    def save_checkpoint(self, episode: int, rewards: Optional[List[float]] = None) -> str:
        """
        Save a model checkpoint.
        
        Args:
            episode: Current episode number
            rewards: List of episode rewards
            
        Returns:
            str: Path to the saved checkpoint
        """
        if rewards is None:
            rewards = self.episode_rewards
            
        checkpoint_path = os.path.join(self.checkpoint_dir, f"dqn_checkpoint_ep{episode}.pth")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'episode_rewards': rewards,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            dict: Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and target model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.train_step = checkpoint.get('train_step', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.loss_history = checkpoint.get('loss_history', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        
        # Update config with loaded config if available
        if 'config' in checkpoint:
            # Only update values that are in the loaded config
            for key, value in checkpoint['config'].items():
                if key in self.config:
                    self.config[key] = value
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint