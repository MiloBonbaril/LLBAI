import numpy as np
import random
from collections import deque, namedtuple
import torch
from typing import List, Tuple, Dict, Any, Union, Optional

# Define the Experience tuple structure
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayMemory:
    """
    Experience replay buffer for DQN training.
    
    Stores experiences as (state, action, reward, next_state, done) tuples
    and provides methods for adding and sampling experiences.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state (tensor or array)
            action: Action taken
            reward: Reward received
            next_state: Next state (tensor or array)
            done: Whether the episode is done
        """
        # Convert tensors to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Create and add experience
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            list: Batch of experiences
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def sample_tensors(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch and convert to PyTorch tensors.
        
        Args:
            batch_size: Number of experiences to sample
            device: PyTorch device to place tensors on
            
        Returns:
            dict: Dictionary containing batched tensors for states, actions, rewards, etc.
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        # Sample experiences
        experiences = self.sample(batch_size)
        
        # Extract and batch components
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.tensor(np.array(batch.state), device=device)
        action_batch = torch.tensor(batch.action, device=device, dtype=torch.long)
        reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(batch.next_state), device=device)
        done_batch = torch.tensor(batch.done, device=device, dtype=torch.bool)
        
        return {
            'states': state_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'next_states': next_state_batch,
            'dones': done_batch
        }
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.
        
        Args:
            batch_size: Minimum number of samples required
            
        Returns:
            bool: True if buffer has enough samples
        """
        return len(self.buffer) >= batch_size


class PrioritizedReplayMemory(ReplayMemory):
    """
    Prioritized experience replay buffer for DQN training.
    
    Samples experiences based on TD error priority, giving higher
    probability to experiences with higher TD error.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, higher = more prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        super().__init__(capacity)
        
        # Replace deque with list for indexed access
        self.buffer = []
        
        # Priority parameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # For annealing beta to 1
        
        # Small constant to avoid zero priority
        self.epsilon = 1e-6
        
        # Storage for priorities
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # Current position in buffer
    
    def add(self, state, action, reward, next_state, done, priority=None):
        """
        Add an experience to the buffer with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            priority: Priority value (if None, max priority is used)
        """
        # Convert tensors to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Create experience
        experience = Experience(state, action, reward, next_state, done)
        
        # Get max priority for new experience
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        # Use provided priority or max priority
        priority = priority if priority is not None else max_priority
        
        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        # Update priority
        self.priorities[self.position] = priority
        
        # Update position
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], torch.Tensor]:
        """
        Sample a batch of experiences based on priority.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            tuple: (experiences, indices, importance_weights)
        """
        # Ensure we have enough experiences
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            batch_size = buffer_size
        
        # Calculate sampling probabilities
        priorities = self.priorities[:buffer_size] ** self.alpha
        probs = priorities / np.sum(priorities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(buffer_size, batch_size, replace=False, p=probs)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance weights
        weights = (buffer_size * probs[indices]) ** -self.beta
        weights /= np.max(weights)  # Normalize
        
        # Convert weights to tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights_tensor
    
    def sample_tensors(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Sample a batch and convert to tensors with importance weights.
        
        Args:
            batch_size: Number of experiences to sample
            device: PyTorch device
            
        Returns:
            dict: Dictionary with batched tensors and weights
        """
        # Sample experiences and get indices and weights
        experiences, indices, weights = self.sample(batch_size)
        
        # Extract and batch components
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.tensor(np.array(batch.state), device=device)
        action_batch = torch.tensor(batch.action, device=device, dtype=torch.long)
        reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(batch.next_state), device=device)
        done_batch = torch.tensor(batch.done, device=device, dtype=torch.bool)
        
        return {
            'states': state_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'next_states': next_state_batch,
            'dones': done_batch,
            'indices': indices,
            'weights': weights.to(device)
        }
    
    def update_priorities(self, indices: List[int], priorities: Union[List[float], np.ndarray, torch.Tensor]):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: List of experience indices
            priorities: List of new priority values
        """
        # Convert tensor to numpy if needed
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.cpu().numpy()
        
        # Add small constant to prevent zero priority
        priorities = np.abs(priorities) + self.epsilon
        
        # Update priorities
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)