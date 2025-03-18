import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LLBAgent(nn.Module):
    """
    CNN-based neural network model for Lethal League Blaze agent.
    
    Takes stacked game frames as input and outputs action probabilities
    and state value estimation.
    """
    
    def __init__(self, input_shape, num_actions):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Input shape as (stack_size, height, width)
            num_actions (int): Number of possible actions the agent can take
        """
        super(LLBAgent, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions for the fully connected layer
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        
        # Action (policy) head
        self.policy_head = nn.Linear(512, num_actions)
        
        # Value head (for actor-critic methods)
        self.value_head = nn.Linear(512, 1)
    
    def _get_conv_output_size(self):
        """Calculate the size of the flattened output from the conv layers."""
        # Create a dummy input tensor
        x = torch.zeros(1, *self.input_shape)
        
        # Forward pass through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Get the flattened size
        return int(np.prod(x.size()))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, stack_size, height, width)
            
        Returns:
            tuple: (action_logits, state_value)
                - action_logits: Action probabilities before softmax
                - state_value: Estimated value of the input state
        """
        # Ensure proper dimensions
        if len(x.size()) == 3:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
        
        # Conv layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = F.relu(self.fc1(x))
        
        # Policy head (action probabilities)
        action_logits = self.policy_head(x)
        
        # Value head (state value estimation)
        state_value = self.value_head(x)
        
        return action_logits, state_value
    
    def get_action_probs(self, x):
        """
        Get action probabilities from the policy head.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Action probabilities after softmax
        """
        action_logits, _ = self.forward(x)
        return F.softmax(action_logits, dim=1)
    
    def get_q_values(self, x):
        """
        Get Q-values for DQN training.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        action_logits, _ = self.forward(x)
        return action_logits  # For DQN, logits directly represent Q-values
    
    def save(self, file_path):
        """Save model weights to file."""
        torch.save(self.state_dict(), file_path)
    
    def load(self, file_path, device='cpu'):
        """Load model weights from file."""
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.eval()