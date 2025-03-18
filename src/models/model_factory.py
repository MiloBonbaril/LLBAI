import torch
import os
from src.models.cnn_model import LLBAgent


class ModelFactory:
    """
    Factory class for creating, saving, and loading models.
    
    This centralizes model management and provides utilities for
    model checkpointing and loading.
    """
    
    def __init__(self, config):
        """
        Initialize the model factory.
        
        Args:
            config (dict): Configuration dictionary with model parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu")
        print(f"Using device: {self.device}")
        
    def create_model(self):
        """
        Create a new model instance based on configuration.
        
        Returns:
            LLBAgent: Initialized model on the appropriate device
        """
        # Get parameters from config
        input_shape = self.config.get("input_shape", (4, 144, 256))
        num_actions = self.config.get("num_actions", 10)
        
        # Create model
        model = LLBAgent(input_shape, num_actions)
        model.to(self.device)
        
        return model
    
    def save_checkpoint(self, model, optimizer, episode, rewards, checkpoint_dir='./models'):
        """
        Save a model checkpoint including optimizer state and training info.
        
        Args:
            model (LLBAgent): Model to save
            optimizer: Optimizer used for training
            episode (int): Current episode number
            rewards (list): List of rewards for tracking progress
            checkpoint_dir (str): Directory to save checkpoints
        
        Returns:
            str: Path to the saved checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint filename
        checkpoint_file = os.path.join(checkpoint_dir, f"llb_agent_episode_{episode}.pth")
        
        # Create checkpoint dictionary
        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rewards': rewards,
            'config': self.config
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")
        
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_path, model=None, optimizer=None):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            model (LLBAgent, optional): Model to load weights into
            optimizer (optional): Optimizer to load state into
            
        Returns:
            tuple: (model, optimizer, episode, rewards, config)
        """
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get config from checkpoint
        loaded_config = checkpoint.get('config', {})
        
        # Create new model if not provided
        if model is None:
            # Use loaded config if available, otherwise current config
            config_to_use = loaded_config if loaded_config else self.config
            
            input_shape = config_to_use.get("input_shape", (4, 144, 256))
            num_actions = config_to_use.get("num_actions", 10)
            
            model = LLBAgent(input_shape, num_actions)
            model.to(self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get episode and rewards
        episode = checkpoint.get('episode', 0)
        rewards = checkpoint.get('rewards', [])
        
        return model, optimizer, episode, rewards, loaded_config