import unittest
import os
import sys
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import coverage

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.learning.dqn import DQNTrainer
from src.models.cnn_model import LLBAgent

# Create a coverage object
cov = coverage.Coverage(source=['src.learning.dqn'])


class TestDQNTrainer(unittest.TestCase):
    """Test cases for the DQNTrainer class."""
    
    @classmethod
    def setUpClass(cls):
        """Start coverage tracking before any tests."""
        cov.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop coverage tracking after all tests."""
        cov.stop()
        cov.save()
        print("\nCoverage Report:")
        cov.report(show_missing=True)
        
        # Generate HTML report
        try:
            cov.html_report(directory='htmlcov')
            print("HTML coverage report generated in 'htmlcov' directory")
        except Exception as e:
            print(f"Could not generate HTML report: {e}")
    
    def setUp(self):
        """Set up test environment before each test."""
        # Set up test configuration
        self.config = {
            "batch_size": 8,
            "gamma": 0.99,
            "learning_rate": 0.0001,
            "tau": 0.005,
            "target_update_freq": 5,
            "use_double_dqn": True,
            "use_prioritized_replay": False,
            "clip_grad_norm": 10.0,
            "initial_epsilon": 1.0,
            "final_epsilon": 0.1,
            "epsilon_decay_steps": 1000,
            "input_shape": (4, 84, 84),
            "num_actions": 5,
            "device": "cpu",
            "memory_capacity": 1000,
        }
        
        # Create a small model for testing
        self.input_shape = (4, 84, 84)
        self.num_actions = 5
        self.model = LLBAgent(self.input_shape, self.num_actions)
        
        # Create DQN trainer
        self.trainer = DQNTrainer(self.config, model=self.model)
    
    def test_initialization(self):
        """Test if DQNTrainer initializes correctly."""
        # Check that models were created
        self.assertIsInstance(self.trainer.model, LLBAgent)
        self.assertIsInstance(self.trainer.target_model, LLBAgent)
        
        # Check that optimizer was created
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Optimizer)
        
        # Check that memory was created
        self.assertIsNotNone(self.trainer.memory)
        
        # Check parameters
        self.assertEqual(self.trainer.batch_size, self.config["batch_size"])
        self.assertEqual(self.trainer.gamma, self.config["gamma"])
        self.assertEqual(self.trainer.tau, self.config["tau"])
        self.assertEqual(self.trainer.initial_epsilon, self.config["initial_epsilon"])
        self.assertEqual(self.trainer.epsilon, self.config["initial_epsilon"])
        self.assertEqual(self.trainer.use_double_dqn, self.config["use_double_dqn"])
    
    def test_select_action_random(self):
        """Test random action selection (exploration)."""
        # Create a state tensor
        state = torch.zeros((1, *self.input_shape), dtype=torch.float32)
        
        # Set epsilon to 1.0 for full exploration
        self.trainer.epsilon = 1.0
        
        # Select actions many times
        actions = [self.trainer.select_action(state) for _ in range(100)]
        
        # Check that we get various actions (random)
        unique_actions = set(actions)
        self.assertGreater(len(unique_actions), 1)  # Should get multiple unique actions
        
        # Check action bounds
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.num_actions)
    
    def test_select_action_greedy(self):
        """Test greedy action selection (exploitation)."""
        # Create a state tensor
        state = torch.zeros((1, *self.input_shape), dtype=torch.float32)
        
        # Set epsilon to 0 for full exploitation
        self.trainer.epsilon = 0.0
        
        # Mock the model to return a fixed output
        q_values = torch.tensor([[0.1, 0.3, 0.2, 0.4, 0.1]])
        self.trainer.model.get_q_values = MagicMock(return_value=q_values)
        
        # Select action
        action = self.trainer.select_action(state)
        
        # Check that the action with highest Q-value was selected (action 3)
        self.assertEqual(action, 3)
    
    def test_update_epsilon(self):
        """Test epsilon update function."""
        # Set up test values
        self.trainer.initial_epsilon = 1.0
        self.trainer.final_epsilon = 0.1
        self.trainer.epsilon_decay_steps = 100
        self.trainer.epsilon = 1.0
        
        # Test at beginning (step 0)
        self.trainer.update_epsilon(0)
        self.assertEqual(self.trainer.epsilon, 1.0)
        
        # Test at middle (step 50)
        self.trainer.update_epsilon(50)
        self.assertEqual(self.trainer.epsilon, 0.55)
        
        # Test at end (step 100)
        self.trainer.update_epsilon(100)
        self.assertEqual(self.trainer.epsilon, 0.1)
        
        # Test beyond end (step 200)
        self.trainer.update_epsilon(200)
        self.assertEqual(self.trainer.epsilon, 0.1)
    
    def test_add_experience(self):
        """Test adding experience to memory."""
        # Create test data
        state = np.zeros(self.input_shape, dtype=np.float32)
        action = 1
        reward = 0.5
        next_state = np.ones(self.input_shape, dtype=np.float32)
        done = False
        
        # Add experience
        self.trainer.add_experience(state, action, reward, next_state, done)
        
        # Check that experience was added to memory
        self.assertEqual(len(self.trainer.memory), 1)
        
        # Check episode reward tracking
        self.assertEqual(self.trainer.current_episode_reward, 0.5)
        
        # Add another experience with done=True
        self.trainer.add_experience(next_state, action, reward, state, True)
        
        # Check that episode reward was reset and added to history
        self.assertEqual(self.trainer.current_episode_reward, 0)
        self.assertEqual(len(self.trainer.episode_rewards), 1)
        self.assertEqual(self.trainer.episode_rewards[0], 1.0)  # 0.5 + 0.5
    
    def test_update_target_network_soft(self):
        """Test soft update of target network."""
        # Set up test parameters
        self.trainer.tau = 0.5  # 50% update
        
        # Change model parameters to ensure they're different
        with torch.no_grad():
            for param in self.trainer.model.parameters():
                param.add_(torch.ones_like(param))
        
        # Get original parameter values
        orig_model_params = []
        orig_target_params = []
        
        for param, target_param in zip(self.trainer.model.parameters(), 
                                     self.trainer.target_model.parameters()):
            orig_model_params.append(param.clone())
            orig_target_params.append(target_param.clone())
        
        # Perform soft update
        self.trainer.update_target_network()
        
        # Check that target parameters were updated correctly
        for i, (param, target_param) in enumerate(zip(self.trainer.model.parameters(), 
                                                   self.trainer.target_model.parameters())):
            # Check that target_param = 0.5 * orig_target_param + 0.5 * model_param
            expected = orig_target_params[i] * 0.5 + orig_model_params[i] * 0.5
            self.assertTrue(torch.allclose(target_param, expected))
    
    def test_update_target_network_hard(self):
        """Test hard update of target network."""
        # Set up test parameters
        self.trainer.tau = 0  # Use hard update
        self.trainer.target_update_freq = 2  # Update every 2 steps
        
        # Change model parameters to ensure they're different
        with torch.no_grad():
            for param in self.trainer.model.parameters():
                param.add_(torch.ones_like(param))
        
        # Get original parameters
        orig_model_params = {name: param.clone() for name, param in self.trainer.model.named_parameters()}
        
        # Update once - should not change target yet
        self.trainer.update_target_step = 0
        self.trainer.update_target_network()
        
        # Check target parameters remain unchanged
        for name, target_param in self.trainer.target_model.named_parameters():
            self.assertFalse(torch.allclose(target_param, orig_model_params[name]))
        
        # Update again - should change target now
        self.trainer.update_target_network()
        
        # Check target parameters match model
        for name, target_param in self.trainer.target_model.named_parameters():
            self.assertTrue(torch.allclose(target_param, orig_model_params[name]))
    
    def test_optimize_model_insufficient_samples(self):
        """Test optimization with insufficient samples."""
        # Memory is empty, so should return 0 loss without optimizing
        loss = self.trainer.optimize_model()
        
        # Check loss
        self.assertEqual(loss, 0.0)
    
    def test_optimize_model_basic(self):
        """Test basic optimization with sufficient samples."""
        # Add some experiences to memory
        for i in range(20):
            state = np.zeros(self.input_shape, dtype=np.float32)
            next_state = np.ones(self.input_shape, dtype=np.float32)
            self.trainer.add_experience(state, i % self.num_actions, 0.5, next_state, False)
        
        # Save original model parameters
        orig_params = []
        for param in self.trainer.model.parameters():
            orig_params.append(param.clone())
        
        # Perform optimization
        loss = self.trainer.optimize_model()
        
        # Check that loss is not zero
        self.assertNotEqual(loss, 0.0)
        
        # Check that model parameters changed
        for i, param in enumerate(self.trainer.model.parameters()):
            self.assertFalse(torch.allclose(param, orig_params[i]))
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        # Create a temp directory for checkpoints
        os.makedirs("temp_checkpoints", exist_ok=True)
        self.config["checkpoint_dir"] = "temp_checkpoints"
        
        # Add some random data to ensure it's saved
        self.trainer.train_step = 100
        self.trainer.epsilon = 0.5
        self.trainer.episode_rewards = [1.0, 2.0, 3.0]
        self.trainer.loss_history = [0.5, 0.4, 0.3]
        
        # Save checkpoint
        checkpoint_path = self.trainer.save_checkpoint(episode=1)
        
        # Check that checkpoint file was created
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Modify some values to check that they're restored
        self.trainer.train_step = 0
        self.trainer.epsilon = 1.0
        self.trainer.episode_rewards = []
        self.trainer.loss_history = []
        
        # Load checkpoint
        self.trainer.load_checkpoint(checkpoint_path)
        
        # Check that values were restored
        self.assertEqual(self.trainer.train_step, 100)
        self.assertEqual(self.trainer.epsilon, 0.5)
        self.assertEqual(self.trainer.episode_rewards, [1.0, 2.0, 3.0])
        self.assertEqual(self.trainer.loss_history, [0.5, 0.4, 0.3])
        
        # Clean up
        os.remove(checkpoint_path)
        os.rmdir("temp_checkpoints")


if __name__ == '__main__':
    unittest.main()