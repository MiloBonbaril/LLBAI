import unittest
import os
import sys
import numpy as np
import torch
from unittest.mock import MagicMock, patch
import coverage

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.learning.agent_runner import AgentRunner

# Create a coverage object
cov = coverage.Coverage(source=['src.learning.agent_runner'])


class TestAgentRunner(unittest.TestCase):
    """Test cases for the AgentRunner class."""
    
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
            #cov.html_report(directory='htmlcov')
            print("HTML coverage report generated in 'htmlcov' directory")
        except Exception as e:
            print(f"Could not generate HTML report: {e}")
    
    def setUp(self):
        """Set up test environment before each test."""
        # Mock dependencies
        self.mock_game_interface = MagicMock()
        self.mock_frame_processor = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model_factory = MagicMock()
        
        # Mock return values
        self.mock_game_interface.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_frame_processor.add_to_stack.return_value = np.zeros((4, 144, 256), dtype=np.float32)
        self.mock_frame_processor.stack_size = 4
        self.mock_model.get_action_probs.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        self.mock_model_factory.create_model.return_value = self.mock_model
        self.mock_model_factory.load_checkpoint.return_value = (self.mock_model, None, None, None, None)
        
        # Test configuration
        self.config = {
            "window_name": "Test Window",
            "fps_limit": 30,
            "frame_height": 144,
            "frame_width": 256,
            "stack_size": 4,
            "num_actions": 10,
            "device": "cpu"
        }
        
        # Create patches
        self.patches = [
            patch('src.learning.agent_runner.GameInterface', return_value=self.mock_game_interface),
            patch('src.learning.agent_runner.FrameProcessor', return_value=self.mock_frame_processor),
            patch('src.learning.agent_runner.ModelFactory', return_value=self.mock_model_factory)
        ]
        
        # Start patches
        for p in self.patches:
            p.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_initialization(self):
        """Test if AgentRunner initializes correctly."""
        # Test with no model path
        agent = AgentRunner(self.config)
        
        # Check that components were initialized correctly
        self.mock_game_interface.assert_called_once()
        self.mock_frame_processor.assert_called_once()
        self.mock_model_factory.assert_called_once()
        self.mock_model_factory.create_model.assert_called_once()
        
        # Check action map
        self.assertEqual(len(agent.action_map), 10)
        self.assertEqual(agent.action_map[0], [])
        self.assertEqual(agent.action_map[5], ["JUMP", "ATTACK"])
    
    def test_initialization_with_model(self):
        """Test initialization with a model path."""
        # Create a temporary file to simulate a model checkpoint
        temp_model_path = "temp_model.pth"
        with open(temp_model_path, "w") as f:
            f.write("dummy model data")
            
        try:
            # Test with model path
            agent = AgentRunner(self.config, model_path=temp_model_path)
            
            # Check that model was loaded
            self.mock_model_factory.load_checkpoint.assert_called_once_with(temp_model_path)
            
        finally:
            # Clean up
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
    
    def test_preprocess_frame(self):
        """Test frame preprocessing."""
        agent = AgentRunner(self.config)
        
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        result = agent.preprocess_frame(test_frame)
        
        # Check that frame processor was called
        self.mock_frame_processor.add_to_stack.assert_called_once()
        
        # Check result is a tensor with correct shape
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 4, 144, 256))
    
    def test_select_action_random(self):
        """Test random action selection."""
        agent = AgentRunner(self.config, exploration_rate=1.0)
        
        # Create a state tensor
        state_tensor = torch.zeros((1, 4, 144, 256))
        
        # Test selecting an action with 100% exploration
        action = agent.select_action(state_tensor)
        
        # Check that action is within valid range
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, len(agent.action_map))
        
        # The model should not have been called (because we used random action)
        self.mock_model.get_action_probs.assert_not_called()
    
    def test_select_action_model(self):
        """Test model-based action selection."""
        agent = AgentRunner(self.config, exploration_rate=0.0)
        agent.model = self.mock_model  # Ensure we use our mock
        
        # Create a state tensor
        state_tensor = torch.zeros((1, 4, 144, 256))
        
        # Test selecting an action with 0% exploration
        action = agent.select_action(state_tensor)
        
        # Check that model was called
        self.mock_model.get_action_probs.assert_called_once()
        
        # Since our mock returns increasing probabilities, it should select the last action
        self.assertEqual(action, 9)
    
    def test_execute_action(self):
        """Test executing an action."""
        agent = AgentRunner(self.config)
        
        # Test executing an action
        agent.execute_action(5, duration=0.1)
        
        # Check that game interface was called with correct action
        self.mock_game_interface.take_action.assert_called_once_with(["JUMP", "ATTACK"], 0.1)
    
    def test_run_episode(self):
        """Test running an episode."""
        agent = AgentRunner(self.config)
        
        # Test running a short episode (10 steps)
        stats = agent.run_episode(max_steps=10, render=False)
        
        # Check that frame processor was reset
        self.mock_frame_processor.reset.assert_called_once()
        
        # Check that get_frame was called at least 10 times
        self.assertGreaterEqual(self.mock_game_interface.get_frame.call_count, 10)
        
        # Check that stats were returned
        self.assertIn("episode", stats)
        self.assertIn("steps", stats)
        self.assertIn("time", stats)
        self.assertIn("avg_fps", stats)
    
    @patch('time.sleep', return_value=None)  # Mock time.sleep to speed up test
    def test_run(self, mock_sleep):
        """Test running multiple episodes."""
        agent = AgentRunner(self.config)
        
        # Create a shorter run_episode method to speed up testing
        def mock_run_episode(max_steps, render):
            return {"episode": agent.episode_count, "steps": 5, "time": 1.0, "avg_fps": 5.0}
            
        agent.run_episode = mock_run_episode
        
        # Test running 3 episodes
        stats = agent.run(num_episodes=3, max_steps=5, render=False)
        
        # Check that we got stats for 3 episodes
        self.assertEqual(len(stats), 3)
        
        # Check that agent's episode count was updated
        self.assertEqual(agent.episode_count, 3)
    
    def test_close(self):
        """Test closing the agent."""
        agent = AgentRunner(self.config)
        
        # Test closing
        agent.close()
        
        # Check that game interface was closed
        self.mock_game_interface.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()