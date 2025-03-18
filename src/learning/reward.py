import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time


class RewardSystem:
    """
    Calculates rewards for the Lethal League Blaze agent based on game state.
    
    This class defines the reward function for reinforcement learning,
    providing feedback to the agent about its actions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward system.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        self.config = config
        
        # Get reward values from config with defaults
        self.survival_reward = config.get("survival_reward", 0.01)
        self.hit_ball_reward = config.get("hit_ball_reward", 1.0)
        self.miss_ball_penalty = config.get("miss_ball_penalty", -0.5)
        self.score_point_reward = config.get("score_point_reward", 5.0)
        self.lost_point_penalty = config.get("lost_point_penalty", -3.0)
        self.game_win_reward = config.get("game_win_reward", 10.0)
        self.game_loss_penalty = config.get("game_loss_penalty", -5.0)
        
        # State tracking
        self.last_frame = None
        self.last_time = time.time()
        self.score_history = []
        self.hit_history = []
        self.current_score = [0, 0]  # [player, opponent]
        
        # Ball detection parameters
        self.ball_detection_enabled = config.get("ball_detection_enabled", True)
        self.ball_color_lower = np.array(config.get("ball_color_lower", [180, 180, 0]), dtype=np.uint8)
        self.ball_color_upper = np.array(config.get("ball_color_upper", [255, 255, 100]), dtype=np.uint8)
        
        # Score detection parameters
        self.score_detection_enabled = config.get("score_detection_enabled", True)
        self.score_region = config.get("score_region", [10, 10, 100, 30])  # [x, y, width, height]
        
        # Debug mode
        self.debug = config.get("debug_reward", False)
    
    def reset(self):
        """Reset the reward system state."""
        self.last_frame = None
        self.last_time = time.time()
        self.score_history = []
        self.hit_history = []
        self.current_score = [0, 0]
    
    def calculate_reward(self, frame: np.ndarray, action: int, 
                          done: bool = False, win: bool = False) -> float:
        """
        Calculate reward based on the current game state.
        
        Args:
            frame: Current game frame
            action: Action taken by the agent
            done: Whether the episode is done
            win: Whether the agent won (if done)
            
        Returns:
            float: Calculated reward
        """
        total_reward = 0.0
        
        # Basic survival reward
        time_diff = time.time() - self.last_time
        survival_reward = self.survival_reward * min(time_diff, 0.5)  # Cap at 0.5 seconds
        total_reward += survival_reward
        
        # Detect ball hits (if enabled)
        if self.ball_detection_enabled and self.last_frame is not None:
            hit_reward = self._detect_ball_hit(self.last_frame, frame, action)
            total_reward += hit_reward
        
        # Detect score changes (if enabled)
        if self.score_detection_enabled:
            score_reward = self._detect_score_change(frame)
            total_reward += score_reward
        
        # Terminal rewards
        if done:
            if win:
                total_reward += self.game_win_reward
            else:
                total_reward += self.game_loss_penalty
        
        # Update state
        self.last_frame = frame.copy()
        self.last_time = time.time()
        
        # Debug output
        if self.debug:
            print(f"Reward: {total_reward:.4f} (survival={survival_reward:.4f})")
        
        return total_reward
    
    def _detect_ball_hit(self, prev_frame: np.ndarray, curr_frame: np.ndarray, action: int) -> float:
        """
        Detect if the agent hit the ball.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            action: Action taken by the agent
            
        Returns:
            float: Reward for hitting the ball or penalty for missing
        """
        # This is a simplified ball hit detection
        # For actual implementation, we would need more sophisticated computer vision
        
        # Only check for hits if an attack action was taken
        if action not in [4, 5, 8, 9]:  # ATTACK, JUMP+ATTACK, GRAB, BUNT
            return 0.0
            
        try:
            # Convert frames to HSV for better color detection
            prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
            
            # Create masks for ball color
            prev_mask = cv2.inRange(prev_hsv, self.ball_color_lower, self.ball_color_upper)
            curr_mask = cv2.inRange(curr_hsv, self.ball_color_lower, self.ball_color_upper)
            
            # Calculate ball areas
            prev_ball_area = np.sum(prev_mask > 0)
            curr_ball_area = np.sum(curr_mask > 0)
            
            # Calculate ball movement
            prev_ball_moments = cv2.moments(prev_mask)
            curr_ball_moments = cv2.moments(curr_mask)
            
            # If ball is visible in both frames
            if prev_ball_moments["m00"] > 0 and curr_ball_moments["m00"] > 0:
                # Calculate ball centers
                prev_ball_x = int(prev_ball_moments["m10"] / prev_ball_moments["m00"])
                prev_ball_y = int(prev_ball_moments["m01"] / prev_ball_moments["m00"])
                
                curr_ball_x = int(curr_ball_moments["m10"] / curr_ball_moments["m00"])
                curr_ball_y = int(curr_ball_moments["m01"] / curr_ball_moments["m00"])
                
                # Calculate ball movement
                movement_x = abs(curr_ball_x - prev_ball_x)
                movement_y = abs(curr_ball_y - prev_ball_y)
                
                # Check for significant ball movement after attack
                if movement_x > 20 or movement_y > 20:
                    # Ball was hit
                    self.hit_history.append(1)
                    return self.hit_ball_reward
            
            # If we took an attack action but didn't hit the ball
            self.hit_history.append(0)
            return self.miss_ball_penalty
            
        except Exception as e:
            # If there's an error in detection, don't give any reward
            print(f"Error in ball hit detection: {e}")
            return 0.0
    
    def _detect_score_change(self, frame: np.ndarray) -> float:
        """
        Detect changes in the game score.
        
        Args:
            frame: Current game frame
            
        Returns:
            float: Reward for scoring or penalty for being scored on
        """
        # This is a placeholder for score detection
        # For actual implementation, we would need OCR or template matching
        
        # Extract score region
        x, y, w, h = self.score_region
        score_region = frame[y:y+h, x:x+w]
        
        # Dummy implementation: detect score change by frame differences
        # In a real implementation, we would use OCR to read the score
        
        # For now, return 0 (no score change detected)
        return 0.0
    
    def get_score(self) -> Tuple[int, int]:
        """
        Get the current game score.
        
        Returns:
            tuple: (player_score, opponent_score)
        """
        return tuple(self.current_score)
    
    def get_hit_rate(self) -> float:
        """
        Get the ball hit success rate.
        
        Returns:
            float: Hit rate (0.0 to 1.0)
        """
        if not self.hit_history:
            return 0.0
        
        return sum(self.hit_history) / len(self.hit_history)


class ManualRewardSystem(RewardSystem):
    """
    Manual reward system that gets rewards from human input.
    
    Useful for initial training when automatic detection is difficult.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the manual reward system."""
        super().__init__(config)
        
        self.manual_rewards = {
            'hit': config.get("manual_hit_reward", 1.0),
            'miss': config.get("manual_miss_penalty", -0.5),
            'score': config.get("manual_score_reward", 5.0),
            'lost': config.get("manual_lost_penalty", -3.0),
            'win': config.get("manual_win_reward", 10.0),
            'lose': config.get("manual_lose_penalty", -5.0)
        }
        
        print("Manual reward system initialized. Use the following keys during training:")
        print("  h: Ball hit")
        print("  m: Ball miss")
        print("  s: Scored a point")
        print("  l: Lost a point")
        print("  w: Won the game")
        print("  d: Lost the game")
    
    def process_key(self, key: str) -> float:
        """
        Process a key press and return the corresponding reward.
        
        Args:
            key: Key pressed by the user
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        if key == 'h':
            reward = self.manual_rewards['hit']
            self.hit_history.append(1)
            print(f"Ball hit! Reward: {reward}")
        elif key == 'm':
            reward = self.manual_rewards['miss']
            self.hit_history.append(0)
            print(f"Ball miss! Reward: {reward}")
        elif key == 's':
            reward = self.manual_rewards['score']
            self.current_score[0] += 1
            self.score_history.append((self.current_score[0], self.current_score[1]))
            print(f"Scored a point! Reward: {reward}, Score: {self.current_score}")
        elif key == 'l':
            reward = self.manual_rewards['lost']
            self.current_score[1] += 1
            self.score_history.append((self.current_score[0], self.current_score[1]))
            print(f"Lost a point! Reward: {reward}, Score: {self.current_score}")
        elif key == 'w':
            reward = self.manual_rewards['win']
            print(f"Won the game! Reward: {reward}")
        elif key == 'd':
            reward = self.manual_rewards['lose']
            print(f"Lost the game! Reward: {reward}")
        
        return reward
    
    def calculate_reward(self, frame: np.ndarray, action: int, 
                          done: bool = False, win: bool = False) -> float:
        """
        Calculate basic survival reward, manual rewards must be added separately.
        
        Args:
            frame: Current game frame
            action: Action taken by the agent
            done: Whether the episode is done
            win: Whether the agent won (if done)
            
        Returns:
            float: Basic survival reward
        """
        # Only calculate survival reward, manual rewards will be added via process_key
        time_diff = time.time() - self.last_time
        survival_reward = self.survival_reward * min(time_diff, 0.5)  # Cap at 0.5 seconds
        
        # Update state
        self.last_frame = frame.copy()
        self.last_time = time.time()
        
        # Debug output
        if self.debug:
            print(f"Survival reward: {survival_reward:.4f}")
        
        return survival_reward