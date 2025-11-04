"""
Checkpoint functionality for saving and loading agent state.

This allows for resuming runs after crashes or interruptions.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from .schemas import GameActionRecord, Cost, Usage, CompletionTokensDetails

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing of agent state"""
    
    CHECKPOINT_DIR = ".checkpoint"
    
    def __init__(self, card_id: str):
        """
        Initialize checkpoint manager.
        
        Args:
            card_id: Scorecard ID to use as checkpoint directory name
        """
        self.card_id = card_id
        self.checkpoint_path = Path(self.CHECKPOINT_DIR) / card_id
        
    def save_state(
        self,
        config: str,
        game_id: str,
        guid: Optional[str],
        max_actions: int,
        retry_attempts: int,
        num_plays: int,
        action_counter: int,
        total_cost: Cost,
        total_usage: Usage,
        action_history: List[GameActionRecord],
        memory_prompt: str,
        previous_action: Optional[Dict[str, Any]],
        previous_images: List[Image.Image],
        previous_score: int,
        current_play: int = 1,
        play_action_counter: int = 0,
    ):
        """
        Save complete agent state to checkpoint.
        
        Args:
            config: Model configuration name
            game_id: Current game ID
            guid: Current session GUID
            max_actions: Maximum actions setting
            retry_attempts: Retry attempts setting
            num_plays: Number of plays setting
            action_counter: Total action counter
            total_cost: Total cost so far
            total_usage: Total token usage so far
            action_history: Complete action history
            memory_prompt: Current memory/conversation state
            previous_action: Previous action taken
            previous_images: Previous frame images
            previous_score: Previous score
            current_play: Current play number
            play_action_counter: Actions in current play
        """
        logger.info(f"Saving checkpoint to {self.checkpoint_path}")
        
        # Create checkpoint directory
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "card_id": self.card_id,
            "config": config,
            "game_id": game_id,
            "guid": guid,
            "max_actions": max_actions,
            "retry_attempts": retry_attempts,
            "num_plays": num_plays,
            "action_counter": action_counter,
            "current_play": current_play,
            "play_action_counter": play_action_counter,
            "previous_score": previous_score,
            "checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(self.checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save costs and usage
        costs = {
            "total_cost": total_cost.model_dump(),
            "total_usage": total_usage.model_dump(),
        }
        
        with open(self.checkpoint_path / "costs.json", "w") as f:
            json.dump(costs, f, indent=2)
        
        # Save action history
        action_history_data = [action.model_dump() for action in action_history]
        with open(self.checkpoint_path / "action_history.json", "w") as f:
            json.dump(action_history_data, f, indent=2)
        
        # Save memory
        with open(self.checkpoint_path / "memory.txt", "w") as f:
            f.write(memory_prompt)
        
        # Save previous action
        if previous_action:
            with open(self.checkpoint_path / "previous_action.json", "w") as f:
                json.dump(previous_action, f, indent=2)
        
        # Save previous images
        if previous_images:
            images_dir = self.checkpoint_path / "previous_images"
            images_dir.mkdir(exist_ok=True)
            
            for i, img in enumerate(previous_images):
                img.save(images_dir / f"frame_{i}.png")
        
        logger.info(f"Checkpoint saved successfully")
    
    def load_state(self) -> Dict[str, Any]:
        """
        Load agent state from checkpoint.
        
        Returns:
            Dictionary containing all saved state
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid or incomplete
        """
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load metadata
        metadata_path = self.checkpoint_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError("Checkpoint missing metadata.json")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load costs
        costs_path = self.checkpoint_path / "costs.json"
        if costs_path.exists():
            with open(costs_path) as f:
                costs_data = json.load(f)
                total_cost = Cost(**costs_data["total_cost"])
                total_usage = Usage(**costs_data["total_usage"])
        else:
            # Default values if costs file is missing
            total_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
            total_usage = Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                completion_tokens_details=CompletionTokensDetails()
            )
        
        # Load action history
        action_history = []
        action_history_path = self.checkpoint_path / "action_history.json"
        if action_history_path.exists():
            with open(action_history_path) as f:
                action_history_data = json.load(f)
                action_history = [GameActionRecord(**action) for action in action_history_data]
        
        # Load memory
        memory_prompt = ""
        memory_path = self.checkpoint_path / "memory.txt"
        if memory_path.exists():
            with open(memory_path) as f:
                memory_prompt = f.read()
        
        # Load previous action
        previous_action = None
        previous_action_path = self.checkpoint_path / "previous_action.json"
        if previous_action_path.exists():
            with open(previous_action_path) as f:
                previous_action = json.load(f)
        
        # Load previous images
        previous_images = []
        images_dir = self.checkpoint_path / "previous_images"
        if images_dir.exists():
            image_files = sorted(images_dir.glob("frame_*.png"))
            for img_path in image_files:
                with Image.open(img_path) as img:
                    # Create a copy to fully decouple from file handle
                    previous_images.append(img.copy())
        
        logger.info(f"Checkpoint loaded successfully")
        
        return {
            "metadata": metadata,
            "total_cost": total_cost,
            "total_usage": total_usage,
            "action_history": action_history,
            "memory_prompt": memory_prompt,
            "previous_action": previous_action,
            "previous_images": previous_images,
        }
    
    def checkpoint_exists(self) -> bool:
        """Check if checkpoint exists for this card_id"""
        return self.checkpoint_path.exists() and (self.checkpoint_path / "metadata.json").exists()
    
    def delete_checkpoint(self):
        """Delete the checkpoint directory"""
        if self.checkpoint_path.exists():
            import shutil
            shutil.rmtree(self.checkpoint_path)
            logger.info(f"Deleted checkpoint: {self.checkpoint_path}")
    
    @staticmethod
    def list_checkpoints() -> List[str]:
        """List all available checkpoint card_ids"""
        checkpoint_dir = Path(CheckpointManager.CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for card_dir in checkpoint_dir.iterdir():
            if card_dir.is_dir() and (card_dir / "metadata.json").exists():
                checkpoints.append(card_dir.name)
        
        return sorted(checkpoints)
    
    @staticmethod
    def get_checkpoint_info(card_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a checkpoint"""
        checkpoint_path = Path(CheckpointManager.CHECKPOINT_DIR) / card_id
        metadata_path = checkpoint_path / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path) as f:
            return json.load(f)

