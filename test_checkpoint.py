#!/usr/bin/env python3
"""
Simple test script for checkpoint functionality.

This script tests that checkpoints can be created and loaded correctly.
"""
import sys
import os
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arcagi3.checkpoint import CheckpointManager
from arcagi3.schemas import Cost, Usage, GameActionRecord, ActionData, CompletionTokensDetails
from PIL import Image
import json


def create_test_checkpoint():
    """Create a test checkpoint with sample data"""
    print("Creating test checkpoint...")
    
    test_card_id = "test-checkpoint-12345"
    manager = CheckpointManager(test_card_id)
    
    # Create sample data
    test_cost = Cost(prompt_cost=0.5, completion_cost=0.3, total_cost=0.8)
    test_usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        completion_tokens_details=CompletionTokensDetails()
    )
    
    test_actions = [
        GameActionRecord(
            action_num=1,
            action="ACTION1",
            action_data=None,
            reasoning={"human_action": "Move up", "reasoning": "Testing"},
            result_score=0,
            result_state="IN_PROGRESS"
        ),
        GameActionRecord(
            action_num=2,
            action="ACTION5",
            action_data=ActionData(x=10, y=20),
            reasoning={"human_action": "Perform action", "reasoning": "Testing action"},
            result_score=1,
            result_state="IN_PROGRESS"
        ),
    ]
    
    # Create a simple test image
    test_image = Image.new('RGB', (64, 64), color='red')
    
    # Save checkpoint
    manager.save_state(
        config="test-config",
        game_id="test-game-123",
        guid="test-guid-456",
        max_actions=40,
        retry_attempts=3,
        num_plays=1,
        action_counter=2,
        total_cost=test_cost,
        total_usage=test_usage,
        action_history=test_actions,
        memory_prompt="## Test Memory\nThis is a test memory.",
        previous_action={"human_action": "Test action", "reasoning": "Testing"},
        previous_images=[test_image, test_image],
        previous_score=1,
        current_play=1,
        play_action_counter=2,
    )
    
    print(f"✓ Test checkpoint created at: {manager.checkpoint_path}")
    return test_card_id


def verify_checkpoint(card_id):
    """Verify that checkpoint can be loaded correctly"""
    print(f"\nVerifying checkpoint: {card_id}")
    
    manager = CheckpointManager(card_id)
    
    # Check if checkpoint exists
    if not manager.checkpoint_exists():
        print("✗ Checkpoint does not exist!")
        return False
    
    print("✓ Checkpoint exists")
    
    # Load checkpoint
    try:
        state = manager.load_state()
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False
    
    print("✓ Checkpoint loaded successfully")
    
    # Verify metadata
    metadata = state["metadata"]
    assert metadata["card_id"] == card_id, "Card ID mismatch"
    assert metadata["config"] == "test-config", "Config mismatch"
    assert metadata["game_id"] == "test-game-123", "Game ID mismatch"
    assert metadata["action_counter"] == 2, "Action counter mismatch"
    print("✓ Metadata verified")
    
    # Verify costs
    assert state["total_cost"].total_cost == 0.8, "Cost mismatch"
    assert state["total_usage"].total_tokens == 1500, "Usage mismatch"
    print("✓ Costs and usage verified")
    
    # Verify action history
    assert len(state["action_history"]) == 2, "Action history length mismatch"
    assert state["action_history"][0].action == "ACTION1", "Action mismatch"
    print("✓ Action history verified")
    
    # Verify memory
    assert "Test Memory" in state["memory_prompt"], "Memory prompt mismatch"
    print("✓ Memory verified")
    
    # Verify previous action
    assert state["previous_action"]["human_action"] == "Test action", "Previous action mismatch"
    print("✓ Previous action verified")
    
    # Verify images
    assert len(state["previous_images"]) == 2, "Previous images count mismatch"
    print("✓ Previous images verified")
    
    return True


def test_checkpoint_list():
    """Test listing checkpoints"""
    print("\nTesting checkpoint listing...")
    
    checkpoints = CheckpointManager.list_checkpoints()
    print(f"✓ Found {len(checkpoints)} checkpoint(s)")
    
    for card_id in checkpoints:
        info = CheckpointManager.get_checkpoint_info(card_id)
        if info:
            print(f"  - {card_id}: {info.get('game_id', 'N/A')}")


def cleanup_test_checkpoint(card_id):
    """Clean up test checkpoint"""
    print(f"\nCleaning up test checkpoint: {card_id}")
    
    manager = CheckpointManager(card_id)
    manager.delete_checkpoint()
    
    if not manager.checkpoint_exists():
        print("✓ Test checkpoint deleted successfully")
    else:
        print("✗ Failed to delete test checkpoint")


def main():
    """Run checkpoint tests"""
    print("=" * 60)
    print("Checkpoint Functionality Test")
    print("=" * 60)
    
    try:
        # Create test checkpoint
        test_card_id = create_test_checkpoint()
        
        # Verify checkpoint
        if not verify_checkpoint(test_card_id):
            print("\n✗ Checkpoint verification failed!")
            return 1
        
        # Test listing
        test_checkpoint_list()
        
        # Clean up
        cleanup_test_checkpoint(test_card_id)
        
        print("\n" + "=" * 60)
        print("✓ All checkpoint tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

