#!/usr/bin/env python3
"""
Comprehensive test suite for checkpoint functionality.

This script tests:
- Basic checkpoint save/load operations
- Checkpoint overwrite and deletion
- Action history reconstruction across plays
- Session restoration scenarios
- Scorecard expiry and checkpoint continuity
"""
import sys
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arcagi3.checkpoint import CheckpointManager
from arcagi3.schemas import Cost, Usage, GameActionRecord, ActionData, CompletionTokensDetails
from arcagi3.agent import MultimodalAgent
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


def test_checkpoint_with_no_images():
    """Test checkpoint save/load with no previous images"""
    print("\nTesting checkpoint with no images...")
    
    test_card_id = "test-no-images-999"
    manager = CheckpointManager(test_card_id)
    
    try:
        # Save checkpoint without images
        manager.save_state(
            config="test-config",
            game_id="test-game-no-images",
            guid="test-guid-no-images",
            max_actions=40,
            retry_attempts=3,
            num_plays=1,
            action_counter=1,
            total_cost=Cost(prompt_cost=0.1, completion_cost=0.05, total_cost=0.15),
            total_usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                completion_tokens_details=CompletionTokensDetails()
            ),
            action_history=[],
            memory_prompt="Test memory",
            previous_action=None,
            previous_images=[],  # No images
            previous_score=0,
            current_play=1,
            play_action_counter=0,
        )
        
        # Load and verify
        state = manager.load_state()
        assert len(state["previous_images"]) == 0, "Should have no images"
        assert state["previous_action"] is None, "Should have no previous action"
        print("✓ Checkpoint with no images handled correctly")
        return True
        
    finally:
        manager.delete_checkpoint()


def test_checkpoint_overwrite():
    """Test that saving to the same card_id overwrites previous checkpoint"""
    print("\nTesting checkpoint overwrite...")
    
    test_card_id = "test-overwrite-888"
    manager = CheckpointManager(test_card_id)
    
    try:
        # Save first checkpoint
        manager.save_state(
            config="config-v1",
            game_id="game-v1",
            guid="guid-v1",
            max_actions=10,
            retry_attempts=3,
            num_plays=1,
            action_counter=1,
            total_cost=Cost(prompt_cost=0.1, completion_cost=0.05, total_cost=0.15),
            total_usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                completion_tokens_details=CompletionTokensDetails()
            ),
            action_history=[],
            memory_prompt="First checkpoint",
            previous_action=None,
            previous_images=[],
            previous_score=0,
            current_play=1,
            play_action_counter=0,
        )
        
        # Save second checkpoint (overwrite)
        manager.save_state(
            config="config-v2",
            game_id="game-v2",
            guid="guid-v2",
            max_actions=20,
            retry_attempts=3,
            num_plays=1,
            action_counter=5,
            total_cost=Cost(prompt_cost=0.2, completion_cost=0.1, total_cost=0.3),
            total_usage=Usage(
                prompt_tokens=200,
                completion_tokens=100,
                total_tokens=300,
                completion_tokens_details=CompletionTokensDetails()
            ),
            action_history=[],
            memory_prompt="Second checkpoint",
            previous_action=None,
            previous_images=[],
            previous_score=10,
            current_play=1,
            play_action_counter=5,
        )
        
        # Load and verify it's the second checkpoint
        state = manager.load_state()
        metadata = state["metadata"]
        
        assert metadata["config"] == "config-v2", "Should have config from second save"
        assert metadata["game_id"] == "game-v2", "Should have game_id from second save"
        assert metadata["guid"] == "guid-v2", "Should have guid from second save"
        assert metadata["action_counter"] == 5, "Should have action_counter from second save"
        assert "Second checkpoint" in state["memory_prompt"], "Should have memory from second save"
        
        print("✓ Checkpoint overwrite works correctly")
        return True
        
    finally:
        manager.delete_checkpoint()


def test_invalid_checkpoint_load():
    """Test loading a non-existent checkpoint"""
    print("\nTesting invalid checkpoint load...")
    
    manager = CheckpointManager("non-existent-checkpoint-12345")
    
    try:
        manager.load_state()
        print("✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError:
        print("✓ Correctly raises FileNotFoundError for non-existent checkpoint")
        return True


def test_mid_play_action_history_reconstruction():
    """
    Test that play_action_history is correctly reconstructed when resuming
    mid-play from a checkpoint.
    
    This tests the fix for the issue where play_action_history was reset
    to an empty list on resume.
    """
    print("\nTesting mid-play action history reconstruction...")
    
    card_id = "test-midplay-12345"
    manager = CheckpointManager(card_id)
    
    try:
        # Simulate a game state mid-play with 5 actions taken
        action_counter = 5
        play_action_counter = 5
        
        # Create 5 action records for the current play
        action_history = [
            GameActionRecord(
                action_num=i,
                action=f"ACTION{i}",
                action_data=None,
                reasoning={"human_action": f"Action {i}", "reasoning": f"Testing action {i}"},
                result_score=i * 10,
                result_state="IN_PROGRESS"
            )
            for i in range(1, 6)
        ]
        
        # Save checkpoint mid-play
        manager.save_state(
            config="test-config",
            game_id="test-game",
            guid="test-guid-123",
            max_actions=40,
            retry_attempts=3,
            num_plays=1,
            action_counter=action_counter,
            total_cost=Cost(prompt_cost=0.1, completion_cost=0.05, total_cost=0.15),
            total_usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                completion_tokens_details=CompletionTokensDetails()
            ),
            action_history=action_history,
            memory_prompt="Test memory state",
            previous_action={"human_action": "Action 5", "reasoning": "Last action"},
            previous_images=[Image.new('RGB', (64, 64), color='blue')],
            previous_score=40,
            current_play=1,
            play_action_counter=play_action_counter,
        )
        
        # Load checkpoint
        state = manager.load_state()
        metadata = state["metadata"]
        
        # Verify metadata
        assert metadata["action_counter"] == 5, "Action counter should be 5"
        assert metadata["play_action_counter"] == 5, "Play action counter should be 5"
        
        # Simulate the reconstruction logic from agent.py
        restored_action_counter = metadata["action_counter"]
        restored_play_action_counter = metadata["play_action_counter"]
        
        # Reconstruct play_action_history
        start_action_num = restored_action_counter - restored_play_action_counter + 1
        end_action_num = restored_action_counter
        
        play_action_history = [
            action for action in state["action_history"]
            if start_action_num <= action.action_num <= end_action_num
        ]
        
        # Verify reconstruction
        assert len(play_action_history) == 5, "Should reconstruct all 5 actions for this play"
        assert play_action_history[0].action_num == 1, "First action should be #1"
        assert play_action_history[-1].action_num == 5, "Last action should be #5"
        
        print("✓ Mid-play action history reconstruction passed")
        return True
        
    finally:
        manager.delete_checkpoint()


def test_multi_play_action_history():
    """
    Test action history tracking across multiple plays with checkpointing.
    
    Simulates:
    - Play 1: 3 actions (actions 1-3)
    - Play 2: 5 actions (actions 4-8) - checkpoint saved mid-play
    - Resume from play 2 checkpoint
    """
    print("\nTesting multi-play action history tracking...")
    
    card_id = "test-multiplay-67890"
    manager = CheckpointManager(card_id)
    
    try:
        # Simulate end of play 1 + mid-way through play 2
        total_actions = 8  # 3 from play 1, 5 from play 2
        play2_actions = 5
        
        # Create action history spanning both plays
        action_history = []
        
        # Play 1 actions (1-3)
        for i in range(1, 4):
            action_history.append(
                GameActionRecord(
                    action_num=i,
                    action=f"ACTION{i}",
                    action_data=None,
                    reasoning={"human_action": f"Play 1 Action {i}"},
                    result_score=i * 10,
                    result_state="IN_PROGRESS" if i < 3 else "GAME_OVER"
                )
            )
        
        # Play 2 actions (4-8)
        for i in range(4, 9):
            action_history.append(
                GameActionRecord(
                    action_num=i,
                    action=f"ACTION{i}",
                    action_data=None,
                    reasoning={"human_action": f"Play 2 Action {i-3}"},
                    result_score=(i - 3) * 10,
                    result_state="IN_PROGRESS"
                )
            )
        
        # Save checkpoint mid-play 2
        manager.save_state(
            config="test-config",
            game_id="test-game",
            guid="test-guid-play2",
            max_actions=40,
            retry_attempts=3,
            num_plays=2,
            action_counter=total_actions,
            total_cost=Cost(prompt_cost=0.2, completion_cost=0.1, total_cost=0.3),
            total_usage=Usage(
                prompt_tokens=200,
                completion_tokens=100,
                total_tokens=300,
                completion_tokens_details=CompletionTokensDetails()
            ),
            action_history=action_history,
            memory_prompt="Memory state after play 1, during play 2",
            previous_action={"human_action": "Play 2 Action 5"},
            previous_images=[Image.new('RGB', (64, 64), color='green')],
            previous_score=50,
            current_play=2,
            play_action_counter=play2_actions,
        )
        
        # Load and verify
        state = manager.load_state()
        metadata = state["metadata"]
        
        assert metadata["action_counter"] == 8, "Total actions should be 8"
        assert metadata["play_action_counter"] == 5, "Play 2 should have 5 actions"
        assert metadata["current_play"] == 2, "Should be on play 2"
        
        # Reconstruct play 2's action history
        restored_action_counter = metadata["action_counter"]
        restored_play_action_counter = metadata["play_action_counter"]
        
        start_action_num = restored_action_counter - restored_play_action_counter + 1
        end_action_num = restored_action_counter
        
        play2_action_history = [
            action for action in state["action_history"]
            if start_action_num <= action.action_num <= end_action_num
        ]
        
        # Verify play 2 actions only
        assert len(play2_action_history) == 5, "Play 2 should have 5 actions"
        assert play2_action_history[0].action_num == 4, "Play 2 first action should be #4"
        assert play2_action_history[-1].action_num == 8, "Play 2 last action should be #8"
        
        print("✓ Multi-play action history tracking passed")
        return True
        
    finally:
        manager.delete_checkpoint()


def test_complex_action_types():
    """
    Test checkpoint with various action types including ACTION6 with coordinates.
    """
    print("\nTesting checkpoint with complex action types...")
    
    card_id = "test-complex-actions"
    manager = CheckpointManager(card_id)
    
    try:
        # Create various action types
        action_history = [
            GameActionRecord(
                action_num=1,
                action="ACTION1",
                action_data=None,
                reasoning={"human_action": "Move up"},
                result_score=0,
                result_state="IN_PROGRESS"
            ),
            GameActionRecord(
                action_num=2,
                action="ACTION6",
                action_data=ActionData(x=25, y=30),
                reasoning={"human_action": "Click at position"},
                result_score=10,
                result_state="IN_PROGRESS"
            ),
            GameActionRecord(
                action_num=3,
                action="ACTION7",
                action_data=None,
                reasoning={"human_action": "Undo last action"},
                result_score=0,
                result_state="IN_PROGRESS"
            ),
        ]
        
        manager.save_state(
            config="test-config",
            game_id="test-game",
            guid="test-guid-complex",
            max_actions=40,
            retry_attempts=3,
            num_plays=1,
            action_counter=3,
            total_cost=Cost(prompt_cost=0.06, completion_cost=0.04, total_cost=0.10),
            total_usage=Usage(
                prompt_tokens=60,
                completion_tokens=40,
                total_tokens=100,
                completion_tokens_details=CompletionTokensDetails()
            ),
            action_history=action_history,
            memory_prompt="Complex actions test",
            previous_action={"human_action": "Undo"},
            previous_images=[Image.new('RGB', (64, 64), color='red')],
            previous_score=0,
            current_play=1,
            play_action_counter=3,
        )
        
        # Load and verify
        state = manager.load_state()
        actions = state["action_history"]
        
        # Verify ACTION6 with coordinates
        action6 = actions[1]
        assert action6.action == "ACTION6", "Second action should be ACTION6"
        assert action6.action_data is not None, "ACTION6 should have action_data"
        assert action6.action_data.x == 25, "X coordinate should be 25"
        assert action6.action_data.y == 30, "Y coordinate should be 30"
        
        print("✓ Complex action types passed")
        return True
        
    finally:
        manager.delete_checkpoint()


def test_scorecard_expiry_checkpoint_continuity():
    """
    Test that when a scorecard expires and a new one is created,
    the agent continues to use the original checkpoint directory.
    """
    print("\nTesting scorecard expiry checkpoint continuity...")
    
    # Create a temporary checkpoint directory for testing
    original_checkpoint_dir = CheckpointManager.CHECKPOINT_DIR
    temp_dir = tempfile.mkdtemp()
    test_checkpoint_dir = os.path.join(temp_dir, ".checkpoint")
    CheckpointManager.CHECKPOINT_DIR = test_checkpoint_dir
    
    try:
        # Original card_id from checkpoint
        original_card_id = "test-card-123"
        new_card_id = "new-card-456"  # New card_id from server after expiry
        
        # Create a mock checkpoint with the original card_id
        checkpoint_mgr = CheckpointManager(original_card_id)
        checkpoint_mgr.save_state(
            config="test-config",
            game_id="test-game",
            guid="test-guid",
            max_actions=40,
            retry_attempts=3,
            num_plays=1,
            action_counter=5,
            total_cost=Cost(prompt_cost=0.1, completion_cost=0.2, total_cost=0.3),
            total_usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                completion_tokens_details=CompletionTokensDetails()
            ),
            action_history=[],
            memory_prompt="Test memory",
            previous_action=None,
            previous_images=[],
            previous_score=0,
            current_play=1,
            play_action_counter=5,
        )
        
        # Verify original checkpoint exists
        assert checkpoint_mgr.checkpoint_exists(), "Original checkpoint should exist"
        original_checkpoint_path = checkpoint_mgr.checkpoint_path
        
        # Simulate the scenario: scorecard expired, new one created
        mock_game_client = Mock()
        
        with patch('arcagi3.agent.create_provider') as mock_create_provider:
            # Mock the provider
            mock_provider = Mock()
            mock_model_config = Mock()
            mock_model_config.pricing = Mock(input=1.0, output=2.0)
            mock_provider.model_config = mock_model_config
            mock_create_provider.return_value = mock_provider
            
            # Create agent with new card_id for API calls, but original checkpoint_card_id
            agent = MultimodalAgent(
                config="test-config",
                game_client=mock_game_client,
                card_id=new_card_id,  # New card_id from server
                max_actions=40,
                retry_attempts=3,
                num_plays=1,
                checkpoint_frequency=1,
                checkpoint_card_id=original_card_id,  # Original checkpoint card_id preserved
            )
            
            # Verify the agent is using the original checkpoint directory
            assert agent.checkpoint_manager.card_id == original_card_id, \
                "Agent should use original card_id for checkpoints"
            assert agent.checkpoint_manager.checkpoint_path == original_checkpoint_path, \
                "Agent should use original checkpoint path"
            
            # Verify the agent can load the checkpoint
            assert agent.checkpoint_manager.checkpoint_exists(), \
                "Agent should see the original checkpoint"
            
            # Load the checkpoint state
            state = agent.checkpoint_manager.load_state()
            assert state["metadata"]["game_id"] == "test-game", \
                "Agent should load original checkpoint data"
            assert state["metadata"]["action_counter"] == 5, \
                "Agent should load original action counter"
            
            # Save a new checkpoint and verify it goes to the original location
            agent.save_checkpoint()
            assert original_checkpoint_path.exists(), \
                "Checkpoint should still exist at original path"
            
            # Verify no checkpoint was created for the new card_id
            new_checkpoint_path = Path(test_checkpoint_dir) / new_card_id
            assert not new_checkpoint_path.exists(), \
                "No checkpoint should be created for new card_id"
        
        print("✓ Scorecard expiry checkpoint continuity passed")
        return True
        
    finally:
        # Clean up
        CheckpointManager.CHECKPOINT_DIR = original_checkpoint_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run all checkpoint tests"""
    print("=" * 60)
    print("Comprehensive Checkpoint Test Suite")
    print("=" * 60)
    
    tests = [
        # Basic tests
        ("Create and verify checkpoint", lambda: create_test_checkpoint() and verify_checkpoint(create_test_checkpoint())),
        ("Checkpoint with no images", test_checkpoint_with_no_images),
        ("Checkpoint overwrite", test_checkpoint_overwrite),
        ("Invalid checkpoint load", test_invalid_checkpoint_load),
        # Integration tests
        ("Mid-play action history reconstruction", test_mid_play_action_history_reconstruction),
        ("Multi-play action history tracking", test_multi_play_action_history),
        ("Complex action types", test_complex_action_types),
        # Scorecard expiry test
        ("Scorecard expiry checkpoint continuity", test_scorecard_expiry_checkpoint_continuity),
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    # First create a test checkpoint for basic tests
    try:
        print("\n" + "=" * 60)
        print("BASIC CHECKPOINT TESTS")
        print("=" * 60)
        
        test_card_id = create_test_checkpoint()
        
        if verify_checkpoint(test_card_id):
            passed += 1
            print("✓ Create and verify checkpoint passed")
        else:
            failed += 1
            failed_tests.append("Create and verify checkpoint")
            print("✗ Create and verify checkpoint failed")
        
        test_checkpoint_list()
        cleanup_test_checkpoint(test_card_id)
        
    except Exception as e:
        failed += 1
        failed_tests.append("Create and verify checkpoint")
        print(f"✗ Create and verify checkpoint failed: {e}")
    
    # Run the rest of the tests
    print("\n" + "=" * 60)
    print("EDGE CASE & INTEGRATION TESTS")
    print("=" * 60)
    
    for test_name, test_func in tests[1:]:  # Skip first test as we already ran it
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                failed_tests.append(test_name)
        except Exception as e:
            failed += 1
            failed_tests.append(test_name)
            print(f"\n✗ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
    print("=" * 60)
    
    if failed == 0:
        print("✓ All checkpoint tests passed!")
        return 0
    else:
        print(f"✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

