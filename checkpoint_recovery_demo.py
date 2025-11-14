"""
Demo: Checkpoint Recovery with LLM-driven actions.

Demonstrates:
1. Start a game with LLM deciding actions
2. Execute 2 actions out of 4 total actions
3. Close client (simulating failure)
4. Open new client and resume from checkpoint
5. Execute remaining 2 actions
6. Close scorecard for visualization
"""
import os
import sys
import time
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arcagi3.agent import MultimodalAgent
from arcagi3.game_client import GameClient
from arcagi3.utils import generate_scorecard_tags, read_models_config
from arcagi3.checkpoint import CheckpointManager
from arcagi3.image_utils import grid_to_image

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulatedFailure(Exception):
    """Custom exception to simulate a failure"""
    pass


def execute_first_two_actions(game_id: str, config: str, total_actions: int = 4):
    """
    Execute first 2 actions using LLM, then simulate failure.
    
    Args:
        game_id: Game ID to play
        config: Model configuration name
        total_actions: Total number of actions to execute (4 in this case)
    
    Returns:
        Tuple of (card_id, game_id) for recovery
    """
    print("=" * 60)
    print("PART 1: Starting game and executing first 2 actions")
    print("=" * 60)
    print()
    
    # Check if API key is set
    if not os.getenv("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not found in environment.")
        print("Please set it in your .env file or environment.")
        return None, None
    
    game_client = None
    card_id = None
    agent = None
    
    try:
        # Initialize game client
        print("1. Initializing game client...")
        game_client = GameClient()
        print("   ✅ Game client initialized")
        print()
        
        # Read model config and generate tags
        model_config = read_models_config(config)
        tags = generate_scorecard_tags(model_config)
        print(f"2. Using model config: {config}")
        print(f"   Model: {model_config.model_name}")
        print(f"   Provider: {model_config.provider}")
        print()
        
        # Open scorecard
        print("3. Opening scorecard...")
        scorecard_response = game_client.open_scorecard([game_id], tags=tags)
        card_id = scorecard_response.get("card_id")
        print(f"   ✅ Scorecard opened: {card_id}")
        print()
        
        # Create agent with checkpointing enabled (save after each action)
        print("4. Creating agent with checkpointing enabled...")
        agent = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=total_actions,  # Total actions we plan to execute
            retry_attempts=2,
            checkpoint_frequency=1,  # Save checkpoint after each action
        )
        print(f"   ✅ Agent created with max_actions={total_actions}")
        print(f"   ✅ Checkpoint frequency: after each action")
        print()
        
        # Custom play_game that executes 2 actions then simulates failure
        print("5. Starting game execution...")
        print(f"   Will execute {2} actions, then simulate failure")
        print()
        
        # Start the game
        state = game_client.reset_game(card_id, game_id, guid=None)
        guid = state.get("guid")
        current_score = state.get("score", 0)
        current_state = state.get("state", "IN_PROGRESS")
        
        print(f"   Game initialized: guid={guid}")
        print(f"   Initial score: {current_score}, state: {current_state}")
        print()
        
        # Initialize memory
        available_actions = state.get("available_actions", [])
        agent._initialize_memory(available_actions)
        
        action_counter = 0
        play_action_counter = 0
        actions_to_execute = 2  # Execute 2 actions before failure
        
        # Execute actions using LLM
        while (
            current_state not in ["WIN", "GAME_OVER"]
            and play_action_counter < actions_to_execute
        ):
            try:
                frames = state.get("frame", [])
                if not frames:
                    print("   ⚠️  No frames in state, breaking")
                    break
                
                frame_images = [grid_to_image(frame) for frame in frames]
                
                # LLM analyzes previous action
                analysis = agent._analyze_previous_action(frame_images, current_score)
                
                # LLM chooses next action
                print(f"   🤖 LLM deciding action {play_action_counter + 1}...")
                human_action_dict = agent._choose_human_action(frame_images, analysis)
                human_action = human_action_dict.get("human_action")
                
                if not human_action:
                    print("   ⚠️  No human_action in response")
                    break
                
                print(f"      Decision: {human_action}")
                
                # Convert to game action
                game_action_dict = agent._convert_to_game_action(human_action, frame_images[-1])
                action_name = game_action_dict.get("action")
                
                if not action_name:
                    print("   ⚠️  No action name in response")
                    break
                
                # Prepare action data
                action_data_dict = {}
                if action_name == "ACTION6":
                    x = game_action_dict.get("x", 0)
                    y = game_action_dict.get("y", 0)
                    action_data_dict = {
                        "x": max(0, min(x, 127)) // 2,
                        "y": max(0, min(y, 127)) // 2,
                    }
                
                # Execute action
                reasoning_for_api = human_action_dict.get("reasoning", "")
                state = agent._execute_game_action(action_name, action_data_dict, game_id, guid, reasoning_for_api)
                guid = state.get("guid", guid)
                new_score = state.get("score", current_score)
                current_state = state.get("state", "IN_PROGRESS")
                
                # Update counters
                action_counter += 1
                play_action_counter += 1
                agent.action_counter = action_counter
                agent._play_action_counter = play_action_counter
                
                # Update agent state
                agent._current_guid = guid
                agent._previous_action = human_action_dict
                agent._previous_images = frame_images
                agent._previous_score = current_score
                current_score = new_score
                
                print(f"   ✅ Action {play_action_counter}: {action_name}")
                print(f"      Score: {current_score}, State: {current_state}")
                
                # Save checkpoint after each action
                if agent.checkpoint_frequency > 0 and play_action_counter % agent.checkpoint_frequency == 0:
                    agent.save_checkpoint()
                    print(f"      💾 Checkpoint saved")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error during action: {e}")
                # Save checkpoint on error
                agent.save_checkpoint()
                raise
        
        # Verify checkpoint exists
        checkpoint_mgr = CheckpointManager(card_id)
        if checkpoint_mgr.checkpoint_exists():
            print(f"✅ Checkpoint verified: .checkpoint/{card_id}/")
            checkpoint_info = CheckpointManager.get_checkpoint_info(card_id)
            if checkpoint_info:
                print(f"   Checkpoint info:")
                print(f"     - Game ID: {checkpoint_info.get('game_id')}")
                print(f"     - Actions completed: {checkpoint_info.get('action_counter')}")
                print(f"     - GUID: {checkpoint_info.get('guid')}")
        else:
            print(f"⚠️  WARNING: Checkpoint not found!")
        
        print()
        print(f"💥 SIMULATING FAILURE (closing client)...")
        print(f"   Executed {play_action_counter} out of {total_actions} actions")
        print(f"   Final score: {current_score}")
        print(f"   Final state: {current_state}")
        print()
        
        # Verify scorecard is still open
        try:
            scorecard_info = game_client.get_scorecard(card_id)
            print(f"✅ Scorecard still open on server")
            print(f"   (Can resume within ~15 minutes)")
        except Exception as e:
            print(f"⚠️  Scorecard check failed: {e}")
        
        print()
        print("Closing game client (simulating process failure/termination)...")
        game_client.close()
        print("✅ Game client closed")
        print()
        
        return card_id, game_id
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    finally:
        # Note: We don't close scorecard here - it remains open for recovery
        if game_client:
            try:
                game_client.close()
            except:
                pass


def resume_and_execute_remaining_actions(card_id: str, game_id: str, config: str, total_actions: int = 4):
    """
    Resume from checkpoint and execute remaining actions.
    
    Args:
        card_id: Scorecard ID from previous execution
        game_id: Game ID
        config: Model configuration name
        total_actions: Total number of actions (4 in this case)
    """
    print("=" * 60)
    print("PART 2: Resuming from checkpoint and executing remaining actions")
    print("=" * 60)
    print()
    
    # Check if API key is set
    if not os.getenv("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not found in environment.")
        return
    
    game_client = None
    
    try:
        # Wait a moment to simulate restart delay
        print("1. Simulating restart delay...")
        time.sleep(1)
        print("   ✅ Ready to resume")
        print()
        
        # Create a completely new game client (simulating new process)
        print("2. Creating new game client (simulating new process)...")
        game_client = GameClient()
        print("   ✅ New game client initialized")
        print()
        
        # Verify checkpoint exists
        print("3. Verifying checkpoint...")
        checkpoint_mgr = CheckpointManager(card_id)
        if not checkpoint_mgr.checkpoint_exists():
            print(f"   ❌ Checkpoint not found at: .checkpoint/{card_id}/")
            return
        
        checkpoint_info = CheckpointManager.get_checkpoint_info(card_id)
        if not checkpoint_info:
            print(f"   ❌ Could not read checkpoint info")
            return
        
        print(f"   ✅ Checkpoint found")
        print(f"   Checkpoint info:")
        print(f"     - Game ID: {checkpoint_info.get('game_id')}")
        print(f"     - Actions completed: {checkpoint_info.get('action_counter')}")
        print(f"     - GUID: {checkpoint_info.get('guid')}")
        print()
        
        # Verify scorecard still exists on server
        print("4. Verifying scorecard connection...")
        try:
            scorecard_info = game_client.get_scorecard(card_id, game_id)
            print(f"   ✅ Scorecard {card_id} is accessible")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not verify scorecard: {e}")
            print(f"   Continuing anyway...")
        print()
        
        # Read model config
        model_config = read_models_config(config)
        
        # Create agent with checkpoint resume enabled
        print("5. Creating agent to resume from checkpoint...")
        agent = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=total_actions,  # Total actions we plan to execute
            retry_attempts=2,
            checkpoint_frequency=1,  # Continue saving checkpoints
            checkpoint_card_id=card_id,  # Use same card_id for checkpoint management
        )
        print(f"   ✅ Agent created for resume")
        print()
        
        # Resume game using agent's play_game method with resume_from_checkpoint=True
        print("6. Resuming game from checkpoint...")
        print(f"   Will execute remaining {total_actions - checkpoint_info.get('action_counter', 0)} actions")
        print()
        
        # Use agent's play_game method which handles checkpoint resume
        result = agent.play_game(game_id, resume_from_checkpoint=True)
        
        print()
        print("=" * 60)
        print("Game execution completed!")
        print("=" * 60)
        print(f"Final State: {result.final_state}")
        print(f"Final Score: {result.final_score}")
        print(f"Total Actions Taken: {result.actions_taken}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print()
        
        # Close scorecard to make it viewable
        print("7. Closing scorecard...")
        try:
            game_client.close_scorecard(card_id)
            print(f"   ✅ Scorecard {card_id} closed")
            
            # Display scorecard URL for visualization
            scorecard_url = f"{game_client.ROOT_URL}/scorecards/{card_id}"
            print(f"\nView your scorecard online: {scorecard_url}")
            print(f"{'=' * 60}")
        except Exception as e:
            print(f"   ⚠️  Failed to close scorecard: {e}")
            # Still show URL
            scorecard_url = f"{game_client.ROOT_URL}/scorecards/{card_id}"
            print(f"\nView your scorecard online (may not be available yet): {scorecard_url}")
        
    except Exception as e:
        print(f"\n❌ Error during resume: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if game_client:
            try:
                game_client.close()
                print("\n✅ Client session closed")
            except Exception as e:
                print(f"\n⚠️  Failed to close client session: {e}")


def checkpoint_recovery_demo():
    """Main demo function"""
    print("=" * 60)
    print("Checkpoint Recovery Demo with LLM-driven Actions")
    print("=" * 60)
    print()
    print("This demo will:")
    print("  1. Start a game and execute 2 actions (LLM decides)")
    print("  2. Simulate failure by closing the client")
    print("  3. Resume from checkpoint with new client")
    print("  4. Execute remaining 2 actions (LLM decides)")
    print("  5. Close scorecard for visualization")
    print()
    
    # Configuration
    game_id = "ls20-fa137e247ce6"
    config = "gpt-4o-mini-2024-07-18"  # Use a cheaper model for demo
    total_actions = 4
    
    print(f"Configuration:")
    print(f"  Game ID: {game_id}")
    print(f"  Model: {config}")
    print(f"  Total actions: {total_actions}")
    print()
    
    # Part 1: Execute first 2 actions
    card_id, game_id_result = execute_first_two_actions(game_id, config, total_actions)
    
    if not card_id or not game_id_result:
        print("\n❌ Failed to complete first part. Cannot resume.")
        return
    
    # Part 2: Resume and execute remaining actions
    resume_and_execute_remaining_actions(card_id, game_id_result, config, total_actions)
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the demo
    import argparse
    parser = argparse.ArgumentParser(description="Checkpoint recovery demo with LLM actions")
    parser.add_argument(
        "--game-id",
        type=str,
        default="ls20-fa137e247ce6",
        help="Game ID to use (default: ls20-fa137e247ce6)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model configuration name (default: gpt-4o-mini-2024-07-18)"
    )
    parser.add_argument(
        "--total-actions",
        type=int,
        default=4,
        help="Total number of actions to execute (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Override defaults with command line args if provided
    game_id = args.game_id
    config = args.config
    total_actions = args.total_actions
    
    print("=" * 60)
    print("Checkpoint Recovery Demo with LLM-driven Actions")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  Game ID: {game_id}")
    print(f"  Model: {config}")
    print(f"  Total actions: {total_actions}")
    print()
    
    # Part 1: Execute first 2 actions
    card_id, game_id_result = execute_first_two_actions(game_id, config, total_actions)
    
    if not card_id or not game_id_result:
        print("\n❌ Failed to complete first part. Cannot resume.")
        sys.exit(1)
    
    # Part 2: Resume and execute remaining actions
    resume_and_execute_remaining_actions(card_id, game_id_result, config, total_actions)

