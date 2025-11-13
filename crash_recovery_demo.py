"""
Demo: Simulate a crash scenario and recovery using checkpoints.

This demonstrates the real-world use case:
1. Game is running and executing actions
2. Unexpected crash/interruption occurs (simulated)
3. Checkpoint is saved automatically
4. Scorecard remains open (within 15-minute window)
5. Restart CLI and resume from checkpoint successfully

This simulates what happens when:
- Process crashes (exception, OOM, etc.)
- User interrupts (Ctrl-C)
- Network issues cause failure
- System restart
"""
import os
import sys
import time
import logging
import signal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the main components
from src.arcagi3.agent import MultimodalAgent
from src.arcagi3.game_client import GameClient
from src.arcagi3.utils import generate_scorecard_tags, read_models_config
from src.arcagi3.checkpoint import CheckpointManager
from src.arcagi3.image_utils import grid_to_image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulatedCrash(Exception):
    """Custom exception to simulate a crash"""
    pass


def crash_recovery_demo():
    """Demonstrate crash recovery using checkpoints"""
    print("=" * 60)
    print("Crash Recovery Demo: Simulate Failure and Resume")
    print("=" * 60)
    print()
    
    # Check if API key is set
    if not os.getenv("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not found in environment.")
        print("Please set it in your .env file or environment.")
        return
    
    # Initialize game client
    game_client = GameClient()
    
    try:
        # Get list of available games
        games = game_client.list_games()
        if not games:
            print("No games available. Check your ARC_API_KEY.")
            return
        
        # Find an "ls" game (game_id starting with "ls")
        ls_game = None
        for game in games:
            if game['game_id'].startswith('ls'):
                ls_game = game
                break
        
        if not ls_game:
            print("No 'ls' game found. Available games:")
            for game in games[:5]:  # Show first 5
                print(f"  - {game['game_id']}: {game['title']}")
            print("\nUsing first available game instead...")
            ls_game = games[0]
        
        game_id = ls_game['game_id']
        print(f"Selected game: {game_id}")
        print(f"Game title: {ls_game['title']}")
        print()
        
        # Configure model (use a cheaper model for demo)
        config = "gpt-4o-mini-2024-07-18"
        model_config = read_models_config(config)
        print(f"Using model: {config}")
        print()
        
        # Generate tags for scorecard
        tags = generate_scorecard_tags(model_config)
        
        # ============================================================
        # PART 1: Start game and simulate crash
        # ============================================================
        print("=" * 60)
        print("PART 1: Starting game and simulating crash")
        print("=" * 60)
        print()
        
        # Open scorecard
        scorecard_response = game_client.open_scorecard([game_id], tags=tags)
        card_id = scorecard_response.get("card_id")
        print(f"✅ Scorecard opened: {card_id}")
        print(f"   (Scorecard will remain open for ~15 minutes)")
        print()
        
        # Create agent with checkpointing enabled
        agent = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=5,  # We'll crash before reaching this
            retry_attempts=2,
            checkpoint_frequency=1,  # Save checkpoint after each action
        )
        
        print("Starting game execution...")
        print("Will simulate crash after a few actions...")
        print()
        
        # Custom play_game that simulates a crash
        try:
            # Start the game
            state = game_client.reset_game(card_id, game_id, guid=None)
            guid = state.get("guid")
            current_score = state.get("score", 0)
            current_state = state.get("state", "IN_PROGRESS")
            
            print(f"Game initialized: guid={guid}, score={current_score}, state={current_state}")
            print()
            
            # Initialize memory
            available_actions = state.get("available_actions", [])
            agent._initialize_memory(available_actions)
            
            action_counter = 0
            play_action_counter = 0
            
            # Execute a few actions, then crash
            crash_after_actions = 2  # Crash after 3 actions
            
            while (
                current_state not in ["WIN", "GAME_OVER"]
                and play_action_counter < agent.max_actions
            ):
                try:
                    frames = state.get("frame", [])
                    if not frames:
                        print("No frames in state, breaking")
                        break
                    
                    frame_images = [grid_to_image(frame) for frame in frames]
                    
                    # Analyze previous action
                    analysis = agent._analyze_previous_action(frame_images, current_score)
                    
                    # Choose human action
                    human_action_dict = agent._choose_human_action(frame_images, analysis)
                    human_action = human_action_dict.get("human_action")
                    
                    if not human_action:
                        print("No human_action in response")
                        break
                    
                    # Convert to game action
                    game_action_dict = agent._convert_to_game_action(human_action, frame_images[-1])
                    action_name = game_action_dict.get("action")
                    
                    if not action_name:
                        print("No action name in response")
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
                    
                    # Update state
                    agent._previous_action = human_action_dict
                    agent._previous_images = frame_images
                    agent._previous_score = current_score
                    current_score = new_score
                    
                    print(f"✅ Action {play_action_counter}: {action_name}, Score: {current_score}, State: {current_state}")
                    
                    # Save checkpoint after each action
                    if agent.checkpoint_frequency > 0 and play_action_counter % agent.checkpoint_frequency == 0:
                        agent.save_checkpoint()
                        print(f"   💾 Checkpoint saved")
                    
                    # Simulate crash after N actions
                    if play_action_counter >= crash_after_actions:
                        print()
                        print("💥 SIMULATING CRASH (raising exception)...")
                        print(f"   Crashed after {play_action_counter} actions")
                        print(f"   Checkpoint should be saved at: .checkpoint/{card_id}/")
                        raise SimulatedCrash("Simulated unexpected crash!")
                    
                except SimulatedCrash:
                    # Re-raise to exit the loop
                    raise
                except Exception as e:
                    print(f"Error during action: {e}")
                    # Save checkpoint on error
                    agent.save_checkpoint()
                    raise
        
        except SimulatedCrash:
            print()
            print("❌ CRASH OCCURRED!")
            print(f"   Actions completed before crash: {play_action_counter}")
            print(f"   Final score: {current_score}")
            print(f"   Final state: {current_state}")
            print()
            
            # Verify checkpoint exists
            checkpoint_mgr = CheckpointManager(card_id)
            if checkpoint_mgr.checkpoint_exists():
                print(f"✅ Checkpoint verified: .checkpoint/{card_id}/")
                checkpoint_info = CheckpointManager.get_checkpoint_info(card_id)
                if checkpoint_info:
                    print(f"   Checkpoint info:")
                    print(f"     - Game ID: {checkpoint_info.get('game_id')}")
                    print(f"     - Actions: {checkpoint_info.get('action_counter')}")
                    print(f"     - GUID: {checkpoint_info.get('guid')}")
                    print(f"     - Timestamp: {checkpoint_info.get('checkpoint_timestamp')}")
            else:
                print(f"⚠️  WARNING: Checkpoint not found!")
            print()
            
            # Verify scorecard is still open
            try:
                scorecard_info = game_client.get_scorecard(card_id)
                print(f"✅ Scorecard still open on server")
                print(f"   (Can resume within ~15 minutes)")
            except Exception as e:
                print(f"❌ Scorecard check failed: {e}")
            print()
            
            # Close the game client to simulate process termination
            print("Closing game client (simulating process crash/termination)...")
            game_client.close()
            print("✅ Game client closed")
            print()
        
        # ============================================================
        # PART 2: Simulate restart and resume from checkpoint
        # ============================================================
        print("=" * 60)
        print("PART 2: Restarting and resuming from checkpoint")
        print("=" * 60)
        print()
        print("Simulating restart of CLI...")
        print("(In real scenario, process would have crashed and restarted)")
        print()
        
        # Wait a moment to simulate time passing
        print("Waiting 2 seconds to simulate restart delay...")
        time.sleep(2)
        print()
        
        # Create a completely new game client (simulating new process)
        # NOTE: GameClient is stateless - it's just an HTTP client for API calls.
        # The game state is stored on the server, associated with card_id and guid.
        # Creating a new GameClient does NOT reset the game - it's just a new connection
        # to the same server-side game session.
        print("Creating new game client (simulating new process start)...")
        print("   (GameClient is stateless - game state is on server)")
        print()
        print("IMPORTANT: Connection loss and failure behavior:")
        print("   ✅ Scorecard does NOT close automatically when connection is lost")
        print("   ✅ Scorecard does NOT close automatically on failures/exceptions")
        print("   ✅ Scorecard remains open for ~15 minutes on the server")
        print("   ✅ Game session (guid) persists on server until scorecard expires")
        print("   ✅ You can reconnect and continue the same game session")
        print("   ✅ Failures only affect the local client - server state is preserved")
        print()
        print("   NOTE: You CAN manually close the scorecard with close_scorecard()")
        print("   But it stays open by default to allow checkpoint recovery")
        print()
        game_client_resume = GameClient()
        print("✅ New game client created")
        print("   (Game state persists on server via card_id and guid)")
        print()
        
        # Verify scorecard is still accessible with new client
        print("Verifying scorecard is still open with new client...")
        try:
            scorecard_info = game_client_resume.get_scorecard(card_id)
            print(f"✅ Scorecard still accessible: {card_id}")
            print(f"   (Within 15-minute window)")
        except Exception as e:
            print(f"❌ Scorecard no longer accessible: {e}")
            print("   (Scorecard may have expired - would need new scorecard)")
            print("   Continuing anyway to show checkpoint recovery...")
        print()
        
        # Verify checkpoint contains the same card_id and guid
        print("Verifying checkpoint contains same session IDs...")
        checkpoint_info = CheckpointManager.get_checkpoint_info(card_id)
        if checkpoint_info:
            saved_card_id = checkpoint_info.get('card_id')
            saved_game_id = checkpoint_info.get('game_id')
            saved_guid = checkpoint_info.get('guid')
            print(f"  Checkpoint card_id: {saved_card_id}")
            print(f"  Checkpoint game_id: {saved_game_id}")
            print(f"  Checkpoint guid: {saved_guid}")
            print(f"  Current card_id: {card_id}")
            print(f"  Current game_id: {game_id}")
            if saved_card_id == card_id and saved_game_id == game_id:
                print("  ✅ Using same card_id and game_id - will continue same game session")
            else:
                print("  ⚠️  WARNING: IDs don't match!")
        print()
        
        # Create new agent instance with new client (simulating restart)
        # IMPORTANT: Using the SAME card_id to continue the same game session
        print("Creating new agent instance with new client (simulating restart)...")
        print(f"  Using SAME card_id: {card_id} (to continue same game session)")
        agent_resume = MultimodalAgent(
            config=config,
            game_client=game_client_resume,
            card_id=card_id,  # Same card_id = same scorecard = same game session
            max_actions=8,  # Allow more actions to complete
            retry_attempts=2,
            checkpoint_frequency=1,
        )
        
        # Resume from checkpoint
        # This will restore the saved guid and continue the same game session
        print("Resuming from checkpoint...")
        print("  (Will use saved guid to continue same game session on server)")
        print()
        
        # Verify checkpoint exists before attempting resume
        if not agent_resume.checkpoint_manager.checkpoint_exists():
            print("❌ ERROR: Checkpoint does not exist!")
            print(f"   Expected checkpoint at: .checkpoint/{card_id}/")
            return
        else:
            print(f"✅ Checkpoint verified: .checkpoint/{card_id}/")
        
        # Manually restore checkpoint to verify it works
        print("Manually restoring checkpoint to verify...")
        if agent_resume.restore_from_checkpoint():
            print("✅ Checkpoint restored successfully")
            print(f"   Restored game_id: {agent_resume._current_game_id}")
            print(f"   Restored guid: {agent_resume._current_guid}")
            print(f"   Restored action_counter: {agent_resume.action_counter}")
            print(f"   Restored play_action_counter: {agent_resume._play_action_counter}")
            # Use the restored game_id instead of the one we passed
            if agent_resume._current_game_id:
                game_id = agent_resume._current_game_id
                print(f"   Using restored game_id: {game_id}")
            
            # Verify we have the guid to continue the session
            if agent_resume._current_guid:
                print()
                print(f"✅ Have saved guid: {agent_resume._current_guid}")
                print(f"   Will continue game session from where we left off")
                print(f"   (NOT resetting - continuing from action {agent_resume._play_action_counter})")
            else:
                print()
                print(f"⚠️  No saved guid in checkpoint")
                print(f"   Will need to start new session (but keeping memory)")
        else:
            print("❌ ERROR: Failed to restore checkpoint!")
            return
        print()
        
        try:
            # Now call play_game with resume_from_checkpoint=True
            # IMPORTANT: The agent will NOT reset the game - it will continue from where we stopped
            # It uses the saved guid to continue the same game session on the server
            print("Calling play_game with resume_from_checkpoint=True...")
            print("  (Agent will continue game, NOT reset it)")
            print()
            result_resume = agent_resume.play_game(game_id, resume_from_checkpoint=True)
            
            print()
            print("PART 2 Results:")
            print(f"  ✅ Successfully resumed from checkpoint")
            print(f"  Total actions taken: {result_resume.actions_taken}")
            print(f"  Final score: {result_resume.final_score}")
            print(f"  Final state: {result_resume.final_state}")
            print(f"  Duration: {result_resume.duration_seconds:.2f}s")
            print(f"  Total cost: ${result_resume.total_cost.total_cost:.4f}")
            print()
            
        except Exception as e:
            print(f"❌ Error resuming from checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print()
        
        # ============================================================
        # Summary
        # ============================================================
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Game ID: {game_id}")
        print(f"Card ID: {card_id}")
        print(f"Actions before crash: {crash_after_actions}")
        print(f"Checkpoint location: .checkpoint/{card_id}/")
        print()
        
        # Display scorecard URL if available
        try:
            if 'result_resume' in locals() and hasattr(result_resume, 'scorecard_url'):
                print(f"Scorecard URL: {result_resume.scorecard_url}")
            else:
                # Construct URL manually
                scorecard_url = f"{game_client_resume.ROOT_URL}/scorecards/{card_id}"
                print(f"Scorecard URL: {scorecard_url}")
        except:
            scorecard_url = f"https://three.arcprize.org/scorecards/{card_id}"
            print(f"Scorecard URL: {scorecard_url}")
        print()
        
        print("Key findings:")
        print("  ✅ Checkpoint saved automatically after each action")
        print("  ✅ Scorecard does NOT close automatically on connection loss")
        print("  ✅ Scorecard does NOT close automatically on failures/exceptions/crashes")
        print("  ✅ Scorecard CAN be closed manually with close_scorecard()")
        print("  ✅ Scorecard remains open for ~15 minutes on server (unless manually closed)")
        print("  ✅ Game session (guid) persists until scorecard expires or is closed")
        print("  ✅ Failures only affect local client - server state is preserved")
        print("  ✅ Can reconnect and resume from checkpoint within time window")
        print("  ✅ State, memory, and costs are preserved")
        print()
        print("When scorecard closes:")
        print("  - Manually: game_client.close_scorecard(card_id)")
        print("  - Automatically: Game won (WIN state)")
        print("  - Automatically: After ~15 minutes of inactivity")
        print("  - With flag: --close-on-exit flag set")
        print()
        print("Real-world usage:")
        print("  1. Run: python main.py --game_id <id> --config <config>")
        print("  2. If crash occurs, checkpoint is saved")
        print("  3. Resume: python main.py --checkpoint <card_id>")
        print("  4. Game continues from where it left off")
        print("=" * 60)
        
        # Close scorecard to finalize results and make them viewable
        print()
        print("Closing scorecard to finalize results...")
        try:
            if 'game_client_resume' in locals():
                close_response = game_client_resume.close_scorecard(card_id)
                print(f"✅ Scorecard closed: {card_id}")
                print("   Results are now finalized and viewable online")
            elif 'game_client' in locals():
                close_response = game_client.close_scorecard(card_id)
                print(f"✅ Scorecard closed: {card_id}")
                print("   Results are now finalized and viewable online")
        except Exception as e:
            print(f"⚠️  Note: Could not close scorecard: {e}")
            print("   Scorecard may already be closed or expired")
        print()
        
    except Exception as e:
        logger.error(f"Error in crash recovery demo: {e}", exc_info=True)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up - close any remaining clients
        try:
            if 'game_client' in locals():
                game_client.close()
        except:
            pass
        try:
            if 'game_client_resume' in locals():
                game_client_resume.close()
        except:
            pass


if __name__ == "__main__":
    crash_recovery_demo()

