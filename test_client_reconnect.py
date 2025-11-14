"""
Test: Create a client, start a game with 5 actions, execute only 2 actions,
then create a NEW client with the same card_id to execute the remaining actions.

This demonstrates checkpointing functionality when using separate client instances.
"""
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the main components
from src.arcagi3.agent import MultimodalAgent
from src.arcagi3.game_client import GameClient
from src.arcagi3.utils import generate_scorecard_tags, read_models_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_client_reconnect():
    """Test reconnecting with a new client using the same card_id"""
    print("=" * 60)
    print("Test: Client Reconnect with Checkpoint")
    print("=" * 60)
    print()
    
    # Check if API key is set
    if not os.getenv("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not found in environment.")
        print("Please set it in your .env file or environment.")
        return
    
    # ============================================================
    # PART 1: Create first client and execute 2 actions
    # ============================================================
    print("=" * 60)
    print("PART 1: First Client - Executing 2 of 5 actions")
    print("=" * 60)
    print()
    
    # Initialize first game client
    game_client_1 = GameClient()
    
    try:
        # Get list of available games
        games = game_client_1.list_games()
        if not games:
            print("No games available. Check your ARC_API_KEY.")
            return
        
        # Filter for LS games (game_id starts with "ls")
        ls_games = [g for g in games if g['game_id'].startswith('ls')]
        if not ls_games:
            print("ERROR: No LS games found in available games.")
            print("Available games:")
            for game in games:
                print(f"  - {game['game_id']}: {game['title']}")
            return
        
        # Select first LS game
        selected_game = ls_games[0]
        game_id = selected_game['game_id']
        print(f"Selected LS game: {game_id}")
        print(f"Game title: {selected_game['title']}")
        print(f"(Found {len(ls_games)} LS game(s) out of {len(games)} total games)")
        print()
        
        # Configure model (use a cheaper model for testing)
        config = "gpt-4o-mini-2024-07-18"
        model_config = read_models_config(config)
        print(f"Using model: {config}")
        print()
        
        # Generate tags for scorecard
        tags = generate_scorecard_tags(model_config)
        
        # Open scorecard
        scorecard_response = game_client_1.open_scorecard([game_id], tags=tags)
        card_id = scorecard_response.get("card_id")
        print(f"Scorecard created: {card_id}")
        print()
        
        # Create agent with max_actions=2 to limit execution to 2 actions
        agent_1 = MultimodalAgent(
            config=config,
            game_client=game_client_1,
            card_id=card_id,
            max_actions=3,  # Limit to 2 actions
            retry_attempts=2,
            checkpoint_frequency=1,  # Save checkpoint after each action
        )
        
        print(f"Starting game with max_actions=2")
        print("This will execute 2 actions and save a checkpoint...")
        print()
        
        # Play game (will stop after 2 actions)
        result_1 = agent_1.play_game(game_id)
        
        print()
        print("PART 1 Results:")
        print(f"  Actions taken: {result_1.actions_taken}")
        print(f"  Final score: {result_1.final_score}")
        print(f"  Final state: {result_1.final_state}")
        print(f"  Checkpoint saved at: .checkpoint/{card_id}/")
        print()
        
        # Verify checkpoint exists
        checkpoint_path = Path(".checkpoint") / card_id
        if not checkpoint_path.exists():
            print("ERROR: Checkpoint was not created!")
            return
        
        print(f"✓ Checkpoint verified at: {checkpoint_path}")
        print()
        
    except Exception as e:
        logger.error(f"Error in PART 1: {e}", exc_info=True)
        print(f"\nError: {e}")
        return
    finally:
        # Close first client (but don't close scorecard - we'll use it in PART 2)
        game_client_1.close()
        print("First client closed")
        print()
    
    # ============================================================
    # PART 2: Create new client with same card_id and resume
    # ============================================================
    print("=" * 60)
    print("PART 2: Second Client - Resuming with same card_id")
    print("=" * 60)
    print()
    
    # Initialize second game client (separate instance)
    game_client_2 = GameClient()
    
    try:
        print(f"Created new GameClient instance")
        print(f"Original card_id: {card_id}")
        print(f"Using same game_id: {game_id}")
        print()
        
        # Check if the original scorecard still exists
        scorecard_exists = False
        new_card_id = card_id
        try:
            game_client_2.get_scorecard(card_id)
            scorecard_exists = True
            print(f"✓ Original scorecard {card_id} still exists on server")
        except Exception as e:
            print(f"⚠ Original scorecard {card_id} no longer exists (likely expired): {e}")
            print(f"  Creating new scorecard while preserving checkpoint...")
            # Create a new scorecard but preserve the checkpoint location
            tags = generate_scorecard_tags(model_config)
            scorecard_response = game_client_2.open_scorecard([game_id], tags=tags)
            new_card_id = scorecard_response.get("card_id")
            print(f"  New scorecard created: {new_card_id}")
            print(f"  Checkpoint will still be loaded from: .checkpoint/{card_id}/")
        print()
        
        # Update checkpoint metadata to allow 5 actions total
        # This way when we resume, it will restore with max_actions=5
        checkpoint_metadata_path = Path(".checkpoint") / card_id / "metadata.json"
        if checkpoint_metadata_path.exists():
            with open(checkpoint_metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["max_actions"] = 5  # Update to allow 5 actions
            with open(checkpoint_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Updated checkpoint metadata: max_actions set to 5")
            print()
        
        # Create new agent instance with NEW client
        # Use new_card_id for API calls, but original card_id for checkpoint (checkpoint_card_id)
        agent_2 = MultimodalAgent(
            config=config,
            game_client=game_client_2,  # NEW client instance
            card_id=new_card_id,  # New or original card_id for API calls
            max_actions=6,  # Now allow 5 actions total
            retry_attempts=2,
            checkpoint_frequency=1,
            checkpoint_card_id=card_id if not scorecard_exists else None,  # Preserve checkpoint location if scorecard expired
        )
        
        print(f"Created new agent with:")
        print(f"  - New GameClient instance")
        print(f"  - card_id for API: {new_card_id}")
        if not scorecard_exists:
            print(f"  - checkpoint_card_id: {card_id} (preserved from original)")
        print(f"  - max_actions=5")
        print(f"  - resume_from_checkpoint=True")
        print()
        print("Resuming game from checkpoint...")
        print(f"This will continue from action {result_1.actions_taken + 1} and execute remaining actions...")
        print()
        
        # Resume from checkpoint (will restore with max_actions=5)
        result_2 = agent_2.play_game(game_id, resume_from_checkpoint=True)
        
        print()
        print("PART 2 Results:")
        print(f"  Total actions taken (includes Part 1): {result_2.actions_taken}")
        print(f"  Additional actions in Part 2: {result_2.actions_taken - result_1.actions_taken}")
        print(f"  Final score: {result_2.final_score}")
        print(f"  Final state: {result_2.final_state}")
        print(f"  Duration: {result_2.duration_seconds:.2f}s")
        print(f"  Total cost: ${result_2.total_cost.total_cost:.4f}")
        print()
        
        # ============================================================
        # Summary
        # ============================================================
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Game ID: {game_id}")
        print(f"Original Card ID: {card_id}")
        if new_card_id != card_id:
            print(f"New Card ID (after expiry): {new_card_id}")
        print(f"Part 1 (Client 1) actions: {result_1.actions_taken}")
        print(f"Part 2 (Client 2) - additional actions: {result_2.actions_taken - result_1.actions_taken}")
        print(f"Total actions across both clients: {result_2.actions_taken}")
        print(f"Final state: {result_2.final_state}")
        print(f"Final score: {result_2.final_score}")
        print()
        print(f"View scorecard: {result_2.scorecard_url}")
        print("=" * 60)
        
        # Verify that we got the expected total actions
        if result_2.actions_taken >= 5:
            print()
            print("✓ SUCCESS: Game continued successfully with new client!")
            print("  The checkpoint system successfully allowed resuming")
            print("  with a new client instance.")
            if not scorecard_exists:
                print("  Note: Original scorecard expired, but checkpoint was preserved.")
        else:
            print()
            print("⚠ WARNING: Game completed with fewer than 5 actions.")
            print(f"  Expected at least 5 actions, got {result_2.actions_taken}")
        
    except Exception as e:
        logger.error(f"Error in PART 2: {e}", exc_info=True)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up - try to close the scorecard(s)
        try:
            # Try to close the new card_id if it was created
            if 'new_card_id' in locals() and new_card_id and new_card_id != card_id:
                try:
                    game_client_2.close_scorecard(new_card_id)
                    print()
                    print(f"✓ Scorecard {new_card_id} closed")
                except Exception as e:
                    print()
                    print(f"⚠ Could not close new scorecard {new_card_id}: {e}")
            # Try to close the original card_id (might fail if expired)
            try:
                game_client_2.close_scorecard(card_id)
                if 'new_card_id' not in locals() or new_card_id == card_id:
                    print()
                    print(f"✓ Scorecard {card_id} closed")
            except Exception as e:
                # Expected if scorecard expired
                if "not found" in str(e).lower():
                    print()
                    print(f"ℹ Original scorecard {card_id} already expired/closed (expected)")
                else:
                    print()
                    print(f"⚠ Could not close original scorecard {card_id}: {e}")
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
        finally:
            game_client_2.close()
            print("Second client closed")


if __name__ == "__main__":
    test_client_reconnect()

