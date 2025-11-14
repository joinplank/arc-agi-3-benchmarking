"""
Simple script that executes two "up" actions in the game.

Initializes a game and executes two "up" (ACTION1) actions.
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arcagi3.game_client import GameClient

load_dotenv()


def execute_two_up_actions(game_id: str = "ls20-fa137e247ce6"):
    """
    Execute two "up" actions in the game.
    
    Args:
        game_id: Game ID to use
    
    Returns:
        Tuple of (card_id, game_id, guid) for continuing the game
    """
    print(f"Executing two 'up' actions for game: {game_id}")
    print("=" * 60)
    
    client = None
    card_id = None
    guid = None
    
    try:
        # Initialize game client
        print("\n1. Initializing game client...")
        client = GameClient()
        print("   Client initialized")
        
        # Open scorecard
        print(f"\n2. Opening scorecard...")
        scorecard_response = client.open_scorecard([game_id])
        card_id = scorecard_response.get("card_id")
        print(f"   card_id = {card_id}")
        
        # Reset game (initializes the game)
        print(f"\n3. Resetting game (initializing)...")
        reset_response = client.reset_game(card_id, game_id)
        guid = reset_response.get("guid")
        state = reset_response.get("state", "UNKNOWN")
        print(f"   guid = {guid}, state = {state}")
        
        # Execute first "up" action
        print(f"\n4. Executing first 'up' action (ACTION1)...")
        action_data = {
            "card_id": card_id,
            "game_id": game_id,
            "guid": guid
        }
        response1 = client.execute_action("ACTION1", action_data)
        state1 = response1.get("state", "UNKNOWN")
        score1 = response1.get("score", 0)
        guid = response1.get("guid", guid)  # Update guid for next action
        print(f"   Action 1 result: state = {state1}, score = {score1}")
        
        # Execute second "up" action
        print(f"\n5. Executing second 'up' action (ACTION1)...")
        action_data["guid"] = guid
        response2 = client.execute_action("ACTION1", action_data)
        state2 = response2.get("state", "UNKNOWN")
        score2 = response2.get("score", 0)
        guid = response2.get("guid", guid)  # Update guid for continuation
        print(f"   Action 2 result: state = {state2}, score = {score2}")
        
        print(f"\n{'=' * 60}")
        print("Two 'up' actions completed successfully!")
        print(f"{'=' * 60}")
        
        # Display scorecard URL for visualization
        if card_id:
            scorecard_url = f"{client.ROOT_URL}/scorecards/{card_id}"
            print(f"\nView your scorecard online: {scorecard_url}")
            print(f"{'=' * 60}")
        
        return card_id, game_id, guid
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    finally:
        # Clean up
        if client:
            try:
                # Note: Scorecard is NOT closed - it remains open for further use
                print(f"\n6. Note: Scorecard {card_id} remains open (not closed)")
                client.close()
                print("   Client session closed")
            except Exception as e:
                print(f"   Failed to close client session: {e}")


def execute_two_left_actions(card_id: str, game_id: str, guid: str):
    """
    Open a new client, connect to the same game using card_id, and execute two "left" actions.
    This function does NOT reset the game - it continues from the current state.
    
    Args:
        card_id: Scorecard ID from previous execution
        game_id: Game ID
        guid: Current game GUID from previous execution
    """
    print(f"\n{'=' * 60}")
    print("Opening new client and executing two 'left' actions")
    print(f"{'=' * 60}")
    
    client = None
    
    try:
        # Initialize new game client
        print("\n1. Initializing new game client...")
        client = GameClient()
        print("   New client initialized")
        
        # Verify scorecard still exists
        print(f"\n2. Verifying scorecard connection...")
        try:
            scorecard_info = client.get_scorecard(card_id, game_id)
            print(f"   ✓ Scorecard {card_id} is accessible")
        except Exception as e:
            print(f"   ⚠ Warning: Could not verify scorecard: {e}")
            print(f"   Continuing anyway...")
        
        # Execute first "left" action (ACTION3) - NO RESET
        print(f"\n3. Executing first 'left' action (ACTION3)...")
        action_data = {
            "card_id": card_id,
            "game_id": game_id,
            "guid": guid
        }
        response1 = client.execute_action("ACTION3", action_data)
        state1 = response1.get("state", "UNKNOWN")
        score1 = response1.get("score", 0)
        guid = response1.get("guid", guid)  # Update guid for next action
        print(f"   Action 1 result: state = {state1}, score = {score1}")
        
        # Execute second "left" action (ACTION3)
        print(f"\n4. Executing second 'left' action (ACTION3)...")
        action_data["guid"] = guid
        response2 = client.execute_action("ACTION3", action_data)
        state2 = response2.get("state", "UNKNOWN")
        score2 = response2.get("score", 0)
        print(f"   Action 2 result: state = {state2}, score = {score2}")
        
        print(f"\n{'=' * 60}")
        print("Two 'left' actions completed successfully!")
        print(f"{'=' * 60}")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if client and card_id:
            try:
                # Close scorecard to make it viewable on the website
                print(f"\n5. Closing scorecard...")
                client.close_scorecard(card_id)
                print(f"   Scorecard {card_id} closed")
                
                # Display scorecard URL for visualization
                scorecard_url = f"{client.ROOT_URL}/scorecards/{card_id}"
                print(f"\nView your scorecard online: {scorecard_url}")
                print(f"{'=' * 60}")
            except Exception as e:
                print(f"   Failed to close scorecard: {e}")
                # Still try to show URL even if close failed
                scorecard_url = f"{client.ROOT_URL}/scorecards/{card_id}"
                print(f"\nView your scorecard online (may not be available yet): {scorecard_url}")
        
        if client:
            try:
                client.close()
                print("   Client session closed")
            except Exception as e:
                print(f"   Failed to close client session: {e}")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not found in environment.")
        print("Please set it in your .env file or environment.")
        sys.exit(1)
    
    # Run with default game_id or allow override via command line
    import argparse
    parser = argparse.ArgumentParser(description="Execute two 'up' actions in the game")
    parser.add_argument(
        "--game-id",
        type=str,
        default="ls20-fa137e247ce6",
        help="Game ID to use (default: ls20-fa137e247ce6)"
    )
    
    args = parser.parse_args()
    
    # Execute two "up" actions and get the game state info
    card_id, game_id, guid = execute_two_up_actions(game_id=args.game_id)
    
    # If successful, continue with two "left" actions using a new client
    if card_id and game_id and guid:
        execute_two_left_actions(card_id, game_id, guid)
    else:
        print("\n⚠ Cannot continue with left actions - missing game state information")

