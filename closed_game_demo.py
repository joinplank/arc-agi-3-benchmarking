"""
Demo: Simulate continuing to send actions to a finished and closed game.

This demonstrates what happens when you try to:
1. Complete a game (or close the scorecard)
2. Continue sending actions to that closed/finished game
3. Test error handling and edge cases

This tests the system's behavior when attempting to interact with:
- A closed scorecard
- A game in WIN or GAME_OVER state
"""
import os
import logging
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


def closed_game_demo():
    """Demonstrate continuing actions on a closed/finished game"""
    print("=" * 60)
    print("Closed Game Demo: Continue Actions After Game Finished/Closed")
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
        
        # Select first game
        game_id = games[0]['game_id']
        print(f"Selected game: {game_id}")
        print(f"Game title: {games[0]['title']}")
        print()
        
        # Configure model (use a cheaper model for demo)
        config = "gpt-4o-mini-2024-07-18"
        model_config = read_models_config(config)
        print(f"Using model: {config}")
        print()
        
        # Generate tags for scorecard
        tags = generate_scorecard_tags(model_config)
        
        # ============================================================
        # PART 1: Play game until it finishes or close scorecard
        # ============================================================
        print("=" * 60)
        print("PART 1: Playing game until finish or closing scorecard")
        print("=" * 60)
        print()
        
        # Open scorecard
        scorecard_response = game_client.open_scorecard([game_id], tags=tags)
        card_id = scorecard_response.get("card_id")
        print(f"Scorecard created: {card_id}")
        print()
        
        # Create agent with limited actions to simulate quick finish
        # Or we can just close the scorecard after a few actions
        agent = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=3,  # Limit to 3 actions for demo
            retry_attempts=2,
            checkpoint_frequency=1,
        )
        
        print("Playing game with max_actions=3...")
        print()
        
        # Play game
        result = agent.play_game(game_id)
        
        print()
        print("PART 1 Results:")
        print(f"  Actions taken: {result.actions_taken}")
        print(f"  Final score: {result.final_score}")
        print(f"  Final state: {result.final_state}")
        print()
        
        # Now close the scorecard
        print("Closing scorecard...")
        try:
            close_response = game_client.close_scorecard(card_id)
            print(f"Scorecard closed successfully: {close_response}")
        except Exception as e:
            print(f"Error closing scorecard: {e}")
        print()
        
        # ============================================================
        # PART 2: Try to continue sending actions to closed game
        # ============================================================
        print("=" * 60)
        print("PART 2: Attempting to continue actions on closed game")
        print("=" * 60)
        print()
        
        # Try to get the game state first
        print("Attempting to get game state from closed scorecard...")
        try:
            scorecard_info = game_client.get_scorecard(card_id)
            print(f"Scorecard info retrieved: {scorecard_info}")
        except Exception as e:
            print(f"❌ Error getting scorecard: {e}")
            print("   (Expected: scorecard is closed)")
        print()
        
        # Try to reset the game with closed scorecard
        print("Attempting to reset game with closed scorecard...")
        try:
            state = game_client.reset_game(card_id, game_id, guid=None)
            print(f"✅ Reset successful: {state.get('state')}, Score: {state.get('score')}")
        except Exception as e:
            print(f"❌ Error resetting game: {e}")
            print("   (Expected: cannot reset with closed scorecard)")
        print()
        
        # Try to execute an action with closed scorecard
        print("Attempting to execute action with closed scorecard...")
        try:
            # First try to get a valid guid by resetting (if that worked)
            # Otherwise, we'll try with a dummy guid
            action_data = {
                "card_id": card_id,
                "game_id": game_id,
                "guid": None  # Will fail if scorecard is closed
            }
            response = game_client.execute_action("ACTION1", action_data)
            print(f"✅ Action executed: {response.get('state')}, Score: {response.get('score')}")
        except Exception as e:
            print(f"❌ Error executing action: {e}")
            print("   (Expected: cannot execute actions with closed scorecard)")
        print()
        
        # ============================================================
        # PART 3: Try to create new agent and continue
        # ============================================================
        print("=" * 60)
        print("PART 3: Attempting to create new agent with closed scorecard")
        print("=" * 60)
        print()
        
        # Create a new agent instance with the closed scorecard
        print("Creating new agent with closed scorecard...")
        try:
            agent_closed = MultimodalAgent(
                config=config,
                game_client=game_client,
                card_id=card_id,
                max_actions=5,
                retry_attempts=2,
                checkpoint_frequency=1,
            )
            
            print("Attempting to play game with closed scorecard...")
            result_closed = agent_closed.play_game(game_id)
            print(f"✅ Game played: {result_closed.final_state}, Score: {result_closed.final_score}")
        except Exception as e:
            print(f"❌ Error playing game with closed scorecard: {e}")
            print("   (Expected: cannot play with closed scorecard)")
        print()
        
        # ============================================================
        # PART 4: Try to open a new scorecard and continue
        # ============================================================
        print("=" * 60)
        print("PART 4: Opening new scorecard to continue same game")
        print("=" * 60)
        print()
        
        # Open a new scorecard for the same game
        print("Opening new scorecard for the same game...")
        try:
            new_scorecard_response = game_client.open_scorecard([game_id], tags=tags)
            new_card_id = new_scorecard_response.get("card_id")
            print(f"✅ New scorecard created: {new_card_id}")
            
            # Create agent with new scorecard
            agent_new = MultimodalAgent(
                config=config,
                game_client=game_client,
                card_id=new_card_id,
                max_actions=5,
                retry_attempts=2,
                checkpoint_frequency=1,
            )
            
            print("Playing game with new scorecard...")
            result_new = agent_new.play_game(game_id)
            print(f"✅ Game played with new scorecard:")
            print(f"   Final state: {result_new.final_state}")
            print(f"   Final score: {result_new.final_score}")
            print(f"   Actions taken: {result_new.actions_taken}")
            
            # Clean up new scorecard
            game_client.close_scorecard(new_card_id)
            print(f"Closed new scorecard: {new_card_id}")
        except Exception as e:
            print(f"❌ Error with new scorecard: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # ============================================================
        # Summary
        # ============================================================
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Original game ID: {game_id}")
        print(f"Original card ID: {card_id} (CLOSED)")
        print(f"Original final state: {result.final_state}")
        print(f"Original final score: {result.final_score}")
        print()
        print("Key findings:")
        print("  - Closed scorecards cannot be used for new actions")
        print("  - Need to open a new scorecard to continue playing")
        print("  - Game state is tied to the scorecard session")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in closed game demo: {e}", exc_info=True)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            game_client.close()
        except:
            pass


if __name__ == "__main__":
    closed_game_demo()

