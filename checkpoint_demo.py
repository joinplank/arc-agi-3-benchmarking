"""
Demo: Simulate a game with 5 actions, execute 2, then resume from checkpoint.

This demonstrates the checkpointing functionality:
1. Start a game with max_actions=5
2. Execute only 2 actions (by limiting max_actions to 2)
3. Save checkpoint
4. Resume from checkpoint with max_actions=5 to complete remaining 3 actions
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


def checkpoint_demo():
    """Demonstrate checkpointing with a 5-action game"""
    print("=" * 60)
    print("Checkpoint Demo: 5 Actions Game")
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
        
        # Open scorecard
        scorecard_response = game_client.open_scorecard([game_id], tags=tags)
        card_id = scorecard_response.get("card_id")
        print(f"Scorecard created: {card_id}")
        print()
        
        # ============================================================
        # PART 1: Execute first 2 actions
        # ============================================================
        print("=" * 60)
        print("PART 1: Executing first 2 actions")
        print("=" * 60)
        print()
        
        # Create agent with max_actions=2 to limit execution
        agent = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=2,  # Limit to 2 actions
            retry_attempts=2,
            checkpoint_frequency=1,  # Save checkpoint after each action
        )
        
        print(f"Starting game with max_actions=2")
        print("This will execute 2 actions and save a checkpoint...")
        print()
        
        # Play game (will stop after 2 actions)
        result = agent.play_game(game_id)
        
        print()
        print("PART 1 Results:")
        print(f"  Actions taken: {result.actions_taken}")
        print(f"  Final score: {result.final_score}")
        print(f"  Final state: {result.final_state}")
        print(f"  Checkpoint saved at: .checkpoint/{card_id}/")
        print()
        
        # ============================================================
        # PART 2: Resume from checkpoint and complete remaining actions
        # ============================================================
        print("=" * 60)
        print("PART 2: Resuming from checkpoint")
        print("=" * 60)
        print()
        
        # Create new agent instance with max_actions=5 to allow completion
        # and resume from checkpoint
        agent_resume = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=5,  # Now allow 5 actions total
            retry_attempts=2,
            checkpoint_frequency=1,
        )
        
        print(f"Resuming game with max_actions=5")
        print("This will continue from checkpoint and execute remaining actions...")
        print()
        
        # Update checkpoint metadata to allow 5 actions total
        # This way when we resume, it will restore with max_actions=5
        checkpoint_path = Path(".checkpoint") / card_id / "metadata.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                metadata = json.load(f)
            metadata["max_actions"] = 5  # Update to allow 5 actions
            with open(checkpoint_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Updated checkpoint metadata: max_actions set to 5")
            print()
        
        # Resume from checkpoint (will restore with max_actions=5)
        result_resume = agent_resume.play_game(game_id, resume_from_checkpoint=True)
        
        print()
        print("PART 2 Results:")
        print(f"  Total actions taken: {result_resume.actions_taken}")
        print(f"  Final score: {result_resume.final_score}")
        print(f"  Final state: {result_resume.final_state}")
        print(f"  Duration: {result_resume.duration_seconds:.2f}s")
        print(f"  Total cost: ${result_resume.total_cost.total_cost:.4f}")
        print()
        
        # ============================================================
        # Summary
        # ============================================================
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Game ID: {game_id}")
        print(f"Card ID: {card_id}")
        print(f"Part 1 actions: {result.actions_taken}")
        print(f"Part 2 actions: {result_resume.actions_taken}")
        print(f"Total actions: {result_resume.actions_taken}")
        print(f"Final state: {result_resume.final_state}")
        print(f"Final score: {result_resume.final_score}")
        print()
        print(f"View scorecard: {result_resume.scorecard_url}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in checkpoint demo: {e}", exc_info=True)
        print(f"\nError: {e}")
    finally:
        # Clean up
        try:
            game_client.close_scorecard(card_id)
        except:
            pass
        game_client.close()


if __name__ == "__main__":
    checkpoint_demo()

