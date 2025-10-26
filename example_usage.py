"""
Example usage of the ARC-AGI-3 benchmarking framework.

See README.md for complete documentation and usage examples.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the main components
from src.arcagi3.agent import MultimodalAgent
from src.arcagi3.game_client import GameClient
from src.arcagi3.utils import read_models_config


def example_single_game():
    """Example: Run a single game programmatically"""
    print("Example 1: Running a single game")
    print("=" * 60)
    
    # Initialize game client
    game_client = GameClient()
    
    # Get list of available games
    games = game_client.list_games()
    print(f"Available games: {len(games)}")
    for game in games[:3]:  # Show first 3
        print(f"  - {game['game_id']}: {game['title']}")
    
    if not games:
        print("No games available. Check your ARC_API_KEY.")
        return
    
    # Select first game
    game_id = games[0]['game_id']
    print(f"\nPlaying game: {game_id}")
    
    # Configure model
    config = "gpt-4o-mini-2024-07-18"  # Use a cheaper model for testing
    print(f"Using config: {config}")
    
    scorecard_response = game_client.open_scorecard([game_id])
    card_id = scorecard_response.get("card_id")
    print(f"Scorecard created with card_id: {card_id}")
    
    try:
        # Create agent
        agent = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=10,  # Limit for example
            retry_attempts=2,
        )
        
        # Play game
        result = agent.play_game(game_id)
        
        # Print results
        print(f"\n{'=' * 60}")
        print("Game Results:")
        print(f"{'=' * 60}")
        print(f"Final State: {result.final_state}")
        print(f"Final Score: {result.final_score}")
        print(f"Actions Taken: {result.actions_taken}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Total Cost: ${result.total_cost.total_cost:.4f}")
        print(f"Total Tokens: {result.usage.total_tokens}")
        print(f"\nView your scorecard online: {result.scorecard_url}")
        print(f"{'=' * 60}")
        
    finally:
        # Clean up
        game_client.close_scorecard(card_id)
        game_client.close()


def example_list_games():
    """Example: List all available games"""
    print("\nExample 2: Listing all games")
    print("=" * 60)
    
    game_client = GameClient()
    games = game_client.list_games()
    
    print(f"Total games available: {len(games)}\n")
    for game in games:
        print(f"  {game['game_id']:<30} {game['title']}")
    
    game_client.close()


def example_model_configs():
    """Example: List available model configurations"""
    print("\nExample 3: Available model configurations")
    print("=" * 60)
    
    # Read models.yml to show available configs
    import yaml
    models_file = "src/arcagi3/models.yml"
    
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    print(f"Total model configs: {len(config_data['models'])}\n")
    print("Sample configs:")
    for model in config_data['models'][:10]:  # Show first 10
        print(f"  {model['name']:<40} (Provider: {model['provider']})")


if __name__ == "__main__":
    print("ARC-AGI-3 Benchmarking Framework - Examples")
    print("=" * 60)
    print()
    
    # Check if API key is set
    if not os.getenv("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not found in environment.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    # Run examples
    try:
        example_model_configs()
        example_list_games()
        
        # Uncomment to actually run a game (will cost tokens)
        #example_single_game()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

