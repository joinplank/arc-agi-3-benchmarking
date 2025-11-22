import os
from arcagi3.game_client import GameClient
from arcagi3.utils import read_models_config

def list_games():
    # Need a valid card_id. Usually passed via env or args.
    # I'll try to find one or just use a placeholder if the client handles it.
    # The error message "game_id ... not found" suggests the client works but the ID is wrong.
    # I'll try to list games for the current scorecard if possible.
    
    # Check if we can get a scorecard.
    # The main.py uses a card_id.
    # I'll try to use a dummy one or see if I can get one.
    
    # Actually, let's just try to run the agent with a game ID that might exist or just rely on the fact that the code ran.
    # But to be thorough...
    
    client = GameClient()
    # We need a card_id to list games? 
    # client.list_games(card_id)
    
    # If I don't have a card_id, I might need to start a new scorecard.
    scorecard = client.open_scorecard("training") # or "evaluation"
    print(f"Opened scorecard: {scorecard['id']}")
    
    games = client.list_games(scorecard['id'])
    if games:
        print(f"Found game: {games[0]}")
    else:
        print("No games found")

if __name__ == "__main__":
    try:
        list_games()
    except Exception as e:
        print(f"Error: {e}")
