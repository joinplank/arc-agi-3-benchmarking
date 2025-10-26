"""
ARC-AGI-3 Game Client for API communication.

Based on https://docs.arcprize.org/api-reference/
"""
import json
import logging
import os
from typing import Dict, Any, List, Optional
import requests
from requests import Response, Session


logger = logging.getLogger(__name__)


class GameClient:
    """Client for interacting with the ARC-AGI-3 API"""
    
    ROOT_URL = "https://three.arcprize.org"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the game client.
        
        Args:
            api_key: ARC API key. If not provided, reads from ARC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ARC_API_KEY")
        if not self.api_key:
            raise ValueError("ARC_API_KEY not found in environment or parameters")
        
        self.headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self._session = Session()
        self._session.headers.update(self.headers)
    
    def list_games(self) -> List[Dict[str, str]]:
        """
        Get list of available games.
        
        Returns:
            List of game dictionaries with 'game_id' and 'title'
            
        Example response:
            [
                {"game_id": "ls20-016295f7601e", "title": "LS20"},
                {"game_id": "ft09-16726c5b26ff", "title": "FT09"}
            ]
        """
        url = f"{self.ROOT_URL}/api/games"
        response = self._session.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to list games: {response.text}")
            response.raise_for_status()
        
        return response.json()
    
    def open_scorecard(self, card_id: str, game_ids: List[str]) -> Dict[str, Any]:
        """
        Open a new scorecard for tracking game progress.
        
        Args:
            card_id: Unique identifier for this scorecard
            game_ids: List of game IDs to include in the scorecard
            
        Returns:
            Scorecard response from API
        """
        url = f"{self.ROOT_URL}/api/scorecard/open"
        data = {
            "card_id": card_id,
            "game_ids": game_ids
        }
        
        response = self._session.post(url, json=data)
        
        if response.status_code != 200:
            logger.error(f"Failed to open scorecard: {response.text}")
            response.raise_for_status()
        
        return response.json()
    
    def close_scorecard(self, card_id: str) -> Dict[str, Any]:
        """
        Close a scorecard.
        
        Args:
            card_id: Scorecard identifier
            
        Returns:
            Response from API
        """
        url = f"{self.ROOT_URL}/api/scorecard/close"
        data = {"card_id": card_id}
        
        response = self._session.post(url, json=data)
        
        if response.status_code != 200:
            logger.error(f"Failed to close scorecard: {response.text}")
            response.raise_for_status()
        
        return response.json()
    
    def get_scorecard(self, card_id: str, game_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve scorecard information.
        
        Args:
            card_id: Scorecard identifier
            game_id: Optional specific game ID to get scorecard for
            
        Returns:
            Scorecard data from API
        """
        if game_id:
            url = f"{self.ROOT_URL}/api/scorecard/{card_id}/{game_id}"
        else:
            url = f"{self.ROOT_URL}/api/scorecard/{card_id}"
        
        response = self._session.get(url, timeout=5)
        
        if response.status_code != 200:
            logger.error(f"Failed to get scorecard: {response.text}")
            response.raise_for_status()
        
        return response.json()
    
    def execute_action(
        self,
        action: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a game action.
        
        Args:
            action: Action name (e.g., "RESET", "ACTION1", "ACTION6")
            data: Action data including guid, game_id, and optional x/y for ACTION6
            
        Returns:
            Game state response from API
        """
        url = f"{self.ROOT_URL}/api/cmd/{action}"
        
        action_data = data or {}
        
        response = self._session.post(url, json=action_data)
        
        if response.status_code != 200:
            logger.error(f"Failed to execute action {action}: {response.text}")
            response.raise_for_status()
        
        response_data = response.json()
        
        if "error" in response_data:
            logger.warning(f"API returned error for action {action}: {response_data['error']}")
        
        return response_data
    
    def reset_game(self, card_id: str, game_id: str, guid: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset a game to initial state.
        
        Args:
            card_id: Scorecard identifier
            game_id: Game identifier
            guid: Optional game instance GUID from previous game state
            
        Returns:
            Initial game state
        """
        data = {
            "card_id": card_id,
            "game_id": game_id
        }
        if guid:
            data["guid"] = guid
        
        return self.execute_action("RESET", data)
    
    def close(self):
        """Close the session"""
        if hasattr(self, '_session'):
            self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

