"""
Multimodal Agent for playing ARC-AGI-3 games.

Adapted from the original multimodal agent to use provider adapters.
"""
import json
import os
import logging
import time
from textwrap import dedent
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from PIL import Image

from .adapters import create_provider
from .game_client import GameClient
from .image_utils import grid_to_image, image_to_base64, make_image_block, image_diff
from .schemas import (
    GameAction,
    GameState,
    GameResult,
    GameActionRecord,
    ActionData,
    Cost,
    Usage,
    CompletionTokensDetails,
    StreamResponse
)
from .utils.retry import retry_with_exponential_backoff
from .utils import load_hints, find_hints_file
from .checkpoint import CheckpointManager


logger = logging.getLogger(__name__)


# Map game actions to human-readable descriptions
HUMAN_ACTIONS = {
    "ACTION1": "Move Up",
    "ACTION2": "Move Down",
    "ACTION3": "Move Left",
    "ACTION4": "Move Right",
    "ACTION5": "Perform Action",
    "ACTION6": "Click object on screen (describe object and relative position)",
    "ACTION7": "Undo",
}


HUMAN_ACTIONS_LIST = list(HUMAN_ACTIONS.keys())

def get_human_inputs_text(available_actions: List[str]) -> str:
    """Convert available actions to human-readable text"""
    text = "\n"
    for action in available_actions:
        if action in HUMAN_ACTIONS:
            text += f"{HUMAN_ACTIONS[action]}\n"
    return text


def grid_to_text_matrix(grid: List[List[int]]) -> str:
    """
    Convert a grid matrix to a readable text representation.
    
    Args:
        grid: 64x64 grid of integers (0-15) representing colors
        
    Returns:
        Formatted text representation of the grid
    """
    # Format as JSON for clarity and compactness
    return json.dumps(grid, separators=(',', ','))


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON from various possible formats in the response.
    
    Handles:
    - Bare JSON { ... }
    - Fenced JSON ```json ... ```
    - Generic fence ``` ... ```
    - Wrapper text
    """
    import re
    
    if not response_text or not response_text.strip():
        raise ValueError("Empty response text")
    
    # Try fenced ```json ... ``` blocks (with better regex for multiline)
    fence = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.S | re.I | re.M)
    if fence:
        json_str = fence.group(1).strip()
    else:
        # Try any ``` ... ``` fence
        fence = re.search(r"```[a-z]*\s*(\{.*?\})\s*```", response_text, re.S | re.M)
        if fence:
            json_str = fence.group(1).strip()
        else:
            # Fallback: find the first '{' and match balanced braces
            start = response_text.find("{")
            if start == -1:
                raise ValueError(f"No JSON object detected in response. Response was: {response_text[:200]}")
            
            # Find matching closing brace, skipping strings to avoid false matches
            brace_count = 0
            end = start
            in_string = False
            escape_next = False
            
            for i in range(start, len(response_text)):
                char = response_text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break
            
            if brace_count != 0:
                # If we couldn't find balanced braces, the JSON might be truncated
                # Try to get what we have and let json.loads fail with a better error
                logger.warning(f"Unbalanced braces in JSON (count: {brace_count}). JSON might be truncated.")
                json_str = response_text[start:].strip()
                # Try to close the JSON
                json_str = json_str.rstrip() + "}"
            else:
                json_str = response_text[start : end + 1].strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Clean control characters and try again
        try:
            import unicodedata
            cleaned = ''.join(char if unicodedata.category(char)[0] != 'C' or char in '\n\r\t' else ' ' for char in json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON: {e}. JSON string was: {json_str[:200]}")


class MultimodalAgent:
    """Agent that plays ARC-AGI-3 games using multimodal LLMs"""
    
    SYSTEM_PROMPT = dedent("""\
        You are an abstract reasoning agent that is attempting to solve
        turn-based interactive environments displayed to you as PNGs along
        text for goals, analysis, and planning.
    
        All games have simple abtract graphics and problems that can be 
        solved using nothing but core knowledge.
    """).strip()
    
    ACTION_INSTRUCT = dedent("""\
        Given the frames and the provided game information above, provide
        your desired action as if you were a human playing the game describing
        your next action to an LLM which will figure out how to perform it.
                             
        ```json
        {
            "human_action": "Click on the red square near the bottom left corner",
            "reasoning": "...",
            "expected_result": "..."                             
        }
                             
        These are going to be multistep games, but only concern yourself with
        the next action.  You should favor moves/actions before trying to click
        on objects, only start clicking once you're sure movement/actions do nothing.

                             
        Only response with the JSON, nothing else.
    """).strip()
    
    ANALYZE_INSTRUCT = dedent("""\
        ## Instruct

        Given your action, including your expected outcome, and the provided results
        via the associated images provide a complet analysis of the outcome, thinking
        though what happened.  When analizing the images think about the x,y location
        of objects, their colors, and how they relate to the game state.
                              
        The images attached here are as follows (Zero Indexed):
        - 0: Final Frame before your Action
        - 1-N: Frames as a result of your action.
        - A Helper image showing pixels in red that changed between the Final Frame 
          before your action and the last frame after your action.  Any changes 
          larger than a few pixels should be considered significant.
                              
        When examining the images try to identify objects or environmental patterns
        and their locations.
                              
        Provide your analysis and then after providing `---` update your memory scratchpad.
        The memory scratchpad is a place for you to remember anything that will help you
        play the game better. You can structure it however you want - it's your scratchpad
        to use as you see fit. IMPORTANT: The memory scratchpad should be plain text.
        Use natural language, bullet points, or any text format you prefer. Keep the memory 
        scratchpad concise and within approximately {memory_limit} words to help manage context window size. 
        Focus on what's most important for understanding the game environment and rules to beat 
        the game in as few moves as possible.
        ---
    """).strip()
    
    FIND_ACTION_INSTRUCT = dedent("""\
        Instruct: Given the provided image and the desired action above decide what to do
        base on the following information:                      
        {{action_list}}
        
        ```json
        {
            "action": "ACTION1",
            "x": 0,
            "y": 0
        }
        ```
        Respond with the JSON, nothing else.
    """).strip()
    
    def __init__(
        self,
        config: str,
        game_client: GameClient,
        card_id: str,
        max_actions: int = 40,
        retry_attempts: int = 3,
        num_plays: int = 1,
        use_vision: bool = True,
        memory_word_limit: Optional[int] = None,
        checkpoint_frequency: int = 1,
        checkpoint_card_id: Optional[str] = None,
    ):
        """
        Initialize the multimodal agent.

        Args:
            config: Model configuration name from models.yml
            game_client: GameClient for API communication
            card_id: Scorecard identifier for API calls
            max_actions: Maximum actions to take before stopping
            retry_attempts: Number of retry attempts for failed API calls
            num_plays: Number of times to play the game (continues session with memory)
            use_vision: Whether to use vision (images) or text-only mode
            memory_word_limit: Maximum number of words allowed in memory scratchpad (default: from config or 500)
            checkpoint_frequency: Save checkpoint every N actions (default: 1, 0 to disable)
            checkpoint_card_id: Optional card_id for checkpoint directory (defaults to card_id if not provided)
        """
        self.config = config
        self.game_client = game_client
        self.card_id = card_id
        self.max_actions = max_actions
        self.retry_attempts = retry_attempts
        self.num_plays = num_plays
        
        # Initialize provider adapter (needed to access model config)
        self.provider = create_provider(config)
        
        # Set memory_word_limit: explicit parameter > model config > default (500)
        if memory_word_limit is not None:
            self.memory_word_limit = memory_word_limit
        else:
            self.memory_word_limit = self.provider.model_config.kwargs.get("memory_word_limit", 500)
        
        self.checkpoint_frequency = checkpoint_frequency

        self.hints_file = find_hints_file()
        self.current_game_id: Optional[str] = None
        self.current_hint: Optional[str] = None
        
        # Initialize provider adapter
        self.provider = create_provider(config)
        self._model_supports_vision = bool(getattr(self.provider.model_config, "is_multimodal", False))
        self._use_vision = bool(use_vision and self._model_supports_vision)

        if not self._model_supports_vision:
            if use_vision:
                logger.warning(
                    "Model config `%s` does not support multimodal; continuing without vision.",
                    self.config,
                )
            else:
                logger.info(
                    "Model config `%s` is text-only; vision disabled.",
                    self.config,
                )
        elif self._use_vision:
            logger.info("Vision is enabled for this agent. Images will be used.")
        else:
            logger.warning("Vision is disabled for this agent. Only text will be used.")
        
        # Tracking variables
        self.action_counter = 0
        self.total_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
        self.total_usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_tokens_details=CompletionTokensDetails()
        )
        self.action_history: List[GameActionRecord] = []
        
        # Memory for the agent
        self._available_actions: List[str] = []
        self._memory_prompt = ""
        self._available_actions_prompt = ""  # Store available actions separately
        self._previous_action: Optional[Dict[str, Any]] = None
        self._previous_images: List[Image.Image] = []
        self._previous_grids: List[List[List[int]]] = []  # Store raw grids for text-based providers
        self._resumed_current_grids: Optional[List[List[List[int]]]] = None # Store current grids from resumed checkpoint
        self._previous_score = 0

        self._previous_prompt = ""

        # Checkpoint manager - use checkpoint_card_id if provided, otherwise use card_id
        # This allows resuming from original checkpoint even when scorecard changes
        effective_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id
        self.checkpoint_manager = CheckpointManager(effective_checkpoint_id)

        # Current play tracking (for checkpoint restoration)
        self._current_play = 1
        self._play_action_counter = 0
        self._current_guid: Optional[str] = None

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt, prepending any hint for the current game if available.
        
        Returns:
            System prompt with hint prepended if available
        """
        system_prompt = self.SYSTEM_PROMPT
        
        # Prepend hint if available for current game
        if self.current_hint:
            hint = self.current_hint.strip()
            if hint:
                system_prompt = f"{system_prompt}\n\n ALSO USE these hints in order to complete the game: \n {hint} "
                logger.info(f"Using hint for game {self.current_game_id}")
        
        return system_prompt
        
    def _initialize_memory(self, available_actions: List[str]):
        """Initialize the agent's memory as empty, storing available actions separately"""
        # Memory starts empty - LLM can structure it however it wants
        human_actions = "\n".join(available_actions)
        self._available_actions_prompt = dedent(f"""\
            ## Known Human Game Inputs
{human_actions}
        """).strip()
        self._memory_prompt = ""  # Initialize memory as empty
        logger.info(f"Memory initialized empty, available actions stored separately")
    
    def _get_memory_with_actions(self) -> str:
        """Get memory merged with available actions text"""
        if self._memory_prompt:
            return f"{self._available_actions_prompt}\n\n{self._memory_prompt}"
        return self._available_actions_prompt
    
    def _get_memory_word_count(self) -> int:
        """Get the word count of the current memory"""
        return len(self._memory_prompt.split(" ")) if self._memory_prompt else 0
    
    def _compress_memory(self) -> str:
        """Ask LLM to compress memory if it exceeds the limit"""
        if not self._memory_prompt:
            return ""
        
        current_word_count = self._get_memory_word_count()
        compress_prompt = dedent(f"""\
            Your memory scratchpad has grown too large ({current_word_count} words).
            Please compress it to approximately {self.memory_word_limit} words while keeping
            the most important information for playing the game.
            
            Current memory:
            {self._memory_prompt}
            
            Provide only the compressed memory scratchpad, nothing else.
        """).strip()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": compress_prompt,
            },
        ]
        
        try:
            response = self.provider.call_provider(messages)
            prompt_tokens, completion_tokens = self.provider.extract_usage(response)
            self._update_costs(prompt_tokens, completion_tokens)
            
            compressed = self.provider.extract_content(response).strip()
            compressed_word_count = len(compressed.split(" ")) if compressed else 0
            logger.info(f"Compressed memory from {current_word_count} to {compressed_word_count} words")
            return compressed
        except Exception as e:
            logger.error(f"Failed to compress memory: {e}")
            # Fallback to truncation
            return self._truncate_memory()
    
    def _truncate_memory(self) -> str:
        """Truncate memory to word limit by keeping the first N words"""
        if not self._memory_prompt:
            return ""
        
        words = self._memory_prompt.split(" ")
        if len(words) <= self.memory_word_limit:
            return self._memory_prompt
        
        truncated = " ".join(words[:self.memory_word_limit])
        logger.warning(f"Truncated memory from {len(words)} to {self.memory_word_limit} words")
        return truncated
    
    def _enforce_memory_limit(self):
        """Check memory size and compress or truncate if it exceeds the limit"""
        word_count = self._get_memory_word_count()
        if word_count <= self.memory_word_limit:
            return
        
        logger.info(f"Memory exceeds limit ({word_count} > {self.memory_word_limit} words). Attempting compression...")
        # Try compression first, fallback to truncation if it fails
        self._memory_prompt = self._compress_memory()
        
        # If compression didn't work or still exceeds limit, truncate
        if self._get_memory_word_count() > self.memory_word_limit:
            self._memory_prompt = self._truncate_memory()

    def save_checkpoint(self, current_grids: Optional[List[List[List[int]]]] = None):
        """Save current agent state to checkpoint"""
        try:
            self.checkpoint_manager.save_state(
                config=self.config,
                game_id=self.current_game_id,
                guid=self._current_guid,
                max_actions=self.max_actions,
                retry_attempts=self.retry_attempts,
                num_plays=self.num_plays,
                action_counter=self.action_counter,
                total_cost=self.total_cost,
                total_usage=self.total_usage,
                action_history=self.action_history,
                memory_prompt=self._memory_prompt,
                previous_action=self._previous_action,
                previous_images=self._previous_images,
                previous_score=self._previous_score,
                current_play=self._current_play,
                play_action_counter=self._play_action_counter,
                use_vision=self._use_vision,
                previous_grids=self._previous_grids,
                current_grids=current_grids,
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    def restore_from_checkpoint(self):
        """Restore agent state from checkpoint"""
        logger.info(f"Restoring agent state from checkpoint: {self.checkpoint_manager.card_id}")

        try:
            state = self.checkpoint_manager.load_state()

            # Restore metadata
            metadata = state["metadata"]
            self.current_game_id = metadata["game_id"]
            self._current_guid = metadata.get("guid")
            self.max_actions = metadata["max_actions"]
            self.retry_attempts = metadata["retry_attempts"]
            self.num_plays = metadata["num_plays"]
            self.action_counter = metadata["action_counter"]
            self._current_play = metadata.get("current_play", 1)
            self._play_action_counter = metadata.get("play_action_counter", 0)
            self._previous_score = metadata.get("previous_score", 0)

            # Restore costs and usage
            self.total_cost = state["total_cost"]
            self.total_usage = state["total_usage"]

            # Restore action history
            self.action_history = state["action_history"]

            # Restore memory and state
            self._memory_prompt = state["memory_prompt"]
            self._previous_action = state["previous_action"]
            self._previous_images = state["previous_images"]
            self._previous_grids = state.get("previous_grids", [])
            self._resumed_current_grids = state.get("current_grids", [])

            logger.info(
                f"Restored checkpoint: game_id={self.current_game_id}, "
                f"action_counter={self.action_counter}, "
                f"play={self._current_play}/{self.num_plays}, "
                f"guid={self._current_guid}"
            )

            return True
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}", exc_info=True)
            return False

    def _update_costs(self, prompt_tokens: int, completion_tokens: int):
        """Update cost and usage tracking"""
        # Get pricing from model config
        input_cost_per_token = self.provider.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.provider.model_config.pricing.output / 1_000_000
        
        prompt_cost = prompt_tokens * input_cost_per_token
        completion_cost = completion_tokens * output_cost_per_token
        
        self.total_cost.prompt_cost += prompt_cost
        self.total_cost.completion_cost += completion_cost
        self.total_cost.total_cost += prompt_cost + completion_cost
        
        self.total_usage.prompt_tokens += prompt_tokens
        self.total_usage.completion_tokens += completion_tokens
        self.total_usage.total_tokens += prompt_tokens + completion_tokens

    def _analyze_previous_action(
        self,
        current_frame_images: List[Image.Image],
        current_frame_grids: List[List[List[int]]],
        current_score: int
    ) -> str:
        """Analyze the results of the previous action"""
        if not self._previous_action:
            return "no previous action"
        
        level_complete = ""
        if current_score > self._previous_score:
            level_complete = "NEW LEVEL!!!! - Whatever you did must have been good!"
        
        analyze_prompt = f"{level_complete}\n\n{self.ANALYZE_INSTRUCT.format(memory_limit=self.memory_word_limit)}\n\n{self._get_memory_with_actions()}"
        
        if self._model_supports_vision and self._use_vision:
            # For multimodal providers, use images
            all_imgs = [
                self._previous_images[-1],
                *current_frame_images,
                image_diff(self._previous_images[-1], current_frame_images[-1]),
            ]
            
            # Build message with images
            msg_parts = [
                make_image_block(image_to_base64(img))
                for img in all_imgs
            ] + [{"type": "text", "text": analyze_prompt}]
        else:
            # For text-only providers, use text matrices
            msg_parts = []
            
            # Previous frame
            msg_parts.append({
                "type": "text",
                "text": f"Frame 0 (before action):\n{grid_to_text_matrix(self._previous_grids[-1])}"
            })
            
            # Current frames
            for i, grid in enumerate(current_frame_grids):
                msg_parts.append({
                    "type": "text",
                    "text": f"Frame {i+1} (after action):\n{grid_to_text_matrix(grid)}"
                })
            
            # Add the prompt
            msg_parts.append({"type": "text", "text": analyze_prompt})
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": [{"type": "text", "text": self._previous_prompt}],
            },
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps(self._previous_action)}\n```",
            },
            {
                "role": "user",
                "content": msg_parts,
            },
        ]
        
        response = self.provider.call_provider(messages)
        
        # Track costs - handle different response formats
        prompt_tokens, completion_tokens = self.provider.extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens)
        
        # Extract analysis and update memory
        analysis_message = self.provider.extract_content(response)
        logger.info(f"Analysis: {analysis_message[:200]}...")
        before, _, after = analysis_message.partition("---")
        analysis = before.strip()
        if after.strip():
            self._memory_prompt = after.strip()
            word_count = self._get_memory_word_count()
            logger.info(f"Memory updated ({word_count} words):\n{self._get_memory_with_actions()}")
            # Enforce memory word limit
            self._enforce_memory_limit()
            # Log memory again after enforcement (in case it was compressed/truncated)
            final_word_count = self._get_memory_word_count()
            if final_word_count != word_count:
                logger.info(f"Memory after enforcement ({final_word_count} words):\n{self._get_memory_with_actions()}")
        return analysis
    
    def _choose_human_action(
        self,
        frame_images: List[Image.Image],
        frame_grids: List[List[List[int]]],
        analysis: str
    ) -> Dict[str, Any]:
        """Choose the next human-level action"""
        if len(analysis) > 20:
            self._previous_prompt = f"{analysis}\n\n{self._get_memory_with_actions()}\n\n{self.ACTION_INSTRUCT}"
        else:
            self._previous_prompt = f"{self._get_memory_with_actions()}\n\n{self.ACTION_INSTRUCT}"
        
        if self._model_supports_vision and self._use_vision:
            # For multimodal providers, use images
            content = [
                *[make_image_block(image_to_base64(img)) for img in frame_images],
            ]
        else:
            # For text-only providers, use text matrices
            content = []
            for i, grid in enumerate(frame_grids):
                content.append({
                    "type": "text",
                    "text": f"Frame {i}:\n{grid_to_text_matrix(grid)}"
                })
        content.append({"type": "text", "text": self._previous_prompt})
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": content,
            },
        ]
        
        response = self.provider.call_provider(messages)
        
        # Track costs
        prompt_tokens, completion_tokens = self.provider.extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens)
        
        action_message = self.provider.extract_content(response)

        logger.info(f"Human action: {action_message[:200]}...")
        
        try:
            return extract_json_from_response(action_message)
        except ValueError as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            logger.debug(f"Full response: {action_message}")
            # Re-raise to be caught by game loop
            raise
    
    def _convert_to_game_action(
        self,
        human_action: str,
        last_frame_image: Image.Image,
        last_frame_grid: List[List[int]]
    ) -> Dict[str, Any]:
        """Convert human action description to game action"""
        available_actions = [f"{HUMAN_ACTIONS_LIST[int(a) - 1]} = {HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[int(a) - 1]]}" for a in self._available_actions]
        
        content = []
        if self._model_supports_vision and self._use_vision:
            # For multimodal providers, use image
            content.append(
                make_image_block(image_to_base64(last_frame_image)),
            )
        else:
            # For text-only providers, use text matrix
            content.append(
                {
                    "type": "text",
                    "text": f"Current frame:\n{grid_to_text_matrix(last_frame_grid)}"
                },
            )
        content.append({
            "type": "text",
            "text": human_action + "\n\n" + self.FIND_ACTION_INSTRUCT.replace("{{action_list}}", "\n".join(available_actions)),
        })
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": content,
            },
        ]
        
        response = self.provider.call_provider(messages)
        
        # Track costs
        prompt_tokens, completion_tokens = self.provider.extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens)
        
        action_message = self.provider.extract_content(response)
        logger.info(f"Game action: {action_message[:200]}...")
        
        try:
            return extract_json_from_response(action_message)
        except ValueError as e:
            logger.error(f"Failed to extract JSON from game action response: {e}")
            logger.debug(f"Full response: {action_message}")
            raise
    
    def _execute_game_action(
        self,
        action_name: str,
        action_data: Optional[Dict[str, Any]],
        game_id: str,
        guid: Optional[str],
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute action via game client"""
        data = {"game_id": game_id}
        if guid:
            data["guid"] = guid
        if action_data:
            data.update(action_data)
        if reasoning:
            data["reasoning"] = reasoning
        
        return self.game_client.execute_action(action_name, data)
    
    def play_game(self, game_id: str, resume_from_checkpoint: bool = False) -> GameResult:
        """
        Play a complete game and return results.

        Args:
            game_id: Game identifier to play
            resume_from_checkpoint: If True, resume from existing checkpoint

        Returns:
            GameResult with complete game information (best result if multiple plays)
        """
        # Restore from checkpoint if requested
        if resume_from_checkpoint:
            if not self.checkpoint_manager.checkpoint_exists():
                logger.warning(f"No checkpoint found for {self.checkpoint_manager.card_id}, starting fresh")
                resume_from_checkpoint = False
            else:
                if not self.restore_from_checkpoint():
                    logger.error("Failed to restore checkpoint, starting fresh")
                    resume_from_checkpoint = False
                else:
                    # Use the restored game_id if available
                    if self.current_game_id:
                        game_id = self.current_game_id
                        logger.info(f"Resuming game {game_id} from checkpoint")

        # Store current game ID
        self.current_game_id = game_id

        logger.info(f"Starting game {game_id} with config {self.config} ({self.num_plays} play(s))")
        overall_start_time = time.time()

        # Load hint for this specific game if available (only if not resuming, since hint is already in memory)
        if not resume_from_checkpoint:
            if self.hints_file:
                hints = load_hints(self.hints_file, game_id=game_id)
                self.current_hint = hints.get(game_id) if hints else None
                if self.current_hint:
                    logger.info(f"Found hint for game {game_id}")
                else:
                    logger.debug(f"No hint found for game {game_id}")
            else:
                self.current_hint = None

        best_result: Optional[GameResult] = None
        guid: Optional[str] = self._current_guid if resume_from_checkpoint else None

        # Determine starting play number
        start_play = self._current_play if resume_from_checkpoint else 1

        for play_num in range(start_play, self.num_plays + 1):
            self._current_play = play_num
            play_start_time = time.time()
            
            if play_num > 1:
                logger.info(f"Starting play {play_num}/{self.num_plays} (continuing session with memory)")
            
            # Skip reset if resuming from checkpoint in the middle of a play
            if resume_from_checkpoint and play_num == start_play and self._play_action_counter > 0:
                logger.info(f"Resuming play {play_num} at action {self._play_action_counter}")
                # Try to continue from existing session with guid (WITHOUT calling reset)
                session_restored = False

                # Use the saved GUID directly and continue without reset
                if self._current_guid:
                    guid = self._current_guid
                    current_score = self._previous_score
                    current_state = "IN_PROGRESS"
                    session_restored = True
                    # Create a minimal state structure - will be updated after first action
                    # Use current grids from checkpoint if available (fixes stale state bug), 
                    # otherwise fallback to previous grids (legacy behavior)
                    restored_frame = self._resumed_current_grids if self._resumed_current_grids else self._previous_grids
                    
                    state = {
                        "guid": guid,
                        "score": current_score,
                        "state": current_state,
                        "frame": restored_frame if restored_frame else []
                    }
                    logger.info(f"Continuing session with guid: {guid}, score: {current_score}")
                    logger.info(f"Resuming from action {self._play_action_counter} (no reset sent)")

                if not session_restored:
                    # If we don't have a GUID for some reason, reset with memory
                    logger.info("No GUID found, starting new game session with restored memory...")
                    state = self.game_client.reset_game(self.card_id, game_id, guid=None)
                    guid = state.get("guid")
                    current_score = state.get("score", 0)
                    current_state = state.get("state", "IN_PROGRESS")
                    logger.info(f"New session started (guid: {guid}, keeping {self.action_counter} actions in memory)")

                play_action_counter = self._play_action_counter if session_restored else 0

                # Reconstruct play_action_history from self.action_history for this play
                # Actions for this play have action_num from (action_counter - play_action_counter + 1) to action_counter
                if session_restored and play_action_counter > 0:
                    start_action_num = self.action_counter - play_action_counter + 1
                    end_action_num = self.action_counter
                    play_action_history: List[GameActionRecord] = [
                        action for action in self.action_history
                        if start_action_num <= action.action_num <= end_action_num
                    ]
                    logger.info(f"Reconstructed {len(play_action_history)} action(s) for current play from checkpoint")
                else:
                    # Session not restored or no actions yet in this play
                    play_action_history: List[GameActionRecord] = []

                resume_from_checkpoint = False  # Only skip reset once
            else:
                # Reset game (use guid to continue session if not first play)
                state = self.game_client.reset_game(self.card_id, game_id, guid=guid)
                guid = state.get("guid")
                current_score = state.get("score", 0)
                current_state = state.get("state", "IN_PROGRESS")

                # Initialize memory only on first play, otherwise keep existing memory
                if play_num == 1 and not self._memory_prompt:
                    self._available_actions = state.get("available_actions", list(HUMAN_ACTIONS.keys()))
                    available_codes = [f"{HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[int(a) - 1]]}" for a in self._available_actions]
                    self._initialize_memory(available_codes)
                else:
                    logger.info(f"Continuing with memory from previous play(s)")

                # Reset play-specific counters (but keep cumulative cost/usage)
                play_action_counter = 0
                play_action_history: List[GameActionRecord] = []

            # Store guid
            self._current_guid = guid
            self._play_action_counter = play_action_counter
            
            # Main game loop
            while (
                current_state not in ["WIN", "GAME_OVER"]
                and play_action_counter < self.max_actions
            ):
                try:
                    frames = state.get("frame", [])
                    if not frames:
                        logger.warning("No frames in state, breaking")
                        break
                    
                    # Store raw grids and convert to images
                    frame_grids = frames  # frames are already grid matrices from API
                    frame_images = [grid_to_image(frame) for frame in frames]
                    
                    analysis = self._analyze_previous_action(frame_images, frame_grids, current_score)
                    
                    human_action_dict = self._choose_human_action(frame_images, frame_grids, analysis)
                    human_action = human_action_dict.get("human_action")
                    
                    if not human_action:
                        logger.error("No human_action in response")
                        break
                    
                    game_action_dict = self._convert_to_game_action(human_action, frame_images[-1], frame_grids[-1])
                    action_name = game_action_dict.get("action")
                    
                    if not action_name:
                        logger.error("No action name in response")
                        break
                    
                    action_data_dict = {}
                    if action_name == "ACTION6":
                        x = game_action_dict.get("x", 0)
                        y = game_action_dict.get("y", 0)
                        action_data_dict = {
                            "x": max(0, min(x, 127)) // 2,
                            "y": max(0, min(y, 127)) // 2,
                        }
                    
                    action_field = action_name
                    if action_name == "ACTION6" and action_data_dict:
                        action_field = f"{action_name}: [{action_data_dict}]"

                    reasoning_for_api = {
                        "analysis": analysis[:1000] if len(analysis) > 1000 else analysis,
                        "action": action_field,
                        "human_action": human_action,
                        "reasoning": (human_action_dict.get("reasoning", "") or "")[:300],
                        "expected": (human_action_dict.get("expected_result", "") or "")[:300],
                        "tokens:": [self.total_usage.prompt_tokens, self.total_usage.completion_tokens],
                    }
                    state = self._execute_game_action(action_name, action_data_dict, game_id, guid, reasoning_for_api)
                    guid = state.get("guid", guid)
                    new_score = state.get("score", current_score)
                    current_state = state.get("state", "IN_PROGRESS")
                    new_frames = state.get("frame", [])
                    
                    self.action_counter += 1
                    action_record = GameActionRecord(
                        action_num=self.action_counter,
                        action=action_name,
                        action_data=ActionData(**action_data_dict) if action_data_dict else None,
                        reasoning={
                            "human_action": human_action,
                            "reasoning": human_action_dict.get("reasoning", ""),
                            "expected": human_action_dict.get("expected_result", ""),
                            "analysis": analysis[:500] if len(analysis) > 500 else analysis,
                        },
                        result_score=new_score,
                        result_state=current_state,
                    )
                    play_action_history.append(action_record)
                    self.action_history.append(action_record)
                    
                    self._previous_action = human_action_dict
                    self._previous_images = frame_images
                    self._previous_grids = frame_grids
                    self._previous_score = current_score
                    current_score = new_score
                    play_action_counter += 1
                    self._play_action_counter = play_action_counter

                    logger.info(
                        f"Play {play_num}, Action {play_action_counter}: {action_name}, "
                        f"Score: {current_score}, State: {current_state}"
                    )

                    # Save checkpoint periodically
                    if self.checkpoint_frequency > 0 and play_action_counter % self.checkpoint_frequency == 0:
                        logger.info(f"Saving checkpoint at action {play_action_counter}")
                        self.save_checkpoint(current_grids=new_frames)

                except Exception as e:
                    logger.error(f"Error during game loop: {e}", exc_info=True)
                    break
            
            play_duration = time.time() - play_start_time
            scorecard_url = f"{self.game_client.ROOT_URL}/scorecards/{self.card_id}"
            
            play_result = GameResult(
                game_id=game_id,
                config=self.config,
                final_score=current_score,
                final_state=current_state,
                actions_taken=play_action_counter,
                duration_seconds=play_duration,
                total_cost=self.total_cost,
                usage=self.total_usage,
                actions=play_action_history,
                final_memory=self._get_memory_with_actions(),
                timestamp=datetime.now(timezone.utc),
                scorecard_url=scorecard_url
            )
            
            logger.info(
                f"Play {play_num}/{self.num_plays} completed: {current_state}, "
                f"Score: {current_score}, Actions: {play_action_counter}"
            )
            
            # Track best result (WIN > highest score)
            if best_result is None:
                best_result = play_result
            elif current_state == "WIN" and best_result.final_state != "WIN":
                best_result = play_result
            elif current_state == "WIN" and best_result.final_state == "WIN":
                if current_score > best_result.final_score:
                    best_result = play_result
            elif current_score > best_result.final_score:
                best_result = play_result

            # Save checkpoint after play completes
            # Use frames from state which should be current
            current_frames = state.get("frame", [])
            self.save_checkpoint(current_grids=current_frames)

            # Stop if we won or no more plays
            if current_state == "WIN":
                logger.info(f"Game won on play {play_num}! Stopping early.")
                break

            if play_num < self.num_plays:
                logger.info(f"Play {play_num} ended ({current_state}). Continuing to next play...")
        
        overall_duration = time.time() - overall_start_time
        
        # Update best result with overall stats
        best_result.actions_taken = self.action_counter
        best_result.duration_seconds = overall_duration
        
        logger.info(
            f"All plays completed. Best: {best_result.final_state}, "
            f"Score: {best_result.final_score}, Total Actions: {self.action_counter}, "
            f"Cost: ${self.total_cost.total_cost:.4f}"
        )

        # Clean up checkpoint on successful completion
        if best_result.final_state == "WIN":
            logger.info("Game won, deleting checkpoint")
            self.checkpoint_manager.delete_checkpoint()

        return best_result

