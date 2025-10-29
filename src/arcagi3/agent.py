"""
Multimodal Agent for playing ARC-AGI-3 games.

Adapted from the original multimodal agent to use provider adapters.
"""
import json
import logging
import time
from textwrap import dedent
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from PIL import Image

from .adapters import create_provider
from .game_client import GameClient
from .image_utils import grid_to_image, image_to_base64, make_image_block, image_diff
from .memory import GameMemory
from .schemas import (
    GameAction,
    GameState,
    GameResult,
    GameActionRecord,
    ActionData,
    Cost,
    Usage,
    CompletionTokensDetails,
)
from .utils.retry import retry_with_exponential_backoff


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


def get_human_inputs_text(available_actions: List[str]) -> str:
    """Convert available actions to human-readable text"""
    text = ""
    for action in available_actions:
        if action in HUMAN_ACTIONS:
            text += f"\n- {HUMAN_ACTIONS[action]}"
    return text


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
    
    # Try fenced ```json ... ``` blocks
    fence = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.S | re.I)
    if fence:
        json_str = fence.group(1)
    else:
        # Try any ``` ... ``` fence
        fence = re.search(r"```\s*(\{.*?\})\s*```", response_text, re.S)
        if fence:
            json_str = fence.group(1)
        else:
            # Fallback: first '{' to last '}'
            start, end = response_text.find("{"), response_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object detected in response")
            json_str = response_text[start : end + 1]
    
    return json.loads(json_str)


class MultimodalAgent:
    """Agent that plays ARC-AGI-3 games using multimodal LLMs"""
    
    SYSTEM_PROMPT = dedent("""\
        You are an abstract reasoning agent that is attempting to solve
        turn-based interactive environments displayed to you as PNGs along
        with text for goals, analysis, and planning.
    
        All games have simple abstract graphics and problems that can be 
        solved using nothing but core knowledge.
    """).strip()
    
    ACTION_INSTRUCT = dedent("""\
        Given the frames and the provided game information above, provide
        your desired action as if you were a human playing the game describing
        your next action to an LLM which will figure out how to perform it.
                             
        Return JSON in this format:
        {
            "human_action": "Click on the red square near the bottom left corner",
            "reasoning": "Explain your reasoning here",
            "expected_result": "Describe what you expect to happen"
        }
                             
        These are multistep games - only concern yourself with the next action.
        Favor moves/actions before trying to click on objects. Only start clicking
        once you're sure movement/actions do nothing.
                             
        Only respond with the JSON, nothing else.
    """).strip()
    
    ANALYZE_INSTRUCT = dedent("""\
        ## Analysis Instructions

        Given your previous action, expected outcome, and the actual results
        shown in the images, provide a complete analysis of what happened.
                              
        The images attached are (zero indexed):
        - 0: Final frame before your action
        - 1-N: Frames resulting from your action
        - Last: Helper image showing pixels in red that changed between before and after.
          Any changes larger than a few pixels should be considered significant.
                              
        When examining images, identify objects, environmental patterns, and their locations.
                              
        Provide your analysis, then after a line with "---", update the game information
        below while keeping the structure intact. Include what you've tried and what
        you'd like to try in the future. Note that "Known Human Game Inputs" should
        never be changed as these are provided by the game itself. You can update
        "Main Goal" if you discover the overall game objective.
        
        Be as specific as possible in the Action Log, indicating what input was tried,
        the outcome, and your confidence level. Remember that certain actions might
        currently be blocked by the game environment.
        
        Use all this information to understand the game rules and beat the game in
        as few moves as possible.        
        ---
    """).strip()
    
    FIND_ACTION_INSTRUCT = dedent("""\
        Given the provided image and the desired action above, decide what to do
        based on the following information:
                                  
        "Move Up" = ACTION1
        "Move Down" = ACTION2
        "Move Left" = ACTION3
        "Move Right" = ACTION4
        "Perform Action" = ACTION5
        "Click on Object" = ACTION6 (you need to provide x, y coordinates in exact pixels)
        "Undo" = ACTION7
        
        Return JSON in this format:
        {
            "action": "ACTION1",
            "x": 0,
            "y": 0
        }
        
        Only include x and y if the action is ACTION6. Otherwise omit them.
        Respond with the JSON, nothing else.
    """).strip()
    
    def __init__(
        self,
        config: str,
        game_client: GameClient,
        card_id: str,
        max_actions: int = 40,
        retry_attempts: int = 3,
    ):
        """
        Initialize the multimodal agent.
        
        Args:
            config: Model configuration name from models.yml
            game_client: GameClient for API communication
            card_id: Scorecard identifier
            max_actions: Maximum actions to take before stopping
            retry_attempts: Number of retry attempts for failed API calls
        """
        self.config = config
        self.game_client = game_client
        self.card_id = card_id
        self.max_actions = max_actions
        self.retry_attempts = retry_attempts
        
        # Initialize provider adapter
        self.provider = create_provider(config)
        
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
        self.memory = GameMemory()
        self._previous_action: Optional[Dict[str, Any]] = None
        self._previous_images: List[Image.Image] = []
        self._previous_score = 0
        
    def _initialize_memory(self, available_actions: List[str]):
        """Initialize the agent's memory with game info"""
        self.memory.initialize(available_actions)
    
    def _extract_usage(self, response: Any) -> tuple[int, int]:
        """Extract token usage from provider response"""
        # OpenAI format
        if hasattr(response, 'usage'):
            if hasattr(response.usage, 'prompt_tokens'):
                return response.usage.prompt_tokens, response.usage.completion_tokens
            # Anthropic format
            elif hasattr(response.usage, 'input_tokens'):
                return response.usage.input_tokens, response.usage.output_tokens
        return 0, 0
    
    def _extract_content(self, response: Any) -> str:
        """Extract text content from provider response"""
        # OpenAI format
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        # Anthropic format
        elif hasattr(response, 'content') and response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
        return ""
    
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
    
    @retry_with_exponential_backoff(max_retries=3)
    def _call_provider(self, messages: List[Dict[str, Any]]) -> Any:
        """Call provider with retry logic"""
        # Use the provider's client directly for multimodal support
        # Most providers follow OpenAI-style API
        provider_name = self.provider.model_config.provider
        
        if provider_name == "openai":
            return self.provider.client.chat.completions.create(
                model=self.provider.model_config.model_name,
                messages=messages,
                **self.provider.model_config.kwargs
            )
        elif provider_name == "anthropic":
            # Anthropic has different message format - extract text and images
            return self.provider.client.messages.create(
                model=self.provider.model_config.model_name,
                messages=messages,
                **self.provider.model_config.kwargs
            )
        else:
            # For other providers, try OpenAI-compatible format
            try:
                return self.provider.client.chat.completions.create(
                    model=self.provider.model_config.model_name,
                    messages=messages,
                    **self.provider.model_config.kwargs
                )
            except AttributeError:
                # Fallback to direct client call
                return self.provider.client.create(
                    model=self.provider.model_config.model_name,
                    messages=messages,
                    **self.provider.model_config.kwargs
                )
    
    def _analyze_previous_action(
        self,
        current_frame_images: List[Image.Image],
        current_score: int
    ) -> str:
        """Analyze the results of the previous action"""
        if not self._previous_action:
            return "no previous action"
        
        level_complete = ""
        if current_score > self._previous_score:
            level_complete = "NEW LEVEL!!!! - Whatever you did must have been good!"
        
        memory_prompt = self.memory.to_prompt_text(HUMAN_ACTIONS)
        analyze_prompt = f"{level_complete}\n\n{self.ANALYZE_INSTRUCT}\n\n{memory_prompt}"
        
        # Build images: previous last frame, current frames, diff
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
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "text", "text": str(self._previous_action)}],
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
        
        response = self._call_provider(messages)
        
        # Track costs - handle different response formats
        prompt_tokens, completion_tokens = self._extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens)
        
        # Extract analysis and update memory
        analysis_message = self._extract_content(response)
        logger.info(f"Analysis: {analysis_message[:200]}...")
        
        # Update memory from LLM analysis response
        analysis = self.memory.update_from_analysis(analysis_message)
        
        return analysis
    
    def _choose_human_action(
        self,
        frame_images: List[Image.Image],
        analysis: str
    ) -> Dict[str, Any]:
        """Choose the next human-level action"""
        memory_prompt = self.memory.to_prompt_text(HUMAN_ACTIONS)
        if len(analysis) > 20:
            prompt = f"{analysis}\n\n{memory_prompt}\n\n{self.ACTION_INSTRUCT}"
        else:
            prompt = f"{memory_prompt}\n\n{self.ACTION_INSTRUCT}"
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    *[make_image_block(image_to_base64(img)) for img in frame_images],
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        response = self._call_provider(messages)
        
        # Track costs
        prompt_tokens, completion_tokens = self._extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens)
        
        action_message = self._extract_content(response)
        logger.info(f"Human action: {action_message[:200]}...")
        
        return extract_json_from_response(action_message)
    
    def _convert_to_game_action(
        self,
        human_action: str,
        last_frame_image: Image.Image
    ) -> Dict[str, Any]:
        """Convert human action description to game action"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    make_image_block(image_to_base64(last_frame_image)),
                    {
                        "type": "text",
                        "text": human_action + "\n\n" + self.FIND_ACTION_INSTRUCT,
                    },
                ],
            },
        ]
        
        response = self._call_provider(messages)
        
        # Track costs
        prompt_tokens, completion_tokens = self._extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens)
        
        action_message = self._extract_content(response)
        logger.info(f"Game action: {action_message[:200]}...")
        
        return extract_json_from_response(action_message)
    
    def _execute_game_action(
        self,
        action_name: str,
        action_data: Optional[Dict[str, Any]],
        game_id: str,
        guid: Optional[str]
    ) -> Dict[str, Any]:
        """Execute action via game client"""
        data = {"game_id": game_id}
        if guid:
            data["guid"] = guid
        if action_data:
            data.update(action_data)
        
        return self.game_client.execute_action(action_name, data)
    
    def play_game(self, game_id: str) -> GameResult:
        """
        Play a complete game and return results.
        
        Args:
            game_id: Game identifier to play
            
        Returns:
            GameResult with complete game information
        """
        logger.info(f"Starting game {game_id} with config {self.config}")
        start_time = time.time()
        
        # Reset game
        state = self.game_client.reset_game(self.card_id, game_id)
        guid = state.get("guid")
        current_score = state.get("score", 0)
        current_state = state.get("state", "IN_PROGRESS")
        
        # Initialize memory with available actions
        available_actions = state.get("available_actions", list(HUMAN_ACTIONS.keys()))
        self._initialize_memory(available_actions)
        
        # Main game loop
        while (
            current_state not in ["WIN", "GAME_OVER"]
            and self.action_counter < self.max_actions
        ):
            try:
                # Convert frames to images
                frames = state.get("frame", [])
                if not frames:
                    logger.warning("No frames in state, breaking")
                    break
                
                frame_images = [grid_to_image(frame) for frame in frames]
                
                # Analyze previous action if exists
                analysis = self._analyze_previous_action(frame_images, current_score)
                
                # Choose next action
                human_action_dict = self._choose_human_action(frame_images, analysis)
                human_action = human_action_dict.get("human_action")
                
                if not human_action:
                    logger.error("No human_action in response")
                    break
                
                # Convert to game action
                game_action_dict = self._convert_to_game_action(human_action, frame_images[-1])
                action_name = game_action_dict.get("action")
                
                if not action_name:
                    logger.error("No action name in response")
                    break
                
                # Prepare action data
                action_data_dict = {}
                if action_name == "ACTION6":
                    # Scale coordinates from 128x128 back to 64x64
                    x = game_action_dict.get("x", 0)
                    y = game_action_dict.get("y", 0)
                    action_data_dict = {
                        "x": max(0, min(x, 127)) // 2,
                        "y": max(0, min(y, 127)) // 2,
                    }
                
                # Execute action
                state = self._execute_game_action(action_name, action_data_dict, game_id, guid)
                guid = state.get("guid", guid)
                new_score = state.get("score", current_score)
                current_state = state.get("state", "IN_PROGRESS")
                
                # Record action
                action_record = GameActionRecord(
                    action_num=self.action_counter + 1,
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
                self.action_history.append(action_record)
                
                # Update memory with action log entry
                self.memory.add_action_log_entry(
                    action_num=self.action_counter + 1,
                    action_description=human_action,
                    action_type=action_name,
                    expected_result=human_action_dict.get("expected_result"),
                    actual_result=analysis[:200] if analysis else None,
                    outcome="level_complete" if new_score > current_score else "ongoing",
                    observations=analysis[:200] if analysis else None
                )
                
                # Update tracking
                self._previous_action = human_action_dict
                self._previous_images = frame_images
                self._previous_score = current_score
                current_score = new_score
                self.action_counter += 1
                
                logger.info(
                    f"Action {self.action_counter}: {action_name}, "
                    f"Score: {current_score}, State: {current_state}"
                )
                
            except Exception as e:
                logger.error(f"Error during game loop: {e}", exc_info=True)
                break
        
        # Create result
        duration = time.time() - start_time
        scorecard_url = f"{self.game_client.ROOT_URL}/scorecards/{self.card_id}"
        
        result = GameResult(
            game_id=game_id,
            config=self.config,
            final_score=current_score,
            final_state=current_state,
            actions_taken=self.action_counter,
            duration_seconds=duration,
            total_cost=self.total_cost,
            usage=self.total_usage,
            actions=self.action_history,
            timestamp=datetime.now(timezone.utc),
            scorecard_url=scorecard_url
        )
        
        logger.info(
            f"Game completed: {current_state}, Score: {current_score}, "
            f"Actions: {self.action_counter}, Cost: ${self.total_cost.total_cost:.4f}"
        )
        
        return result

