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
from .schemas import (
    GameAction,
    GameState,
    GameResult,
    GameActionRecord,
    ActionData,
    Cost,
    Usage,
    CompletionTokensDetails,
    APIType,
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
        never be changed as these are provided by the game itself.
        
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
        num_plays: int = 1,
    ):
        """
        Initialize the multimodal agent.
        
        Args:
            config: Model configuration name from models.yml
            game_client: GameClient for API communication
            card_id: Scorecard identifier
            max_actions: Maximum actions to take before stopping
            retry_attempts: Number of retry attempts for failed API calls
            num_plays: Number of times to play the game (continues session with memory)
        """
        self.config = config
        self.game_client = game_client
        self.card_id = card_id
        self.max_actions = max_actions
        self.retry_attempts = retry_attempts
        self.num_plays = num_plays
        
        # Initialize provider adapter
        self.provider = create_provider(config)
        
        # Check if model supports multimodal capabilities (required for this agent)
        if not self.provider.model_config.multimodal:
            raise ValueError(
                f"Model '{self.provider.model_config.name}' does not support multimodal/vision capabilities. "
                f"The multimodal agent requires a model with image support. "
                f"Please use a model with 'multimodal: true' in the configuration (e.g., gpt-4o-mini-2024-07-18)."
            )
        
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
        self._memory_prompt = ""
        self._previous_action: Optional[Dict[str, Any]] = None
        self._previous_images: List[Image.Image] = []
        self._previous_score = 0
        
    def _initialize_memory(self, available_actions: List[str]):
        """Initialize the agent's memory with game info"""
        human_inputs = get_human_inputs_text(available_actions)
        self._memory_prompt = dedent(f"""\
            ## Known Human Game Inputs
            {human_inputs}
                                     
            ## Current Goal
            Use the known human game inputs to interact with the game environment and learn the rules.
                                     
            ## Game Rules
            Nothing is known currently other than this is a turn-based game that I need to solve.
                                     
            ## Action Log
            No actions taken so far.
        """).strip()
    
    def _extract_usage(self, response: Any) -> tuple[int, int]:
        """Extract token usage from provider response"""
        if 'Stream' in str(type(response)):
            return 0, 0
        
        if hasattr(response, 'usage'):
            if hasattr(response.usage, 'prompt_tokens'):
                return response.usage.prompt_tokens, response.usage.completion_tokens
            elif hasattr(response.usage, 'input_tokens'):
                output_tokens = getattr(response.usage, 'output_tokens', 0)
                return response.usage.input_tokens, output_tokens
        return 0, 0
    
    def _extract_content(self, response: Any) -> str:
        """Extract text content from provider response"""
        # Check if it's an OpenAI stream - consume it first
        if 'Stream' in str(type(response)):
            # Consume the stream and get the final response
            logger.debug("Consuming OpenAI stream...")
            full_content = []
            try:
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_content.append(delta.content)
                return ''.join(full_content)
            except Exception as e:
                logger.error(f"Error consuming stream: {e}")
                return ""
        
        # OpenAI Chat Completions format
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            if content is None:
                logger.warning("OpenAI returned None content")
                return ""
            return content
        # OpenAI Responses API format
        elif hasattr(response, 'output_text'):
            return response.output_text or ""
        elif hasattr(response, 'output') and response.output:
            # Fallback for Responses API - get first text block
            if hasattr(response.output[0], 'content') and response.output[0].content:
                if hasattr(response.output[0].content[0], 'text'):
                    return response.output[0].content[0].text or ""
        # Anthropic format
        elif hasattr(response, 'content') and response.content:
            text_parts = []
            for block in response.content:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            return ''.join(text_parts)
        
        logger.warning(f"Unknown response format. Type: {type(response)}")
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
    
    def _convert_image_blocks_for_anthropic(self, content: Any) -> Any:
        """Convert OpenAI-style image_url blocks to Anthropic format"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            converted = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image_url":
                        # Convert from OpenAI format to Anthropic format
                        image_url = block.get("image_url", {})
                        url = image_url.get("url", "")
                        
                        # Extract base64 data from data URL
                        if url.startswith("data:image/png;base64,"):
                            base64_data = url[len("data:image/png;base64,"):]
                            converted.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_data
                                }
                            })
                        else:
                            # Keep as is if not base64
                            converted.append(block)
                    else:
                        converted.append(block)
                else:
                    converted.append(block)
            return converted
        return content
    
    def _convert_image_blocks_for_responses_api(self, content: Any) -> Any:
        """Convert OpenAI Chat Completions blocks to Responses API format"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            converted = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "image_url":
                        # For Responses API, image_url should be a string URL directly
                        image_url = block.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        else:
                            url = image_url

                        if url.startswith("data:image/"):
                            converted.append({
                                "type": "input_image",
                                "image_url": url
                            })
                        else:
                            converted.append(block)
                    elif block_type == "text":
                        converted.append({
                            "type": "input_text",
                            "text": block.get("text", "")
                        })
                    else:
                        converted.append(block)
                else:
                    converted.append(block)
            return converted
        return content
    
    @retry_with_exponential_backoff(max_retries=3)
    def _call_provider(self, messages: List[Dict[str, Any]]) -> Any:
        """Call provider with retry logic"""
        provider_name = self.provider.model_config.provider
        
        if provider_name == "openai":
            has_images = any(
                isinstance(msg.get("content"), list) and 
                any(isinstance(block, dict) and block.get("type") == "image_url" 
                    for block in msg.get("content", []))
                for msg in messages
            )
            
            if hasattr(self.provider.model_config, 'api_type') and self.provider.model_config.api_type == APIType.RESPONSES:
                responses_kwargs = dict(self.provider.model_config.kwargs)
                if "max_tokens" in responses_kwargs:
                    responses_kwargs["max_output_tokens"] = responses_kwargs.pop("max_tokens")
                if "max_completion_tokens" in responses_kwargs:
                    responses_kwargs["max_output_tokens"] = responses_kwargs.pop("max_completion_tokens")

                # Convert messages to Responses API format
                converted_messages = []
                for msg in messages:
                    converted_msg = dict(msg)
                    if isinstance(msg.get("content"), list):
                        converted_msg["content"] = self._convert_image_blocks_for_responses_api(msg["content"])
                    # For string content (like system messages), keep as-is
                    converted_messages.append(converted_msg)

                return self.provider.client.responses.create(
                    model=self.provider.model_config.model_name,
                    input=converted_messages,
                    **responses_kwargs
                )
            else:
                return self.provider.client.chat.completions.create(
                    model=self.provider.model_config.model_name,
                    messages=messages,
                    **self.provider.model_config.kwargs
                )
        elif provider_name == "anthropic":
            system_messages = []
            filtered_messages = []
            
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        system_messages.append(content)
                    elif isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            system_messages.append("\n".join(text_parts))
                else:
                    msg_copy = dict(msg)
                    msg_copy["content"] = self._convert_image_blocks_for_anthropic(msg.get("content"))
                    filtered_messages.append(msg_copy)
            
            system_content = "\n".join(system_messages) if system_messages else None
            
            anthropic_kwargs = dict(self.provider.model_config.kwargs)
            if system_content:
                anthropic_kwargs["system"] = system_content
            
            return self.provider.client.messages.create(
                model=self.provider.model_config.model_name,
                messages=filtered_messages,
                **anthropic_kwargs
            )
        else:
            try:
                return self.provider.client.chat.completions.create(
                    model=self.provider.model_config.model_name,
                    messages=messages,
                    **self.provider.model_config.kwargs
                )
            except AttributeError:
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
        
        analyze_prompt = f"{level_complete}\n\n{self.ANALYZE_INSTRUCT}\n\n{self._memory_prompt}"
        
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
        
        before, _, after = analysis_message.partition("---")
        analysis = before.strip()
        if after.strip():
            self._memory_prompt = after.strip()
        
        return analysis
    
    def _choose_human_action(
        self,
        frame_images: List[Image.Image],
        analysis: str
    ) -> Dict[str, Any]:
        """Choose the next human-level action"""
        if len(analysis) > 20:
            prompt = f"{analysis}\n\n{self._memory_prompt}\n\n{self.ACTION_INSTRUCT}"
        else:
            prompt = f"{self._memory_prompt}\n\n{self.ACTION_INSTRUCT}"
        
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
    
    def play_game(self, game_id: str) -> GameResult:
        """
        Play a complete game and return results.
        
        Args:
            game_id: Game identifier to play
            
        Returns:
            GameResult with complete game information (best result if multiple plays)
        """
        logger.info(f"Starting game {game_id} with config {self.config} ({self.num_plays} play(s))")
        overall_start_time = time.time()
        
        best_result: Optional[GameResult] = None
        guid: Optional[str] = None
        
        for play_num in range(1, self.num_plays + 1):
            play_start_time = time.time()

            if play_num > 1:
                logger.info(f"Starting play {play_num}/{self.num_plays} (continuing session with memory)")

            # Reset game (use guid to continue session if not first play)
            state = self.game_client.reset_game(self.card_id, game_id, guid=guid)
            guid = state.get("guid")
            current_score = state.get("score", 0)
            current_state = state.get("state", "IN_PROGRESS")

            # Initialize memory only on first play, otherwise keep existing memory
            if play_num == 1:
                available_actions = state.get("available_actions", list(HUMAN_ACTIONS.keys()))
                self._initialize_memory(available_actions)
            else:
                logger.info(f"Continuing with memory from previous play(s)")
            
            # Reset play-specific counters (but keep cumulative cost/usage)
            play_action_counter = 0
            play_action_history: List[GameActionRecord] = []
            
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

                    frame_images = [grid_to_image(frame) for frame in frames]
                    analysis = self._analyze_previous_action(frame_images, current_score)
                    
                    human_action_dict = self._choose_human_action(frame_images, analysis)
                    human_action = human_action_dict.get("human_action")
                    
                    if not human_action:
                        logger.error("No human_action in response")
                        break
                    
                    game_action_dict = self._convert_to_game_action(human_action, frame_images[-1])
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
                    
                    reasoning_for_api = human_action_dict.get("reasoning", "")
                    state = self._execute_game_action(action_name, action_data_dict, game_id, guid, reasoning_for_api)
                    guid = state.get("guid", guid)
                    new_score = state.get("score", current_score)
                    current_state = state.get("state", "IN_PROGRESS")
                    
                    action_record = GameActionRecord(
                        action_num=self.action_counter + play_action_counter + 1,
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
                    self._previous_score = current_score
                    current_score = new_score
                    play_action_counter += 1
                    self.action_counter += 1
                    
                    logger.info(
                        f"Play {play_num}, Action {play_action_counter}: {action_name}, "
                        f"Score: {current_score}, State: {current_state}"
                    )
                    
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
                final_memory=self._memory_prompt,
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
        
        return best_result

