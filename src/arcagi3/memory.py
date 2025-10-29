"""
Memory management system for the ARC-AGI-3 agent.

Provides structured memory for tracking game state, rules, goals, and action history.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from textwrap import dedent


@dataclass
class ActionLogEntry:
    """Entry in the action log tracking a single action attempt"""
    action_num: int
    action_description: str  # Human-readable action description
    action_type: str  # ACTION1, ACTION2, etc.
    expected_result: Optional[str] = None
    actual_result: Optional[str] = None
    outcome: Optional[str] = None  # Success, failure, partial, etc.
    confidence: Optional[str] = None  # High, medium, low
    observations: Optional[str] = None  # What was observed
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GameMemory:
    """
    Structured memory for tracking game knowledge during play.
    
    Stores:
    - Available actions (immutable, from game)
    - Main goal (overall game objective)
    - Current goal (short-term/tactical goal)
    - Discovered game rules
    - Action history/log
    - Learned strategies and patterns
    """
    # Immutable game information
    available_actions: List[str] = field(default_factory=list)
    
    # Mutable game knowledge
    main_goal: str = "Game goal unknown so far"
    current_goal: str = ""
    game_rules: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)  # Learned strategies/patterns
    observations: List[str] = field(default_factory=list)  # Key observations about the game
    
    # Action history
    action_log: List[ActionLogEntry] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def initialize(self, available_actions: List[str], initial_main_goal: Optional[str] = None, initial_goal: Optional[str] = None):
        """
        Initialize memory with game information.
        
        Args:
            available_actions: List of available action types
            initial_main_goal: Optional main goal description (overall game objective)
            initial_goal: Optional current/tactical goal description
        """
        self.available_actions = available_actions.copy()
        
        # Main goal (overall game objective)
        if initial_main_goal:
            self.main_goal = initial_main_goal
        else:
            self.main_goal = "Game goal unknown so far"
        
        # Current goal (short-term/tactical)
        if initial_goal:
            self.current_goal = initial_goal
        else:
            self.current_goal = "Use the known human game inputs to interact with the game environment and learn the rules."
        
        self.game_rules = ["Nothing is known currently other than this is a turn-based game that I need to solve."]
        self.action_log = []
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
    
    def add_action_log_entry(
        self,
        action_num: int,
        action_description: str,
        action_type: str,
        expected_result: Optional[str] = None,
        actual_result: Optional[str] = None,
        outcome: Optional[str] = None,
        confidence: Optional[str] = None,
        observations: Optional[str] = None
    ):
        """Add an entry to the action log"""
        entry = ActionLogEntry(
            action_num=action_num,
            action_description=action_description,
            action_type=action_type,
            expected_result=expected_result,
            actual_result=actual_result,
            outcome=outcome,
            confidence=confidence,
            observations=observations
        )
        self.action_log.append(entry)
        self.last_updated = datetime.utcnow()
    
    def update_main_goal(self, new_main_goal: str):
        """Update the main goal (overall game objective)"""
        self.main_goal = new_main_goal
        self.last_updated = datetime.utcnow()
    
    def update_goal(self, new_goal: str):
        """Update the current goal (short-term/tactical)"""
        self.current_goal = new_goal
        self.last_updated = datetime.utcnow()
    
    def add_game_rule(self, rule: str):
        """Add a discovered game rule"""
        if rule and rule not in self.game_rules:
            self.game_rules.append(rule)
            self.last_updated = datetime.utcnow()
    
    def update_game_rules(self, rules: List[str]):
        """Replace all game rules with new list"""
        self.game_rules = rules.copy() if rules else []
        self.last_updated = datetime.utcnow()
    
    def add_strategy(self, strategy: str):
        """Add a learned strategy or pattern"""
        if strategy and strategy not in self.strategies:
            self.strategies.append(strategy)
            self.last_updated = datetime.utcnow()
    
    def add_observation(self, observation: str):
        """Add a key observation about the game"""
        if observation and observation not in self.observations:
            self.observations.append(observation)
            self.last_updated = datetime.utcnow()
    
    def to_prompt_text(self, human_actions_map: Optional[Dict[str, str]] = None) -> str:
        """
        Convert memory to formatted text for inclusion in prompts.
        
        Args:
            human_actions_map: Optional mapping of action codes to human-readable descriptions.
                             If not provided, will show action codes directly.
        
        Returns:
            Formatted string representation of memory
        """
        # Format available actions
        if human_actions_map:
            action_descriptions = []
            for action in self.available_actions:
                if action in human_actions_map:
                    action_descriptions.append(f"- {human_actions_map[action]}")
                else:
                    action_descriptions.append(f"- {action}")
            human_inputs = "\n".join(action_descriptions)
        else:
            human_inputs = "\n".join([f"- {action}" for action in self.available_actions])
        
        # Format game rules
        rules_text = "\n".join([f"- {rule}" for rule in self.game_rules]) if self.game_rules else "Nothing is known currently."
        
        # Format action log
        if self.action_log:
            log_entries = []
            for entry in self.action_log[-10:]:  # Last 10 actions
                log_entry = f"Action {entry.action_num}: {entry.action_description}"
                if entry.outcome:
                    log_entry += f" - {entry.outcome}"
                if entry.observations:
                    log_entry += f" (Observed: {entry.observations})"
                log_entries.append(log_entry)
            log_text = "\n".join(log_entries)
        else:
            log_text = "No actions taken so far."
        
        # Format strategies (if any)
        strategies_text = ""
        if self.strategies:
            strategies_text = "\n\n## Learned Strategies\n" + "\n".join([f"- {s}" for s in self.strategies[-5:]])
        
        # Format observations (if any)
        observations_text = ""
        if self.observations:
            observations_text = "\n\n## Key Observations\n" + "\n".join([f"- {o}" for o in self.observations[-5:]])
        
        prompt = dedent(f"""\
            ## Known Human Game Inputs
            {human_inputs}
                                    
            ## Main Goal
            {self.main_goal}
                                    
            ## Current Goal
            {self.current_goal}
                                    
            ## Game Rules
            {rules_text}
            
            ## Action Log
            {log_text}{strategies_text}{observations_text}
        """).strip()
        
        return prompt
    
    def get_recent_actions(self, n: int = 5) -> List[ActionLogEntry]:
        """Get the last N actions from the log"""
        return self.action_log[-n:] if self.action_log else []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state"""
        return {
            "total_actions": len(self.action_log),
            "game_rules_count": len(self.game_rules),
            "strategies_count": len(self.strategies),
            "observations_count": len(self.observations),
            "main_goal": self.main_goal,
            "current_goal": self.current_goal,
            "last_updated": self.last_updated.isoformat(),
        }
    
    def update_from_analysis(self, analysis_text: str) -> str:
        """
        Update memory from LLM analysis text.
        
        Expected format: The analysis text should contain sections after "---" separator.
        The memory prompt should be in the format that matches the to_prompt_text output.
        
        Args:
            analysis_text: The full analysis text from LLM, with memory update after "---"
        
        Returns:
            The analysis portion (before "---")
        """
        # Split on "---" separator
        parts = analysis_text.split("---", 1)
        analysis = parts[0].strip()
        
        if len(parts) > 1 and parts[1].strip():
            memory_update = parts[1].strip()
            self._parse_memory_update(memory_update)
        
        return analysis
    
    def _parse_memory_update(self, memory_text: str):
        """
        Parse memory update text and update the memory structure.
        
        This parses text in the format:
        ## Current Goal
        ...
        ## Game Rules
        ...
        ## Action Log
        ...
        """
        lines = memory_text.split("\n")
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("##"):
                # Process previous section
                if current_section and current_content:
                    self._apply_section_update(current_section, "\n".join(current_content))
                
                # Start new section
                current_section = line.replace("##", "").strip().lower()
                current_content = []
            elif current_section and line:
                current_content.append(line)
        
        # Process last section
        if current_section and current_content:
            self._apply_section_update(current_section, "\n".join(current_content))
    
    def _apply_section_update(self, section: str, content: str):
        """Apply update to a specific memory section"""
        if section == "main goal":
            self.update_main_goal(content.strip())
        elif section == "current goal":
            self.update_goal(content.strip())
        elif section == "game rules":
            # Parse rules (lines starting with "-")
            rules = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    rule = line[1:].strip()
                    if rule:
                        rules.append(rule)
                elif line and len(line) > 10:  # Non-bullet formatted rule
                    rules.append(line)
            
            if rules:
                self.update_game_rules(rules)
        elif section == "action log":
            # Action log is updated via add_action_log_entry, not parsed from text
            # This section is informational
            pass
        elif section == "learned strategies":
            # Parse strategies (lines starting with "-")
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    strategy = line[1:].strip()
                    if strategy:
                        self.add_strategy(strategy)
        elif section == "key observations":
            # Parse observations (lines starting with "-")
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    observation = line[1:].strip()
                    if observation:
                        self.add_observation(observation)

