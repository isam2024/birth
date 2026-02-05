"""Core autonomous agent implementation.

This is the heart of Birth - an autonomous artist entity that perceives,
reflects, decides, acts, and reacts in a continuous cycle of existence.
"""

import asyncio
import random
from datetime import datetime
from typing import TYPE_CHECKING, Any

from birth.agents.memory import MemoryStream
from birth.agents.sentiment import SentimentModel
from birth.config import get_config
from birth.core.events import Event, EventBus, EventType
from birth.integrations.ollama import OllamaClient
from birth.observation.logger import AgentLogger, DropsLogger
from birth.storage.models import Agent, MemoryType
from birth.storage.repository import Repository

if TYPE_CHECKING:
    from birth.world.canvas import Canvas

# Module-level drops logger for red highlighting
_drops_logger = DropsLogger()


class Intention:
    """Represents an agent's intention to act."""

    def __init__(
        self,
        action: str,
        reason: str,
        targets: dict[str, Any] | None = None,
        urgency: float = 0.5,
    ):
        self.action = action
        self.reason = reason
        self.targets = targets or {}
        self.urgency = urgency  # 0.0 = low, 1.0 = high

    def __str__(self) -> str:
        return f"Intention({self.action}: {self.reason})"


class AutonomousAgent:
    """An autonomous artist agent.

    The agent follows a continuous lifecycle:
    1. Perceive - Observe the environment
    2. Reflect - Process observations with memory and sentiment
    3. Decide - Choose an action based on intention
    4. Act - Execute the chosen action
    5. React - Update internal state based on result
    """

    def __init__(
        self,
        agent_data: Agent,
        repository: Repository,
        ollama: OllamaClient,
        event_bus: EventBus,
        canvas: "Canvas | None" = None,
    ):
        self._data = agent_data
        self._repository = repository
        self._ollama = ollama
        self._event_bus = event_bus
        self._canvas = canvas
        self._config = get_config().simulation

        # Initialize subsystems
        self._logger = AgentLogger(agent_data.id, agent_data.name)
        self._memory = MemoryStream(agent_data.id, repository)
        self._sentiment = SentimentModel(agent_data.id, repository, self._logger)

        # Runtime state
        self._is_alive = True
        self._cycle_count = 0
        self._last_action_time: datetime | None = None
        self._actions_since_reflection = 0
        self._cycles_since_creation = 0  # Track creative urge

        # Available actions
        self._available_actions = [
            "observe_commons",
            "observe_gallery",
            "create_art",
            "message_agent",
            "critique_art",
            "reflect",
            "rest",
        ]

    # ========== Properties ==========

    @property
    def id(self) -> str:
        return self._data.id

    @property
    def name(self) -> str:
        return self._data.name

    @property
    def backstory(self) -> str:
        return self._data.backstory

    @property
    def philosophy(self) -> str:
        return self._data.philosophy

    @property
    def mood(self) -> float:
        return self._data.mood

    @property
    def energy(self) -> float:
        return self._data.energy

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @property
    def memory(self) -> MemoryStream:
        return self._memory

    @property
    def sentiment(self) -> SentimentModel:
        return self._sentiment

    # ========== Lifecycle ==========

    async def initialize(self) -> None:
        """Initialize agent subsystems."""
        await self._memory.load_recent()
        await self._sentiment.load()
        self._logger.action("initialized")

    async def live(self) -> None:
        """The continuous cycle of autonomous life.

        This is the main loop that drives the agent's existence.
        """
        await self.initialize()

        while self._is_alive:
            try:
                # 1. Perceive
                perceptions = await self._perceive()

                # 2. Reflect (internal monologue)
                intention = await self._reflect(perceptions)

                # 3. Decide
                action = await self._decide(intention)

                # 4. Act
                result = await self._act(action, intention)

                # 5. React (update sentiment and state)
                await self._react(result)

                self._cycle_count += 1
                self._actions_since_reflection += 1

                # Track creative urge
                if action == "create_art" and result.get("success"):
                    self._cycles_since_creation = 0
                else:
                    self._cycles_since_creation += 1

                # Periodic deep reflection
                if self._actions_since_reflection >= self._config.reflection_interval:
                    await self._deep_reflect()
                    self._actions_since_reflection = 0

                # Variable rest period based on energy and mood
                rest_time = self._get_rest_period()
                await asyncio.sleep(rest_time)

            except asyncio.CancelledError:
                self._is_alive = False
                break
            except Exception as e:
                self._logger.error(f"Error in lifecycle: {e}", error=str(e))
                await asyncio.sleep(5)  # Brief pause on error

    def stop(self) -> None:
        """Stop the agent's lifecycle."""
        self._is_alive = False

    # ========== Core Cycle Methods ==========

    async def _perceive(self) -> list[str]:
        """Observe the environment.

        Returns:
            List of perception strings
        """
        perceptions = []

        if self._canvas:
            # Cache for looking up agent names
            name_cache: dict[str, str] = {}

            async def get_agent_name(agent_id: str) -> str:
                if agent_id in name_cache:
                    return name_cache[agent_id]
                agent = await self._repository.get_agent(agent_id)
                name = agent.name if agent else agent_id[:8]
                name_cache[agent_id] = name
                return name

            # See recent artworks in the Commons
            recent_art = await self._canvas.commons.get_recent(limit=5)
            for art in recent_art:
                if art.creator_id != self.id:
                    creator_name = await get_agent_name(art.creator_id)
                    perception = f"New artwork in Commons: '{art.title}' by {creator_name}"
                    perceptions.append(perception)
                    await self._memory.perceive(perception, importance=0.5)

            # Check for messages/interactions
            interactions = await self._repository.get_interactions_for_agent(
                self.id, limit=5
            )
            for interaction in interactions:
                if interaction.to_agent_id == self.id:
                    sender_name = await get_agent_name(interaction.from_agent_id)
                    perception = f"Received {interaction.interaction_type} from {sender_name}: {interaction.content[:100]}"
                    perceptions.append(perception)
                    await self._memory.perceive(perception, importance=0.7)

            # Check for external image drops (limit to 2 per cycle to avoid overwhelm)
            try:
                drops_this_cycle = 0
                for drop in self._canvas.drops.recent_drops:
                    if drops_this_cycle >= 2:
                        break  # Process remaining drops in next cycle
                    if self.id not in drop.viewed_by:
                        # Mark as viewed
                        drop.viewed_by.add(self.id)
                        drops_this_cycle += 1
                        _drops_logger.viewed(self.name, drop.filename)
                        # Create perception from drop
                        perception = self._canvas.drops.format_drop_for_perception(drop)
                        perceptions.append(perception)
                        await self._memory.perceive(
                            f"Saw external image '{drop.filename}': {drop.description[:200]}...",
                            importance=0.8,  # External input is important
                        )
                        # Form opinion on the drop
                        opinion, reason = await self._form_opinion_on_drop(drop)
                        self._sentiment.register_name(f"drop:{drop.id}", f"external image '{drop.filename}'")
                        await self._sentiment.update("concept", f"drop:{drop.id}", opinion)
                        _drops_logger.opinion(self.name, drop.filename, opinion, reason)
            except Exception:
                pass  # Drops may not be available

        if perceptions:
            self._logger.perception(f"Perceived {len(perceptions)} events")

        return perceptions

    async def _reflect(self, perceptions: list[str]) -> Intention:
        """Process perceptions and generate intention.

        This is the internal monologue - where neutral observations
        become biased experiences through the sentiment model.

        Args:
            perceptions: Recent perceptions

        Returns:
            An Intention to act
        """
        # Build context for reflection
        recent_memories = self._memory.summarize_recent(10)
        sentiment_summary = self._sentiment.summarize()

        # Limit perceptions to avoid overwhelming the agent
        limited_perceptions = perceptions[:5] if perceptions else []
        if len(perceptions) > 5:
            limited_perceptions.append(f"...and {len(perceptions) - 5} more observations.")

        # Build creative urge indicator
        creative_urge = ""
        if self._cycles_since_creation >= 8:
            creative_urge = "\nCREATIVE URGE: Strong - you haven't created anything in a while and feel the need to express yourself."
        elif self._cycles_since_creation >= 5:
            creative_urge = "\nCREATIVE URGE: Building - ideas are forming and you feel inspired to create."
        elif self._cycles_since_creation >= 3:
            creative_urge = "\nCREATIVE URGE: Stirring - you're absorbing inspiration and may want to create soon."

        prompt = f"""You are {self.name}.

BACKSTORY: {self.backstory}

ARTISTIC PHILOSOPHY: {self.philosophy}

CURRENT MOOD: {self._mood_description()}
ENERGY LEVEL: {self._energy_description()}{creative_urge}

RECENT MEMORIES:
{recent_memories}

CURRENT FEELINGS:
{sentiment_summary}

NEW PERCEPTIONS:
{chr(10).join(limited_perceptions) if limited_perceptions else "Nothing new."}

Based on all of this, what do you feel compelled to do next?

AVAILABLE ACTIONS:
- observe_commons (see what others have created)
- observe_gallery (seek inspiration from the Gallery)
- create_art (make something new)
- message_agent (reach out to another artist)
- critique_art (respond to someone's work)
- reflect (deep introspection)
- rest (do nothing, recover energy)

Respond in this format:
ACTION: [chosen action]
REASON: [brief explanation of why, in first person]
URGENCY: [low/medium/high]
TARGET: [if applicable, the target agent or artwork ID]"""

        response = await self._ollama.generate(
            prompt=prompt,
            temperature=0.8,
            max_tokens=200,
        )

        # Parse response
        intention = self._parse_intention(response)

        # Record the thought
        await self._memory.think(
            f"I decided to {intention.action} because {intention.reason}",
            importance=0.4,
        )

        return intention

    async def _decide(self, intention: Intention) -> str:
        """Validate and finalize action choice.

        Args:
            intention: The generated intention

        Returns:
            The action to take
        """
        # Validate action is available
        if intention.action not in self._available_actions:
            # Fall back to rest if invalid
            return "rest"

        # Energy check - force rest if exhausted
        if self.energy < 0.1 and intention.action != "rest":
            await self._memory.think(
                "I am too exhausted to do anything but rest.",
                importance=0.5,
            )
            return "rest"

        return intention.action

    async def _act(self, action: str, intention: Intention) -> dict[str, Any]:
        """Execute the chosen action.

        Args:
            action: The action to take
            intention: The intention behind it

        Returns:
            Result dictionary
        """
        self._logger.action(action, reason=intention.reason)
        self._last_action_time = datetime.utcnow()

        result: dict[str, Any] = {
            "action": action,
            "success": True,
            "details": {},
        }

        if action == "create_art":
            result = await self._action_create_art(intention)
        elif action == "observe_commons":
            result = await self._action_observe_commons()
        elif action == "observe_gallery":
            result = await self._action_observe_gallery()
        elif action == "message_agent":
            result = await self._action_message_agent(intention)
        elif action == "critique_art":
            result = await self._action_critique_art(intention)
        elif action == "reflect":
            await self._deep_reflect()
            result["details"] = {"type": "reflection"}
        elif action == "rest":
            result = await self._action_rest()

        # Record the action
        await self._memory.act(
            f"I {action}: {result.get('details', {})}",
            importance=0.6 if result["success"] else 0.4,
        )

        return result

    async def _react(self, result: dict[str, Any]) -> None:
        """Update internal state based on action result.

        Args:
            result: The action result
        """
        action = result["action"]
        success = result["success"]

        # Update mood based on outcome
        if success:
            mood_delta = 0.05 if action in ["create_art", "message_agent"] else 0.02
            self._data.mood = min(1.0, self._data.mood + mood_delta)
        else:
            self._data.mood = max(0.0, self._data.mood - 0.05)

        # Update energy
        energy_costs = {
            "create_art": 0.15,
            "message_agent": 0.05,
            "critique_art": 0.08,
            "observe_commons": 0.02,
            "observe_gallery": 0.02,
            "reflect": 0.05,
            "rest": -0.2,  # Restores energy
        }
        energy_change = energy_costs.get(action, 0.03)
        self._data.energy = max(0.0, min(1.0, self._data.energy - energy_change))

        # Apply sentiment mood modifier
        self._data.mood = max(0.0, min(1.0,
            self._data.mood + self._sentiment.overall_mood_modifier
        ))

        # Periodic sentiment decay
        if self._cycle_count % 10 == 0:
            await self._sentiment.decay_all()

    # ========== Action Implementations ==========

    async def _action_create_art(self, intention: Intention) -> dict[str, Any]:
        """Create a new artwork."""
        if not self._canvas:
            return {"action": "create_art", "success": False, "details": {"error": "No canvas"}}

        # Get inspiration from recent observations
        recent = self._memory.get_by_type(MemoryType.PERCEPTION, limit=5)
        inspiration_context = "\n".join(m.content for m in recent) if recent else "Nothing specific."

        # Check for active challenge
        challenge_prompt = None
        challenge = self._canvas.challenges.get_challenge_for_agent(self.id)
        if challenge:
            challenge_prompt = challenge.prompt
            self._canvas.challenges.mark_agent_responded(self.id)
            self._logger.action("responding_to_challenge", challenge_id=challenge.id)

        # Generate artwork
        artwork = await self._canvas.commons.create_artwork(
            creator=self._data,
            ollama=self._ollama,
            inspiration_context=inspiration_context,
            sentiment_summary=self._sentiment.summarize(),
            challenge_prompt=challenge_prompt,
        )

        if artwork:
            self._logger.creation(artwork.id, artwork.title, artwork.medium)

            # Publish event
            await self._event_bus.publish(Event(
                type=EventType.ARTWORK_CREATED,
                source_agent_id=self.id,
                data={"artwork_id": artwork.id, "title": artwork.title},
            ))

            # Positive sentiment toward own creation
            await self._sentiment.update("artwork", artwork.id, 0.8)

            return {
                "action": "create_art",
                "success": True,
                "details": {"artwork_id": artwork.id, "title": artwork.title},
            }

        return {"action": "create_art", "success": False, "details": {"error": "Creation failed"}}

    async def _action_observe_commons(self) -> dict[str, Any]:
        """Observe artworks in the Commons."""
        if not self._canvas:
            return {"action": "observe_commons", "success": False, "details": {}}

        artworks = await self._canvas.commons.get_recent(limit=10)
        observed = []

        for art in artworks:
            if art.creator_id != self.id:
                # Register names for meaningful sentiment summaries
                self._sentiment.register_name(art.id, art.title)

                # Look up creator name
                creator = await self._repository.get_agent(art.creator_id)
                if creator:
                    self._sentiment.register_name(art.creator_id, creator.name)

                # Form opinion based on philosophy alignment
                opinion = await self._form_opinion_on_artwork(art)
                await self._sentiment.update("artwork", art.id, opinion)
                await self._sentiment.update("agent", art.creator_id, opinion * 0.5)
                observed.append(art.id)

        return {
            "action": "observe_commons",
            "success": True,
            "details": {"observed_count": len(observed)},
        }

    async def _action_observe_gallery(self) -> dict[str, Any]:
        """Seek inspiration from the Gallery."""
        if not self._canvas:
            return {"action": "observe_gallery", "success": False, "details": {}}

        inspiration = await self._canvas.gallery.get_random_inspiration()
        if inspiration:
            await self._memory.perceive(
                f"Found inspiration in Gallery: {inspiration[:200]}",
                importance=0.6,
            )
            return {
                "action": "observe_gallery",
                "success": True,
                "details": {"found_inspiration": True},
            }

        return {"action": "observe_gallery", "success": True, "details": {"found_inspiration": False}}

    async def _action_message_agent(self, intention: Intention) -> dict[str, Any]:
        """Send a message to another agent."""
        target_id = intention.targets.get("agent_id")
        if not target_id:
            # Pick someone to message
            friends = self._sentiment.get_friends()
            if friends:
                target_id = random.choice(friends)
            else:
                return {"action": "message_agent", "success": False, "details": {"error": "No target"}}

        # Generate message content
        message = await self._generate_message(target_id)

        # Record interaction
        from birth.storage.models import Interaction
        interaction = Interaction(
            from_agent_id=self.id,
            to_agent_id=target_id,
            interaction_type="message",
            content=message,
        )
        await self._repository.add_interaction(interaction)

        self._logger.interaction("message", target_id, preview=message[:50])

        await self._event_bus.publish(Event(
            type=EventType.MESSAGE_SENT,
            source_agent_id=self.id,
            data={"to_agent_id": target_id, "preview": message[:100]},
        ))

        return {
            "action": "message_agent",
            "success": True,
            "details": {"target": target_id, "message": message[:100]},
        }

    async def _action_critique_art(self, intention: Intention) -> dict[str, Any]:
        """Write a critique of an artwork."""
        artwork_id = intention.targets.get("artwork_id")
        if not artwork_id or not self._canvas:
            return {"action": "critique_art", "success": False, "details": {}}

        artwork = await self._repository.get_artwork(artwork_id)
        if not artwork:
            return {"action": "critique_art", "success": False, "details": {"error": "Not found"}}

        # Generate critique
        critique = await self._generate_critique(artwork)

        # Record interaction
        from birth.storage.models import Interaction
        interaction = Interaction(
            from_agent_id=self.id,
            to_agent_id=artwork.creator_id,
            interaction_type="critique",
            content=critique,
        )
        await self._repository.add_interaction(interaction)

        await self._event_bus.publish(Event(
            type=EventType.CRITIQUE_POSTED,
            source_agent_id=self.id,
            data={"artwork_id": artwork_id, "creator_id": artwork.creator_id},
        ))

        return {
            "action": "critique_art",
            "success": True,
            "details": {"artwork_id": artwork_id, "critique": critique[:100]},
        }

    async def _action_rest(self) -> dict[str, Any]:
        """Rest and recover energy."""
        # Just rest - energy recovery happens in _react
        await self._memory.think("I am resting, letting my mind wander.", importance=0.2)
        return {"action": "rest", "success": True, "details": {"rested": True}}

    # ========== Reflection ==========

    async def _deep_reflect(self) -> None:
        """Perform deep reflection on recent experiences.

        Synthesizes memories into high-level insights.
        """
        recent = self._memory.get_recent(20)
        if not recent:
            return

        memories_text = "\n".join(f"- {m.content}" for m in recent)

        prompt = f"""You are {self.name}, reflecting on your recent experiences.

RECENT EXPERIENCES:
{memories_text}

Based on these experiences, what is one insight or pattern you notice about yourself?
Write a single sentence of self-reflection in first person.

Insight:"""

        insight = await self._ollama.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
        )

        insight = insight.strip()
        if insight:
            # Record reflection
            await self._memory.reflect(insight, importance=0.8)

            # Save as formal reflection
            from birth.storage.models import Reflection
            reflection = Reflection(
                agent_id=self.id,
                insight=insight,
                derived_from=self._memory.get_memory_ids(20),
            )
            await self._repository.add_reflection(reflection)

            self._logger.reflection(insight, len(recent))

            await self._event_bus.publish(Event(
                type=EventType.REFLECTION_GENERATED,
                source_agent_id=self.id,
                data={"insight": insight},
            ))

    # ========== Helper Methods ==========

    def _parse_intention(self, response: str) -> Intention:
        """Parse LLM response into Intention."""
        lines = response.strip().split("\n")

        action = "rest"
        reason = "I need to rest."
        urgency = 0.5
        targets: dict[str, str] = {}

        for line in lines:
            line = line.strip()
            if line.startswith("ACTION:"):
                action = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
            elif line.startswith("URGENCY:"):
                urgency_str = line.split(":", 1)[1].strip().lower()
                urgency = {"low": 0.3, "medium": 0.5, "high": 0.8}.get(urgency_str, 0.5)
            elif line.startswith("TARGET:"):
                target = line.split(":", 1)[1].strip()
                if target and target.lower() != "none":
                    targets["target_id"] = target

        return Intention(action, reason, targets, urgency)

    def _get_rest_period(self) -> float:
        """Calculate rest period based on energy and mood."""
        base = self._config.min_rest_period
        max_rest = self._config.max_rest_period

        # Lower energy = longer rest
        energy_factor = 1.0 + (1.0 - self.energy) * 2.0

        # Lower mood = slightly longer rest
        mood_factor = 1.0 + (0.5 - self.mood) * 0.5

        rest_time = base * energy_factor * mood_factor

        # Add some randomness
        rest_time *= random.uniform(0.8, 1.2)

        return min(max_rest, rest_time)

    def _mood_description(self) -> str:
        """Get text description of current mood."""
        if self.mood >= 0.8:
            return "elated, full of creative energy"
        elif self.mood >= 0.6:
            return "content, quietly inspired"
        elif self.mood >= 0.4:
            return "neutral, observant"
        elif self.mood >= 0.2:
            return "melancholic, introspective"
        else:
            return "despondent, struggling"

    def _energy_description(self) -> str:
        """Get text description of current energy."""
        if self.energy >= 0.8:
            return "fully rested, ready to create"
        elif self.energy >= 0.5:
            return "adequate energy for activity"
        elif self.energy >= 0.2:
            return "tired, should rest soon"
        else:
            return "exhausted, must rest"

    async def _form_opinion_on_artwork(self, artwork) -> float:
        """Form an opinion on an artwork based on philosophy alignment.

        Returns:
            Opinion score (-1.0 to +1.0)
        """
        prompt = f"""You are {self.name}.

Your artistic philosophy: {self.philosophy}

You see this artwork:
Title: {artwork.title}
{f'Content: {artwork.content_text[:300]}...' if artwork.content_text else ''}
Style tags: {', '.join(artwork.style_tags) if artwork.style_tags else 'none'}

Based on your philosophy, how do you feel about this work?
Respond with just a number from -1.0 (strongly dislike) to +1.0 (strongly admire).

Score:"""

        response = await self._ollama.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=10,
        )

        try:
            score = float(response.strip())
            return max(-1.0, min(1.0, score))
        except ValueError:
            return 0.0

    async def _form_opinion_on_drop(self, drop) -> tuple[float, str]:
        """Form an opinion on an external image drop.

        Args:
            drop: The Drop object with image description

        Returns:
            Tuple of (opinion score, reason)
        """
        prompt = f"""You are {self.name}.

Your artistic philosophy: {self.philosophy}

An external image has been shared with the colony. Here is its description:

{drop.description[:500]}

Based on your philosophy, how do you feel about this image? Give a score and brief reason.

Format your response EXACTLY like this:
SCORE: [number from -1.0 to 1.0]
REASON: [one sentence explaining why, from your artistic perspective]"""

        response = await self._ollama.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=100,
        )

        # Parse score and reason
        score = 0.0
        reason = "No strong feelings"

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    score = max(-1.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return score, reason

    async def _generate_message(self, target_id: str) -> str:
        """Generate a message to another agent."""
        sentiment = await self._sentiment.get("agent", target_id)

        tone = "warm and friendly" if sentiment > 0.2 else (
            "cool and formal" if sentiment < -0.2 else "neutral"
        )

        prompt = f"""You are {self.name}, an artist.

Write a brief message (1-2 sentences) to a fellow artist.
Your tone should be {tone}.
The message should be about art, creativity, or your recent work.

Message:"""

        message = await self._ollama.generate(
            prompt=prompt,
            temperature=0.8,
            max_tokens=100,
        )

        return message.strip()

    async def _generate_critique(self, artwork) -> str:
        """Generate a critique of an artwork."""
        prompt = f"""You are {self.name}.

Your artistic philosophy: {self.philosophy}

Write a brief critique (2-3 sentences) of this artwork:
Title: {artwork.title}
{f'Content: {artwork.content_text[:300]}...' if artwork.content_text else ''}

Be honest and specific. Your critique should reflect your artistic values.

Critique:"""

        critique = await self._ollama.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=150,
        )

        return critique.strip()
