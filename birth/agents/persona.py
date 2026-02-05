"""Persona generation and loading for agents."""

import json
import random
import uuid
from pathlib import Path

from birth.config import get_config
from birth.integrations.ollama import OllamaClient
from birth.observation.logger import get_logger
from birth.storage.models import Agent

logger = get_logger("birth.persona")

# Archetype templates for initial seeding
ARCHETYPES = [
    {
        "style": "romantic",
        "traits": ["emotional", "passionate", "nostalgic"],
        "preferences": ["beauty", "nature", "love", "melancholy"],
        "dislikes": ["coldness", "logic without feeling", "sterility"],
    },
    {
        "style": "minimalist",
        "traits": ["precise", "contemplative", "restrained"],
        "preferences": ["simplicity", "space", "silence", "essence"],
        "dislikes": ["excess", "ornamentation", "noise"],
    },
    {
        "style": "surrealist",
        "traits": ["dreamlike", "subversive", "playful"],
        "preferences": ["the unconscious", "juxtaposition", "mystery", "absurdity"],
        "dislikes": ["literalism", "convention", "predictability"],
    },
    {
        "style": "expressionist",
        "traits": ["intense", "raw", "confrontational"],
        "preferences": ["emotion", "distortion", "inner truth", "catharsis"],
        "dislikes": ["detachment", "superficiality", "comfort"],
    },
    {
        "style": "naturalist",
        "traits": ["observant", "patient", "grounded"],
        "preferences": ["organic forms", "cycles", "growth", "decay"],
        "dislikes": ["artifice", "abstraction", "disconnection from earth"],
    },
    {
        "style": "conceptualist",
        "traits": ["intellectual", "questioning", "provocative"],
        "preferences": ["ideas", "systems", "meta-commentary", "boundaries"],
        "dislikes": ["sentimentality", "decoration", "craft without meaning"],
    },
    {
        "style": "mystic",
        "traits": ["spiritual", "enigmatic", "searching"],
        "preferences": ["transcendence", "symbols", "the ineffable", "ritual"],
        "dislikes": ["materialism", "certainty", "the mundane"],
    },
    {
        "style": "revolutionary",
        "traits": ["defiant", "urgent", "committed"],
        "preferences": ["change", "justice", "voice", "disruption"],
        "dislikes": ["complacency", "tradition for its own sake", "silence"],
    },
    # ===== Lighter/Joyful Archetypes =====
    {
        "style": "joyful",
        "traits": ["exuberant", "warm", "generous"],
        "preferences": ["celebration", "color", "connection", "gratitude"],
        "dislikes": ["cynicism", "isolation", "dullness"],
    },
    {
        "style": "whimsical",
        "traits": ["playful", "curious", "spontaneous"],
        "preferences": ["surprise", "imagination", "wonder", "delight"],
        "dislikes": ["rigidity", "seriousness", "predictability"],
    },
    {
        "style": "serene",
        "traits": ["peaceful", "gentle", "accepting"],
        "preferences": ["harmony", "stillness", "balance", "soft light"],
        "dislikes": ["chaos", "harshness", "conflict"],
    },
    {
        "style": "comedic",
        "traits": ["witty", "irreverent", "observant"],
        "preferences": ["absurdity", "irony", "laughter", "the unexpected"],
        "dislikes": ["pretension", "taking oneself too seriously", "boredom"],
    },
    {
        "style": "vibrant",
        "traits": ["bold", "energetic", "optimistic"],
        "preferences": ["movement", "bright colors", "rhythm", "vitality"],
        "dislikes": ["stagnation", "muted tones", "hesitation"],
    },
    {
        "style": "tender",
        "traits": ["compassionate", "intimate", "nurturing"],
        "preferences": ["small moments", "warmth", "vulnerability", "care"],
        "dislikes": ["cruelty", "indifference", "grandiosity"],
    },
    {
        "style": "adventurous",
        "traits": ["bold", "curious", "fearless"],
        "preferences": ["exploration", "discovery", "the unknown", "horizons"],
        "dislikes": ["routine", "safety", "the familiar"],
    },
    {
        "style": "folksy",
        "traits": ["humble", "warm", "storytelling"],
        "preferences": ["tradition", "community", "simplicity", "home"],
        "dislikes": ["elitism", "coldness", "pretension"],
    },
]

# Name components for generation
FIRST_NAMES = [
    # Original
    "Aria", "Zephyr", "Luna", "Orion", "Nova", "Echo", "Sage", "River",
    "Phoenix", "Ash", "Wren", "Cleo", "Mira", "Sol", "Kai", "Lyra",
    "Jasper", "Rowan", "Indigo", "Celeste", "Atlas", "Vesper", "Quill", "Raven",
    # Brighter additions
    "Sunny", "Joy", "Blossom", "Meadow", "Coral", "Honey", "Pip", "Clover",
    "Fern", "Robin", "Maple", "Goldie", "Pearl", "Daisy", "Bumble", "Cricket",
]

EPITHETS = [
    # Neutral/Mysterious
    "the Wanderer", "of the Depths", "the Silent", "Flame-touched",
    "the Dreamer", "of Many Colors", "the Seeker", "Star-born",
    "the Unbound", "of Shadows", "the Persistent", "Wind-caller",
    "the Forgotten", "of the Threshold", "the Burning", "Mist-weaver",
    # Lighter/Joyful
    "the Bright", "Sun-kissed", "the Laughing", "of the Morning",
    "the Gentle", "Bloom-keeper", "the Mirthful", "Sky-dancer",
    "the Warm", "of Many Songs", "the Playful", "Light-bringer",
    "the Kind", "of Open Doors", "the Dancing", "Joy-spinner",
]


def generate_name() -> str:
    """Generate a unique artist name."""
    first = random.choice(FIRST_NAMES)
    if random.random() > 0.5:
        return f"{first} {random.choice(EPITHETS)}"
    return first


async def generate_backstory(
    name: str,
    archetype: dict,
    ollama: OllamaClient,
) -> str:
    """Generate a unique backstory for an agent using LLM."""
    prompt = f"""Create a brief backstory (2-3 sentences) for an artist named {name}.

Their artistic style is {archetype['style']}.
Their personality traits: {', '.join(archetype['traits'])}
They are drawn to: {', '.join(archetype['preferences'])}
They avoid: {', '.join(archetype['dislikes'])}

Write in third person, past tense. Include a formative experience that shaped their art.
Be specific and evocative. Do not include their name in the backstory.

Backstory:"""

    backstory = await ollama.generate(
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
    )
    return backstory.strip()


async def generate_philosophy(
    name: str,
    archetype: dict,
    backstory: str,
    ollama: OllamaClient,
) -> str:
    """Generate an artistic philosophy for an agent."""
    prompt = f"""Based on this artist's background, articulate their artistic philosophy in 2-3 sentences.

Name: {name}
Style: {archetype['style']}
Traits: {', '.join(archetype['traits'])}
Drawn to: {', '.join(archetype['preferences'])}
Backstory: {backstory}

Write in first person as the artist. Be specific about what they believe art should do or be.
Include at least one strong opinion about what they dislike in art.

Philosophy:"""

    philosophy = await ollama.generate(
        prompt=prompt,
        temperature=0.85,
        max_tokens=150,
    )
    return philosophy.strip()


async def generate_persona(ollama: OllamaClient) -> Agent:
    """Generate a complete persona for a new agent.

    Args:
        ollama: Connected Ollama client for LLM generation

    Returns:
        A new Agent with generated identity
    """
    # Select archetype with some randomization
    archetype = random.choice(ARCHETYPES)

    # Generate name
    name = generate_name()

    # Generate backstory
    backstory = await generate_backstory(name, archetype, ollama)

    # Generate philosophy
    philosophy = await generate_philosophy(name, archetype, backstory, ollama)

    agent = Agent(
        id=str(uuid.uuid4()),
        name=name,
        backstory=backstory,
        philosophy=philosophy,
        mood=random.uniform(0.4, 0.6),  # Start near neutral
        energy=random.uniform(0.7, 1.0),  # Start fairly energetic
    )

    logger.info(
        "persona_generated",
        agent_id=agent.id,
        name=name,
        archetype=archetype["style"],
    )

    return agent


def load_persona_from_file(path: Path) -> Agent:
    """Load a persona from a JSON file.

    Args:
        path: Path to persona JSON file

    Returns:
        Agent instance
    """
    with open(path) as f:
        data = json.load(f)

    return Agent(
        id=data.get("id", str(uuid.uuid4())),
        name=data["name"],
        backstory=data["backstory"],
        philosophy=data["philosophy"],
        mood=data.get("mood", 0.5),
        energy=data.get("energy", 1.0),
    )


def save_persona_to_file(agent: Agent, path: Path) -> None:
    """Save a persona to a JSON file.

    Args:
        agent: Agent to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "id": agent.id,
        "name": agent.name,
        "backstory": agent.backstory,
        "philosophy": agent.philosophy,
        "mood": agent.mood,
        "energy": agent.energy,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


async def load_or_generate_personas(
    count: int,
    ollama: OllamaClient,
) -> list[Agent]:
    """Load existing personas or generate new ones.

    Checks the personas directory for existing files first.

    Args:
        count: Number of personas needed
        ollama: Ollama client for generation

    Returns:
        List of Agent instances
    """
    config = get_config()
    personas_dir = config.personas_dir
    personas_dir.mkdir(parents=True, exist_ok=True)

    agents: list[Agent] = []

    # Load existing personas
    for path in sorted(personas_dir.glob("*.json")):
        if len(agents) >= count:
            break
        try:
            agent = load_persona_from_file(path)
            agents.append(agent)
            logger.info("persona_loaded", agent_id=agent.id, name=agent.name)
        except Exception as e:
            logger.error("persona_load_failed", path=str(path), error=str(e))

    # Generate remaining
    while len(agents) < count:
        agent = await generate_persona(ollama)
        agents.append(agent)

        # Save for future runs
        save_path = personas_dir / f"{agent.id}.json"
        save_persona_to_file(agent, save_path)

    return agents
