"""Commons - Shared space for agent creations.

The Commons is where all agent-created works are published.
This allows agents to see, react to, and be influenced by each other's creations.
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from birth.config import get_config


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in filenames."""
    # Replace spaces with underscores, remove special chars
    sanitized = re.sub(r'[^\w\-]', '_', name)
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')
from birth.integrations.ollama import OllamaClient
from birth.integrations.stable_diffusion import StableDiffusionClient
from birth.observation.logger import get_logger
from birth.storage.models import Agent, Artwork
from birth.storage.repository import Repository

if TYPE_CHECKING:
    pass

logger = get_logger("birth.commons")


class Commons:
    """Shared space for agent artworks."""

    def __init__(
        self,
        repository: Repository,
        artworks_dir: Path | None = None,
        sd_client: StableDiffusionClient | None = None,
    ):
        config = get_config()
        self._repository = repository
        self._artworks_dir = artworks_dir or config.artworks_dir
        self._sd_client = sd_client

        # Ensure directories exist
        (self._artworks_dir / "text").mkdir(parents=True, exist_ok=True)
        (self._artworks_dir / "images").mkdir(parents=True, exist_ok=True)

    async def create_artwork(
        self,
        creator: Agent,
        ollama: OllamaClient,
        inspiration_context: str = "",
        sentiment_summary: str = "",
        medium: str | None = None,
    ) -> Artwork | None:
        """Create a new artwork.

        Args:
            creator: The creating agent
            ollama: Ollama client for text generation
            inspiration_context: Recent inspirations
            sentiment_summary: Agent's current feelings
            medium: Force specific medium ('text', 'image', 'mixed')

        Returns:
            Created Artwork, or None on failure
        """
        # Decide medium if not specified
        if medium is None:
            medium = await self._decide_medium(creator, ollama)

        artwork_id = str(uuid.uuid4())

        # Generate title and content based on medium
        if medium == "text":
            artwork = await self._create_text_artwork(
                artwork_id, creator, ollama, inspiration_context, sentiment_summary
            )
        elif medium == "image":
            # Create an image prompt (not actual image generation)
            artwork = await self._create_image_prompt_artwork(
                artwork_id, creator, ollama, inspiration_context, sentiment_summary
            )
        else:
            # Mixed: text piece + image prompt
            artwork = await self._create_mixed_artwork(
                artwork_id, creator, ollama, inspiration_context, sentiment_summary
            )

        if artwork:
            await self._repository.create_artwork(artwork)
            logger.info(
                "artwork_created",
                artwork_id=artwork.id,
                creator_id=creator.id,
                title=artwork.title,
                medium=artwork.medium,
            )

        return artwork

    async def _decide_medium(self, creator: Agent, ollama: OllamaClient) -> str:
        """Decide what medium to work in based on agent's philosophy."""
        prompt = f"""You are {creator.name}.
Your philosophy: {creator.philosophy}

What medium do you want to create in right now?
- text: poetry, prose, manifesto, meditation
- image: a visual artwork (you'll write the vision/prompt)
- mixed: both text and visual elements

Just respond with one word:"""

        response = await ollama.generate(prompt=prompt, temperature=0.7, max_tokens=10)
        response = response.strip().lower()

        if response in ["text", "image", "mixed"]:
            return response

        return "mixed"  # Default to mixed for variety

    async def _create_text_artwork(
        self,
        artwork_id: str,
        creator: Agent,
        ollama: OllamaClient,
        inspiration_context: str,
        sentiment_summary: str,
    ) -> Artwork | None:
        """Create a text-based artwork (poem, prose, manifesto)."""
        config = get_config().simulation

        # Generate the artwork
        prompt = f"""You are {creator.name}.

{creator.philosophy}

{creator.backstory}

WHAT YOU'VE EXPERIENCED RECENTLY:
{inspiration_context if inspiration_context else "You've been in contemplation."}

HOW YOU'RE FEELING:
{sentiment_summary if sentiment_summary else "Present and aware."}

Create a piece of text art - whatever form feels right to you now. Let it emerge from who you are and what you're experiencing.

---
TITLE: [your title]

[your work]
---"""

        response = await ollama.generate(
            prompt=prompt,
            temperature=0.85,
            max_tokens=config.art_text_max_length,
        )

        # Parse title and content
        title, content = self._parse_text_artwork(response)

        if not content:
            return None

        # Generate style tags
        style_tags = await self._generate_style_tags(content, ollama)

        # Save to file
        safe_name = _sanitize_name(creator.name)
        text_path = self._artworks_dir / "text" / f"{safe_name}_{artwork_id[:8]}.txt"
        text_path.write_text(f"# {title}\n\nby {creator.name}\n\n{content}")

        return Artwork(
            id=artwork_id,
            creator_id=creator.id,
            title=title,
            medium="text",
            content_text=content,
            image_path=None,
            style_tags=style_tags,
            created_at=datetime.utcnow(),
        )

    async def _create_image_prompt_artwork(
        self,
        artwork_id: str,
        creator: Agent,
        ollama: OllamaClient,
        inspiration_context: str,
        sentiment_summary: str,
    ) -> Artwork | None:
        """Create a visual artwork as an image generation prompt.

        The agent describes their vision in detail, which can later be
        used with image generation tools like ComfyUI or Stable Diffusion.
        """
        # Generate detailed visual description
        prompt = f"""You are {creator.name}.

{creator.philosophy}

{creator.backstory}

WHAT YOU'VE EXPERIENCED RECENTLY:
{inspiration_context if inspiration_context else "You've been in contemplation."}

HOW YOU'RE FEELING:
{sentiment_summary if sentiment_summary else "Present and aware."}

Create a visual artwork. Let it emerge from who you are and what you're experiencing.

TITLE: [your title]

STATEMENT: [what this piece means to you - 2-3 sentences]

PROMPT: [Write a rich, vivid image description for an AI generator. 150-250 words minimum. Paint the scene with words - describe what we SEE in cinematic detail:
- The scene/subject: What is happening? Who or what is present? What are they doing?
- Environment/setting: Where is this? What surrounds the subject?
- Lighting: What is the quality of light? Where does it come from? What mood does it create?
- Specific colors: Name exact colors (crimson, burnt sienna, cerulean, etc.)
- Textures and materials: What surfaces do we see? How do they feel?
- Atmosphere/mood: What emotion pervades the scene?
Do NOT reference yourself, "the artist", or any context. Describe ONLY what appears in the image.]

STYLE: [artistic style, influences, techniques - be specific]

COLORS: [list the key colors in the palette]

COMPOSITION: [how is the image arranged? what dominates? what recedes?]

NEGATIVE: [what to avoid in generation]"""

        response = await ollama.generate(
            prompt=prompt,
            temperature=0.85,
            max_tokens=1200,
        )

        # Parse response
        title = "Untitled Vision"
        statement = ""
        image_prompt = ""
        style = ""
        colors = ""
        composition = ""
        negative = "blurry, low quality, text, watermark, signature"

        current_section = None
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("TITLE:"):
                title = line_stripped.split(":", 1)[1].strip()
                current_section = None
            elif line_stripped.upper().startswith("STATEMENT:"):
                current_section = "statement"
            elif line_stripped.upper().startswith("PROMPT:"):
                current_section = "prompt"
            elif line_stripped.upper().startswith("STYLE:"):
                current_section = "style"
            elif line_stripped.upper().startswith("COLORS:"):
                current_section = "colors"
            elif line_stripped.upper().startswith("COMPOSITION:"):
                current_section = "composition"
            elif line_stripped.upper().startswith("NEGATIVE:"):
                current_section = "negative"
            elif current_section == "statement" and line_stripped:
                statement += line_stripped + " "
            elif current_section == "prompt" and line_stripped:
                image_prompt += line_stripped + " "
            elif current_section == "style" and line_stripped:
                style += line_stripped + " "
            elif current_section == "colors" and line_stripped:
                colors += line_stripped + " "
            elif current_section == "composition" and line_stripped:
                composition += line_stripped + " "
            elif current_section == "negative" and line_stripped:
                negative += line_stripped + " "

        statement = statement.strip()
        image_prompt = image_prompt.strip()
        style = style.strip()
        colors = colors.strip()
        composition = composition.strip()
        negative = negative.strip()

        if not image_prompt:
            return None

        # Combine into displayable content with all sections
        full_content = f"""[VISUAL ARTWORK]

{statement}

---
IMAGE GENERATION PROMPT:
{image_prompt}

STYLE: {style if style else "Not specified"}

COLORS: {colors if colors else "Not specified"}

COMPOSITION: {composition if composition else "Not specified"}

NEGATIVE PROMPT:
{negative}"""

        # Generate style tags
        style_tags = await self._generate_style_tags(image_prompt, ollama)
        style_tags.insert(0, "visual")  # Mark as visual artwork

        # Save prompt to file for potential later use
        prompts_dir = self._artworks_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _sanitize_name(creator.name)
        prompt_path = prompts_dir / f"{safe_name}_{artwork_id[:8]}.txt"
        prompt_path.write_text(
            f"# {title}\n"
            f"# by {creator.name}\n"
            f"# ID: {artwork_id}\n\n"
            f"## Artistic Statement\n{statement}\n\n"
            f"## Image Prompt\n{image_prompt}\n\n"
            f"## Style\n{style if style else 'Not specified'}\n\n"
            f"## Colors\n{colors if colors else 'Not specified'}\n\n"
            f"## Composition\n{composition if composition else 'Not specified'}\n\n"
            f"## Negative Prompt\n{negative}\n"
        )

        return Artwork(
            id=artwork_id,
            creator_id=creator.id,
            title=title,
            medium="image",
            content_text=full_content,
            image_path=str(prompt_path),  # Path to prompt file
            style_tags=style_tags,
            created_at=datetime.utcnow(),
        )

    # Keep original method for future use with actual image generation
    async def _create_image_artwork_with_generation(
        self,
        artwork_id: str,
        creator: Agent,
        ollama: OllamaClient,
        inspiration_context: str,
        sentiment_summary: str,
    ) -> Artwork | None:
        """Create an image artwork with actual generation (requires SD client)."""
        if not self._sd_client:
            return None

        # First create the prompt
        prompt_artwork = await self._create_image_prompt_artwork(
            artwork_id, creator, ollama, inspiration_context, sentiment_summary
        )

        if not prompt_artwork:
            return None

        # Extract the image prompt from content
        content = prompt_artwork.content_text or ""
        image_prompt = ""
        negative = "blurry, low quality, text, watermark"

        if "IMAGE GENERATION PROMPT:" in content:
            parts = content.split("IMAGE GENERATION PROMPT:")
            if len(parts) > 1:
                prompt_section = parts[1].split("NEGATIVE PROMPT:")[0]
                image_prompt = prompt_section.strip()

        if "NEGATIVE PROMPT:" in content:
            negative = content.split("NEGATIVE PROMPT:")[-1].strip()

        if not image_prompt:
            return prompt_artwork  # Return prompt-only version

        # Generate actual image
        try:
            image_path = self._artworks_dir / "images" / f"{artwork_id}.png"
            await self._sd_client.generate_and_save(
                prompt=image_prompt,
                output_path=image_path,
                negative_prompt=negative,
            )
            prompt_artwork.image_path = str(image_path)
            return prompt_artwork

        except Exception as e:
            logger.error("image_generation_failed", error=str(e))
            return prompt_artwork  # Return prompt-only on failure

    async def _create_mixed_artwork(
        self,
        artwork_id: str,
        creator: Agent,
        ollama: OllamaClient,
        inspiration_context: str,
        sentiment_summary: str,
    ) -> Artwork | None:
        """Create a mixed-media artwork (text + accompanying image prompt)."""
        # Start with text
        text_artwork = await self._create_text_artwork(
            artwork_id, creator, ollama, inspiration_context, sentiment_summary
        )

        if not text_artwork:
            return None

        # Generate an accompanying image prompt for the text piece
        if text_artwork.content_text:
            try:
                # Generate image prompt from text
                image_prompt = await ollama.generate(
                    prompt=f"""You are {creator.name}. {creator.philosophy}

You wrote this text piece:
{text_artwork.content_text[:600]}

Create a rich visual companion for this work. Write a vivid, detailed image description (150-250 words) that captures the essence and emotion of your text in visual form.

Paint the scene with words:
- What is the subject/scene? Describe it cinematically.
- What is the lighting? Where does it come from?
- Name specific colors (crimson, burnt sienna, cerulean, etc.)
- What textures and materials are visible?
- What mood/atmosphere pervades the image?

Do NOT reference the text, yourself, or "the artist". Describe ONLY what appears in the image.

End with:
STYLE: [artistic style/technique]
COLORS: [color palette]

PROMPT:""",
                    temperature=0.9,
                    max_tokens=500,
                )

                image_prompt = image_prompt.strip()

                # Save the image prompt
                prompts_dir = self._artworks_dir / "prompts"
                prompts_dir.mkdir(parents=True, exist_ok=True)
                safe_name = _sanitize_name(creator.name)
                prompt_path = prompts_dir / f"{safe_name}_{artwork_id[:8]}_visual.txt"
                prompt_path.write_text(
                    f"# Visual companion for: {text_artwork.title}\n"
                    f"# by {creator.name}\n"
                    f"# ID: {artwork_id}\n\n"
                    f"## Image Prompt\n{image_prompt}\n\n"
                    f"## Negative Prompt\ntext, words, letters, watermark, blurry\n"
                )

                # Append visual prompt to content
                text_artwork.content_text += f"\n\n---\n[VISUAL COMPANION]\n{image_prompt}"
                text_artwork.medium = "mixed"
                text_artwork.image_path = str(prompt_path)
                if "visual" not in text_artwork.style_tags:
                    text_artwork.style_tags.append("visual")

            except Exception as e:
                logger.warning("mixed_image_failed", error=str(e))
                # Still return text-only version

        return text_artwork

    def _parse_text_artwork(self, response: str) -> tuple[str, str]:
        """Parse title and content from LLM response."""
        title = "Untitled"
        content = response.strip()

        # Try to extract title - multiple strategies
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Strategy 1: Explicit TITLE: marker
            if line_stripped.upper().startswith("TITLE:"):
                title = line_stripped.split(":", 1)[1].strip().strip('"\'')
                content = "\n".join(lines[i + 1 :]).strip()
                break

            # Strategy 2: Markdown header
            elif line_stripped.startswith("#"):
                title = line_stripped.lstrip("#").strip().strip('"\'')
                content = "\n".join(lines[i + 1 :]).strip()
                break

            # Strategy 3: First non-empty line in quotes or bold
            elif i < 3 and line_stripped and not line_stripped.startswith("---"):
                # Check for quoted title
                if line_stripped.startswith('"') and line_stripped.endswith('"'):
                    title = line_stripped.strip('"')
                    content = "\n".join(lines[i + 1 :]).strip()
                    break
                # Check for bold title **title**
                elif line_stripped.startswith("**") and "**" in line_stripped[2:]:
                    title = line_stripped.split("**")[1]
                    content = "\n".join(lines[i + 1 :]).strip()
                    break

        # Clean up content - remove leading/trailing dashes and markers
        content = content.strip("-").strip()
        if content.startswith("---"):
            content = content[3:].strip()

        # If title is still empty or just whitespace, try first meaningful line
        if not title or title == "Untitled":
            for line in lines[:5]:
                line = line.strip().strip("#-*\"'")
                if line and len(line) > 3 and len(line) < 100:
                    title = line
                    break

        return title, content

    async def _generate_style_tags(self, content: str, ollama: OllamaClient) -> list[str]:
        """Generate style tags for an artwork."""
        prompt = f"""Analyze this artwork and provide 3-5 style/theme tags.

{content[:500]}

Tags (comma-separated, lowercase):"""

        response = await ollama.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=50,
        )

        tags = [
            tag.strip().lower()
            for tag in response.split(",")
            if tag.strip()
        ]

        return tags[:5]

    async def get_recent(self, limit: int = 20) -> list[Artwork]:
        """Get recent artworks.

        Args:
            limit: Max artworks to return

        Returns:
            List of recent Artworks
        """
        return await self._repository.get_recent_artworks(limit)

    async def get_by_creator(self, creator_id: str) -> list[Artwork]:
        """Get all artworks by a specific creator.

        Args:
            creator_id: Creator's agent ID

        Returns:
            List of Artworks
        """
        return await self._repository.get_artworks_by_creator(creator_id)

    async def get_artwork(self, artwork_id: str) -> Artwork | None:
        """Get a specific artwork.

        Args:
            artwork_id: Artwork ID

        Returns:
            Artwork or None
        """
        return await self._repository.get_artwork(artwork_id)

    @property
    def artworks_dir(self) -> Path:
        """Path to artworks directory."""
        return self._artworks_dir
