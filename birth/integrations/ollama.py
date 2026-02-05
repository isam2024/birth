"""Ollama LLM client for agent cognition."""

import asyncio
import base64
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from birth.config import OllamaSettings, get_config
from birth.observation.logger import get_logger

logger = get_logger("birth.ollama")


class OllamaError(Exception):
    """Base exception for Ollama errors."""

    pass


class OllamaConnectionError(OllamaError):
    """Connection to Ollama failed."""

    pass


class OllamaGenerationError(OllamaError):
    """Text generation failed."""

    pass


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, settings: OllamaSettings | None = None):
        self._settings = settings or get_config().ollama
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OllamaClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=httpx.Timeout(self._settings.timeout),
        )
        # Verify connection
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            logger.info("ollama_connected", models=len(response.json().get("models", [])))
        except httpx.HTTPError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, OllamaGenerationError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            model: Model to use (defaults to config)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Generated text
        """
        if not self._client:
            raise OllamaError("Client not connected")

        model = model or self._settings.model

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if stop:
            payload["options"]["stop"] = stop

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except httpx.HTTPStatusError as e:
            logger.error("ollama_generation_failed", status_code=e.response.status_code)
            raise OllamaGenerationError(f"Generation failed: {e}") from e
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            # Don't log cancelled errors (expected during shutdown)
            if not isinstance(e, asyncio.CancelledError):
                logger.error("ollama_request_failed", error=str(e))
            raise

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate text with streaming response.

        Yields:
            Text chunks as they are generated
        """
        if not self._client:
            raise OllamaError("Client not connected")

        model = model or self._settings.model

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        async with self._client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Chat completion with message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            model: Model to use
            temperature: Sampling temperature

        Returns:
            Assistant's response
        """
        if not self._client:
            raise OllamaError("Client not connected")

        model = model or self._settings.model

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            logger.error("ollama_chat_failed", error=str(e))
            raise OllamaGenerationError(f"Chat failed: {e}") from e

    async def list_models(self) -> list[str]:
        """List available models."""
        if not self._client:
            raise OllamaError("Client not connected")

        response = await self._client.get("/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]

    async def is_model_available(self, model: str | None = None) -> bool:
        """Check if a model is available."""
        model = model or self._settings.model
        available = await self.list_models()
        return model in available

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, OllamaGenerationError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def vision(
        self,
        prompt: str,
        image: bytes | Path | str,
        model: str = "llava",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Analyze an image with a vision model.

        Args:
            prompt: Question or instruction about the image
            image: Image as bytes, file path, or base64 string
            model: Vision model to use (llava, bakllava, llava-llama3, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Model's description/analysis of the image
        """
        if not self._client:
            raise OllamaError("Client not connected")

        # Convert image to base64
        if isinstance(image, Path):
            image_b64 = base64.b64encode(image.read_bytes()).decode("utf-8")
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        else:
            # Assume already base64
            image_b64 = image

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info("vision_completed", model=model, prompt_len=len(prompt))
            return result.get("response", "")
        except httpx.HTTPStatusError as e:
            logger.error("vision_failed", status_code=e.response.status_code)
            raise OllamaGenerationError(f"Vision analysis failed: {e}") from e
        except httpx.HTTPError as e:
            logger.error("vision_request_failed", error=str(e))
            raise

    async def describe_image(
        self,
        image: bytes | Path | str,
        model: str = "llava",
        style: str = "artistic",
    ) -> str:
        """Get an artistic description of an image.

        Args:
            image: Image to describe
            model: Vision model to use
            style: Description style ('artistic', 'detailed', 'emotional')

        Returns:
            Rich description of the image
        """
        prompts = {
            "artistic": (
                "You are an art critic with deep knowledge of visual art, photography, and aesthetics. "
                "Analyze this image thoroughly and write a rich, evocative description.\n\n"
                "Cover these aspects:\n"
                "1. SUBJECT: What is depicted? Describe the main subjects, figures, objects, or scenes.\n"
                "2. COMPOSITION: How is the image structured? Discuss framing, balance, focal points, "
                "use of space, and visual flow.\n"
                "3. COLOR & LIGHT: What is the color palette? Describe the lighting - its quality, "
                "direction, and emotional effect. Name specific colors (crimson, azure, ochre, etc.).\n"
                "4. MOOD & ATMOSPHERE: What feeling does this image evoke? What is its emotional temperature?\n"
                "5. SYMBOLISM & MEANING: What might this image represent? What themes or ideas does it suggest?\n"
                "6. ARTISTIC STYLE: What artistic influences, techniques, or movements does this evoke?\n\n"
                "Write 3-4 rich paragraphs. Be specific, evocative, and poetic."
            ),
            "detailed": (
                "Describe everything you see in this image in comprehensive detail. "
                "Include all objects, people, setting, background, lighting conditions, colors, textures, "
                "and any visible text or symbols. Note spatial relationships between elements. "
                "Describe from foreground to background. Be thorough, precise, and systematic."
            ),
            "emotional": (
                "What emotions and feelings does this image evoke? "
                "What story might it be telling? What questions does it raise? "
                "What memories or associations might it trigger? "
                "If this image could speak, what would it say? "
                "Respond as if you were deeply moved by this image. Be introspective and poetic."
            ),
        }

        prompt = prompts.get(style, prompts["artistic"])
        return await self.vision(prompt, image, model=model)


# Global client instance
_client: OllamaClient | None = None


async def get_ollama_client() -> OllamaClient:
    """Get or create the global Ollama client."""
    global _client
    if _client is None:
        _client = OllamaClient()
        await _client.connect()
    return _client


async def close_ollama_client() -> None:
    """Close the global Ollama client."""
    global _client
    if _client:
        await _client.close()
        _client = None
