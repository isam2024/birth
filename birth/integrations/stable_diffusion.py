"""Stable Diffusion API client for image generation."""

import base64
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from birth.config import StableDiffusionSettings, get_config
from birth.observation.logger import get_logger

logger = get_logger("birth.stable_diffusion")


class StableDiffusionError(Exception):
    """Base exception for Stable Diffusion errors."""

    pass


class StableDiffusionConnectionError(StableDiffusionError):
    """Connection to Stable Diffusion API failed."""

    pass


class StableDiffusionGenerationError(StableDiffusionError):
    """Image generation failed."""

    pass


class StableDiffusionClient:
    """Async client for Stable Diffusion WebUI API (automatic1111)."""

    def __init__(self, settings: StableDiffusionSettings | None = None):
        self._settings = settings or get_config().stable_diffusion
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "StableDiffusionClient":
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
            response = await self._client.get("/sdapi/v1/sd-models")
            response.raise_for_status()
            models = response.json()
            logger.info("stable_diffusion_connected", model_count=len(models))
        except httpx.HTTPError as e:
            raise StableDiffusionConnectionError(
                f"Failed to connect to Stable Diffusion: {e}"
            ) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, StableDiffusionGenerationError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int = -1,
        sampler_name: str = "Euler a",
    ) -> bytes:
        """Generate an image.

        Args:
            prompt: The positive prompt describing the image
            negative_prompt: What to avoid in the image
            width: Image width (default from config)
            height: Image height (default from config)
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (-1 for random)
            sampler_name: Sampling method

        Returns:
            Image bytes (PNG format)
        """
        if not self._client:
            raise StableDiffusionError("Client not connected")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width or self._settings.default_width,
            "height": height or self._settings.default_height,
            "steps": steps or self._settings.default_steps,
            "cfg_scale": cfg_scale or self._settings.default_cfg_scale,
            "seed": seed,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
        }

        try:
            response = await self._client.post("/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            result = response.json()

            images = result.get("images", [])
            if not images:
                raise StableDiffusionGenerationError("No images returned")

            # Decode base64 image
            image_data = base64.b64decode(images[0])
            logger.info(
                "image_generated",
                prompt_length=len(prompt),
                size=f"{payload['width']}x{payload['height']}",
            )
            return image_data

        except httpx.HTTPStatusError as e:
            logger.error("sd_generation_failed", status_code=e.response.status_code)
            raise StableDiffusionGenerationError(f"Generation failed: {e}") from e
        except httpx.HTTPError as e:
            logger.error("sd_request_failed", error=str(e))
            raise

    async def generate_and_save(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: str = "",
        **kwargs,
    ) -> Path:
        """Generate an image and save to file.

        Args:
            prompt: The positive prompt
            output_path: Where to save the image
            negative_prompt: What to avoid
            **kwargs: Additional generation parameters

        Returns:
            Path to the saved image
        """
        image_data = await self.generate(prompt, negative_prompt, **kwargs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_data)

        logger.info("image_saved", path=str(output_path))
        return output_path

    async def img2img(
        self,
        prompt: str,
        init_image: bytes,
        denoising_strength: float = 0.75,
        negative_prompt: str = "",
        **kwargs,
    ) -> bytes:
        """Generate image from image (img2img).

        Args:
            prompt: The positive prompt
            init_image: Initial image bytes
            denoising_strength: How much to change (0.0-1.0)
            negative_prompt: What to avoid
            **kwargs: Additional parameters

        Returns:
            Generated image bytes
        """
        if not self._client:
            raise StableDiffusionError("Client not connected")

        # Encode image to base64
        init_image_b64 = base64.b64encode(init_image).decode("utf-8")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_images": [init_image_b64],
            "denoising_strength": denoising_strength,
            "width": kwargs.get("width", self._settings.default_width),
            "height": kwargs.get("height", self._settings.default_height),
            "steps": kwargs.get("steps", self._settings.default_steps),
            "cfg_scale": kwargs.get("cfg_scale", self._settings.default_cfg_scale),
        }

        response = await self._client.post("/sdapi/v1/img2img", json=payload)
        response.raise_for_status()
        result = response.json()

        images = result.get("images", [])
        if not images:
            raise StableDiffusionGenerationError("No images returned")

        return base64.b64decode(images[0])

    async def get_models(self) -> list[dict[str, Any]]:
        """Get available models."""
        if not self._client:
            raise StableDiffusionError("Client not connected")

        response = await self._client.get("/sdapi/v1/sd-models")
        response.raise_for_status()
        return response.json()

    async def get_samplers(self) -> list[str]:
        """Get available samplers."""
        if not self._client:
            raise StableDiffusionError("Client not connected")

        response = await self._client.get("/sdapi/v1/samplers")
        response.raise_for_status()
        return [s["name"] for s in response.json()]

    async def progress(self) -> dict[str, Any]:
        """Get current generation progress."""
        if not self._client:
            raise StableDiffusionError("Client not connected")

        response = await self._client.get("/sdapi/v1/progress")
        response.raise_for_status()
        return response.json()


# Global client instance
_client: StableDiffusionClient | None = None


async def get_sd_client() -> StableDiffusionClient:
    """Get or create the global Stable Diffusion client."""
    global _client
    if _client is None:
        _client = StableDiffusionClient()
        await _client.connect()
    return _client


async def close_sd_client() -> None:
    """Close the global Stable Diffusion client."""
    global _client
    if _client:
        await _client.close()
        _client = None
