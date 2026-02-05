"""Configuration management for Birth simulation."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama LLM configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    vision_model: str = "llava"  # For image understanding
    timeout: float = 120.0
    max_retries: int = 3


class StableDiffusionSettings(BaseSettings):
    """Stable Diffusion API configuration."""

    model_config = SettingsConfigDict(env_prefix="SD_")

    base_url: str = "http://localhost:7860"
    timeout: float = 300.0
    max_retries: int = 3
    default_steps: int = 30
    default_cfg_scale: float = 7.0
    default_width: int = 512
    default_height: int = 512


class SimulationSettings(BaseSettings):
    """Core simulation parameters."""

    model_config = SettingsConfigDict(env_prefix="SIM_")

    # Agent parameters
    initial_agent_count: int = 20
    min_rest_period: float = 5.0  # seconds
    max_rest_period: float = 30.0  # seconds

    # Memory parameters
    memory_importance_threshold: float = 0.6
    recent_memory_limit: int = 50
    reflection_interval: int = 10  # actions between reflections

    # Sentiment parameters
    sentiment_decay_rate: float = 0.01  # per cycle
    sentiment_update_weight: float = 0.3

    # Creation parameters
    art_text_max_length: int = 2000
    art_description_max_length: int = 500


class Config(BaseSettings):
    """Main configuration container."""

    model_config = SettingsConfigDict(
        env_prefix="BIRTH_",
        env_nested_delimiter="__",
    )

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path | None = None
    output_dir: Path | None = None
    database_path: Path | None = None

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_to_file: bool = True

    # Sub-configurations
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    stable_diffusion: StableDiffusionSettings = Field(default_factory=StableDiffusionSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)

    def model_post_init(self, __context) -> None:
        """Set derived paths after initialization."""
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.output_dir is None:
            self.output_dir = self.base_dir / "outputs"
        if self.database_path is None:
            self.database_path = self.data_dir / "birth.db"

    @property
    def gallery_dir(self) -> Path:
        """Path to gallery of inspirational assets."""
        return self.data_dir / "gallery"

    @property
    def personas_dir(self) -> Path:
        """Path to persona templates."""
        return self.data_dir / "personas"

    @property
    def artworks_dir(self) -> Path:
        """Path to generated artworks."""
        return self.output_dir / "artworks"

    @property
    def logs_dir(self) -> Path:
        """Path to log files."""
        return self.output_dir / "logs"

    @property
    def drops_dir(self) -> Path:
        """Path to image drops folder - images placed here are shown to agents."""
        return self.data_dir / "drops"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    config = Config()
    return config
