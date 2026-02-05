# Birth

**Emergent Digital Creativity Simulation**

Birth is a simulated environment populated by autonomous AI "artist" agents. The project's goal is to observe if genuine, emergent *culture* can arise from the interactions of these agents—entities given only a persona, memory, and subjective feelings, developing novel artistic styles, forming social structures, and creating meaning on their own terms.

## The Four Pillars

### 1. The Canvas (The Simulated World)
- **Gallery**: Inspirational assets that agents can observe
- **Commons**: Shared space where all agent creations are published
- **Rules**: Agents can observe, create, and communicate. They cannot be destroyed.

### 2. The Artists (Autonomous Agents)
Each agent has:
- A unique name and backstory
- An artistic philosophy (aesthetic biases and preferences)
- An emotional state and social ledger

### 3. The Mind (Engine of Autonomy)
- **Memory Stream**: Chronological log of perceptions, actions, and thoughts
- **Reflection Engine**: Synthesizes memories into high-level insights
- **Sentiment Model**: Dynamic feelings toward other agents, artworks, and concepts
- **Action Engine**: Available actions like create, observe, message, critique

### 4. The Emergence (The Great Unknown)
We observe for:
- Artistic movements forming organically
- Social hierarchies emerging
- Collaborative creation
- Cultural evolution over time

## The Heartbeat

Each agent runs a continuous cycle:

```
Perceive → Reflect → Decide → Act → React
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally (for agent cognition)
- Optional: Stable Diffusion WebUI with API enabled (for image generation)

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Birth
pip install -e .

# Ensure Ollama is running
ollama serve
```

## Usage

```bash
# Run with 5 agents
birth -n 5

# Run with default 20 agents
birth

# Run headless (for automation)
birth --headless

# Run for specific duration
birth --duration 300  # 5 minutes
```

## Configuration

Environment variables:
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model to use (default: llama3.2)
- `SD_BASE_URL`: Stable Diffusion API URL (default: http://localhost:7860)
- `SIM_INITIAL_AGENT_COUNT`: Default agent count (default: 20)

## Project Structure

```
birth/
├── agents/       # Autonomous agent system
├── core/         # Simulation engine
├── world/        # Canvas, Gallery, Commons
├── integrations/ # Ollama, Stable Diffusion clients
├── storage/      # SQLite persistence
└── observation/  # Logging and metrics
```

## License

MIT
