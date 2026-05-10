# AI Agent Web Research to LinkedIn Post

A practical CrewAI project that runs a 3-agent workflow to research a topic and generate a polished LinkedIn post.

The pipeline is designed for reliability with local LLM fallback, optional tool usage, and clear markdown output.

## What this project does

Given a topic, the workflow:

1. Runs web research with a dedicated internet researcher agent.
2. Runs YouTube intelligence research with a dedicated video analyst agent.
3. Synthesizes both research outputs into a publication-ready LinkedIn post.
4. Saves the final result to `output/linkedin_post.md`.

## Architecture

- Agent 1: Senior Internet Research Strategist
- Agent 2: Senior YouTube Intelligence Analyst
- Agent 3: Senior LinkedIn Thought-Leadership Writer
- Process: Sequential CrewAI process
- Collaboration: Delegation enabled for research agents
- Writer guidance: Knowledge source loaded from `skills/linkedin_post_writing_guide.md`

## Key features

- Local model support through llama.cpp OpenAI-compatible server
- Optional external tools toggle for small-model compatibility
- Verbose agent execution logs for debugging and transparency
- Topic configurable via environment variable
- Output persisted as markdown for easy publishing

## Requirements

- Python >= 3.10 and < 3.14
- `uv` package manager
- CrewAI and crewai-tools dependencies
- Optional: local GGUF model + llama.cpp server

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure environment variables in `.env`.

Example local setup:

```env
MODEL=hosted_vllm/qwen2.5-0.5b-instruct-q4_k_m
VLLM_BASE_URL=http://127.0.0.1:8000/v1
USE_TOOLS=false
TOPIC=AI Agents for Sales Prospecting
```

Example cloud-assisted setup:

```env
USE_TOOLS=true
SERPER_API_KEY=your_serper_key
```

## Run

If using local inference, start llama.cpp server first.

Then run:

```bash
python src/ai_agent_web_research_linkedin/main.py
```

## Output

- Final LinkedIn post file: `output/linkedin_post.md`

## Repository highlights

- `src/ai_agent_web_research_linkedin/main.py`: Crew, agents, and tasks
- `src/ai_agent_web_research_linkedin/skills/linkedin_post_writing_guide.md`: Writer quality guide used as knowledge source
- `output/`: Generated markdown results
- `models/`: Local GGUF model files

## Notes

- `knowledge_sources` is the correct CrewAI parameter for agent knowledge injection.
- This project is optimized for practical execution on local/free setups when cloud quotas fail.
