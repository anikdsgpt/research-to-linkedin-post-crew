from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    use_tools: bool
    use_knowledge: bool
    topic: str


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        model=os.getenv("MODEL", "hosted_vllm/qwen2.5-0.5b-instruct-q4_k_m"),
        use_tools=os.getenv("USE_TOOLS", "true").strip().lower() == "true",
        use_knowledge=os.getenv("USE_KNOWLEDGE", "true").strip().lower() == "true",
        topic=os.getenv("TOPIC", "AI Agents for Sales Prospecting"),
    )


def load_linkedin_knowledge_sources(use_knowledge: bool) -> list[StringKnowledgeSource]:
    if not use_knowledge:
        return []

    skill_file = Path(__file__).resolve().parent.parent / "skills" / "linkedin_post_writing_guide.md"
    content = skill_file.read_text(encoding="utf-8")
    return [StringKnowledgeSource(content=content)]
