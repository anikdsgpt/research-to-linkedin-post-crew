from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _looks_like_article(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "introduction:",
        "key findings",
        "sources:",
        "conclusion:",
        "comprehensive overview",
        "---",
    ]
    return any(m in lowered for m in markers)


def _build_strict_prompt(topic: str) -> str:
    return (
        f"Editorial cinematic illustration about {topic}: a sales professional reviewing an AI prospecting "
        "dashboard with lead scoring and intent signals; modern office setting; medium shot; clean composition "
        "with subject on left and analytics panels on right; blue-gray palette with warm accent highlights; "
        "soft directional window light; confident strategic mood; high detail; no text, no logo, no watermark."
    )


def sanitize_image_prompt_output(topic: str) -> bool:
    post_file = Path("output/linkedin_post.md")
    prompt_file = Path("output/generated_images/image_prompt.txt")

    if not post_file.exists() or not prompt_file.exists():
        return False

    post_text = post_file.read_text(encoding="utf-8")
    prompt_text = prompt_file.read_text(encoding="utf-8")
    post_norm = _normalize(post_text)
    prompt_norm = _normalize(prompt_text)

    if not prompt_norm:
        prompt_file.write_text(_build_strict_prompt(topic), encoding="utf-8")
        return True

    similarity = SequenceMatcher(None, post_norm, prompt_norm).ratio()
    too_long = len(prompt_norm.split()) > 120
    invalid_shape = _looks_like_article(prompt_text)

    if similarity >= 0.55 or too_long or invalid_shape:
        prompt_file.write_text(_build_strict_prompt(topic), encoding="utf-8")
        return True

    return False
