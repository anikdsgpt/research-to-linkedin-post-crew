#!/usr/bin/env python

import json
import os
import re
import time
from pathlib import Path
from urllib.request import urlretrieve

from crewai import Agent, Task, Crew, Process
from crewai_tools import DallETool, SerperDevTool, YoutubeVideoSearchTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

serper_tool = SerperDevTool()
youtube_video_search_tool = YoutubeVideoSearchTool()
dalle_tool = DallETool(model="dall-e-3", size="1024x1024", quality="standard", n=1)
use_tools = os.getenv("USE_TOOLS", "true").strip().lower() == "true"
use_dalle = os.getenv("USE_DALLE", "true").strip().lower() == "true"
image_output_dir = Path("output") / "generated_images"

# Load LinkedIn writing skill guide
skill_file_path = os.path.join(os.path.dirname(__file__), "skills", "linkedin_post_writing_guide.md")
with open(skill_file_path, "r", encoding="utf-8") as f:
    linkedin_skill_content = f.read()

linkedin_skill_source = StringKnowledgeSource(content=linkedin_skill_content)

web_researcher_agent = Agent(
    role="Senior Internet Research Strategist",
    goal="Research the internet thoroughly for {topic} and return accurate, current, and high-value findings.",
    backstory="You are an elite open-web intelligence researcher with a strong editorial mindset. You are known for identifying credible sources fast, filtering out noise and hype, cross-checking claims, and surfacing the most important facts, trends, and signals about {topic}. Your work is structured, evidence-driven, and optimized for downstream content creation.",
    tools=[serper_tool] if use_tools else [],
    allow_delegation=True,
    verbose=True,
)

youtube_researcher_agent = Agent(
    role="Senior YouTube Intelligence Analyst",
    goal="Analyze YouTube deeply for {topic}, extract high-signal insights, and run a second-round validation research pass.",
    backstory="You are an expert video intelligence analyst focused on topic mining from YouTube. You identify authoritative channels, isolate actionable insights, detect repeated patterns across creators, and verify claims through follow-up research. You turn noisy video content into structured, trustworthy intelligence about {topic}.",
    tools=[youtube_video_search_tool] if use_tools else [],
    allow_delegation=True,
    verbose=True,
)

linkedin_post_writer_agent = Agent(
    role="Senior LinkedIn Thought-Leadership Writer",
    goal="Write strong, insight-driven LinkedIn posts on {topic} using validated findings from web and YouTube research.",
    backstory="You are a top-tier LinkedIn content strategist and writer who translates complex research into credible thought-leadership content. You write with clarity, authority, and audience relevance, combining hooks, clear narrative flow, and practical takeaways. Your posts on {topic} are built to educate professionals, spark discussion, and build trust.",
    knowledge_sources=[linkedin_skill_source],
    verbose=True,
)

image_creator_agent = Agent(
    role="Senior AI Visual Creative Director",
    goal="Generate a professional, high-quality visual for the LinkedIn post on {topic}.",
    backstory="You are an expert AI image prompt engineer and creative director. You transform written insights into clear, brand-safe, professional visuals suitable for LinkedIn audiences. You create concise prompts that produce clean, modern imagery aligned with business storytelling.",
    tools=[dalle_tool] if use_dalle else [],
    verbose=True,
)

web_research_task = Task(
    description="Research the open internet thoroughly for {topic}. Focus on high-credibility sources, current trends, key facts, and practical insights. Filter out noise, unsupported claims, and duplicate information. Do not ask the user follow-up questions. Assume the topic is final and proceed with best-effort research.",
    expected_output="A structured web research brief on {topic} with key findings, notable trends, and source-backed insights that can be used by downstream tasks.",
    agent=web_researcher_agent,
)

youtube_research_task = Task(
    description="Analyze YouTube content for {topic}. Identify high-signal videos/channels, extract core insights, and perform a second-round validation pass by cross-checking important claims. Do not ask the user follow-up questions. Assume the topic is final and proceed with best-effort research.",
    expected_output="A structured YouTube intelligence brief on {topic} containing validated takeaways, recurring themes, and practical insights aligned with web research.",
    agent=youtube_researcher_agent,
)

linkedin_writing_task = Task(
    description="Using the validated findings from web and YouTube research, write a professional LinkedIn post about {topic}. Ensure the post has a strong hook, clear narrative, actionable insights, and a concise call to action. Do not ask the user follow-up questions. If any upstream details are limited, still produce a complete and publication-ready post using the best available context. Return only the final LinkedIn post text. Do not return tool calls, JSON, XML tags, or requests for additional input.",
    expected_output="One polished, publication-ready LinkedIn post on {topic} that is credible, engaging, and suitable for a professional audience. The output must be only the post text and must not include questions.",
    agent=linkedin_post_writer_agent,
    context=[web_research_task, youtube_research_task],
    output_file="output/linkedin_post.md",
)

image_generation_task = Task(
    description="Read the LinkedIn post from context and create a single professional image prompt aligned to its core message. Use the Dall-E Tool exactly once to generate one image. Return only the tool JSON response containing image_url and image_description.",
    expected_output="A JSON string containing image_url and image_description for the generated LinkedIn visual.",
    agent=image_creator_agent,
    context=[linkedin_writing_task],
    output_file="output/generated_images/image_generation_result.json",
)


def _extract_image_url(raw_text: str) -> str | None:
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and parsed.get("image_url"):
            return str(parsed["image_url"])
    except json.JSONDecodeError:
        pass

    match = re.search(r"https?://[^\s\"']+", raw_text)
    if match:
        return match.group(0)
    return None


def _save_generated_image_file() -> str | None:
    result_path = image_output_dir / "image_generation_result.json"
    if not result_path.exists():
        return None

    raw_text = result_path.read_text(encoding="utf-8", errors="ignore")
    image_url = _extract_image_url(raw_text)
    if not image_url:
        return None

    image_output_dir.mkdir(parents=True, exist_ok=True)
    image_file = image_output_dir / f"linkedin_visual_{int(time.time())}.png"
    urlretrieve(image_url, image_file)
    return str(image_file)

topic_research_to_linkedin_crew = Crew(
    agents=[
        web_researcher_agent,
        youtube_researcher_agent,
        linkedin_post_writer_agent,
        image_creator_agent,
    ],
    tasks=[
        web_research_task,
        youtube_research_task,
        linkedin_writing_task,
        image_generation_task,
    ],
    process=Process.sequential,
    verbose=True,
)


def run_crew(topic: str):
    result = topic_research_to_linkedin_crew.kickoff(inputs={"topic": topic})
    saved_image = _save_generated_image_file()
    if saved_image:
        print(f"Generated image saved to: {saved_image}")
    return result


def run():
    topic = os.getenv("TOPIC", "AI Agents for Sales Prospecting")
    return run_crew(topic)


if __name__ == "__main__":
    run()
