#!/usr/bin/env python

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, YoutubeVideoSearchTool

from ai_agent_web_research_linkedin.config.output_sanitizer import sanitize_image_prompt_output
from ai_agent_web_research_linkedin.config.runtime import (
    load_linkedin_knowledge_sources,
    load_runtime_config,
)

config = load_runtime_config()
default_llm = LLM(model=config.model, max_tokens=320, temperature=0.3)
serper_tool = SerperDevTool() if config.use_tools else None
youtube_video_search_tool = YoutubeVideoSearchTool() if config.use_tools else None
linkedin_knowledge_sources = load_linkedin_knowledge_sources(config.use_knowledge)

web_researcher_agent = Agent(
    role="Senior Internet Research Strategist",
    goal="Research the internet thoroughly for {topic} and return accurate, current, and high-value findings.",
    backstory="You are an elite open-web intelligence researcher with a strong editorial mindset. You are known for identifying credible sources fast, filtering out noise and hype, cross-checking claims, and surfacing the most important facts, trends, and signals about {topic}. Your work is structured, evidence-driven, and optimized for downstream content creation.",
    tools=[serper_tool] if serper_tool else [],
    llm=default_llm,
    allow_delegation=False,
    max_iter=1,
    verbose=True,
)

youtube_researcher_agent = Agent(
    role="Senior YouTube Intelligence Analyst",
    goal="Analyze YouTube deeply for {topic}, extract high-signal insights, and run a second-round validation research pass.",
    backstory="You are an expert video intelligence analyst focused on topic mining from YouTube. You identify authoritative channels, isolate actionable insights, detect repeated patterns across creators, and verify claims through follow-up research. You turn noisy video content into structured, trustworthy intelligence about {topic}.",
    tools=[youtube_video_search_tool] if youtube_video_search_tool else [],
    llm=default_llm,
    allow_delegation=False,
    max_iter=1,
    verbose=True,
)

linkedin_post_writer_agent = Agent(
    role="Senior LinkedIn Thought-Leadership Writer",
    goal="Write strong, insight-driven LinkedIn posts on {topic} using validated findings from web and YouTube research.",
    backstory="You are a top-tier LinkedIn content strategist and writer who translates complex research into credible thought-leadership content. You write with clarity, authority, and audience relevance, combining hooks, clear narrative flow, and practical takeaways. Your posts on {topic} are built to educate professionals, spark discussion, and build trust.",
    knowledge_sources=linkedin_knowledge_sources,
    llm=default_llm,
    max_iter=1,
    verbose=True,
)

image_creator_agent = Agent(
    role="Senior Visual Prompt Strategist",
    goal="Create a high-quality image generation prompt for the LinkedIn post on {topic}.",
    backstory="You are a specialist in visual prompt engineering. You convert business narratives into precise image prompts with style, composition, lighting, mood, and scene details so they can be used in any image model.",
    llm=default_llm,
    max_iter=1,
    verbose=True,
)

web_research_task = Task(
    description="Research {topic} and return only concise, high-signal findings. Include 5 bullet points total: 2 trends, 2 practical insights, 1 risk/caveat. Keep the full response under 140 words. No headings, no citations list, no markdown sections.",
    expected_output="Exactly 5 concise bullets under 140 words total.",
    agent=web_researcher_agent,
)

youtube_research_task = Task(
    description="Analyze YouTube signals for {topic} and return a compact brief: 3 key takeaways and 2 practical actions. Keep the full response under 140 words. No headings, no long lists, no citations block.",
    expected_output="Exactly 5 concise bullets under 140 words total.",
    agent=youtube_researcher_agent,
)

linkedin_writing_task = Task(
    description="Write one professional LinkedIn post on {topic} using prior context. Keep it between 110 and 150 words. Use 1 hook line, 1 short body paragraph, and 1 CTA line. Return only the post text.",
    expected_output="One LinkedIn post, 110-150 words, plain text only.",
    agent=linkedin_post_writer_agent,
    context=[web_research_task, youtube_research_task],
    output_file="output/linkedin_post.md",
)

image_prompt_task = Task(
    description="Read the LinkedIn post and produce one image prompt only. Maximum 65 words. Include subject, setting, composition, style, color palette, lighting, and mood. Do not repeat the LinkedIn post text. No headings. No quotes. One line only.",
    expected_output="A single-line image prompt under 65 words.",
    agent=image_creator_agent,
    context=[linkedin_writing_task],
    output_file="output/generated_images/image_prompt.txt",
)

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
        image_prompt_task,
    ],
    process=Process.sequential,
    verbose=True,
)


def run_crew(topic: str):
    result = topic_research_to_linkedin_crew.kickoff(inputs={"topic": topic})
    was_sanitized = sanitize_image_prompt_output(topic)
    if was_sanitized:
        print("Adjusted image prompt output to enforce prompt-only format.")
    return result


def run():
    return run_crew(config.topic)


if __name__ == "__main__":
    run()
