#!/usr/bin/env python

import os

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, YoutubeVideoSearchTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

serper_tool = SerperDevTool()
youtube_video_search_tool = YoutubeVideoSearchTool()
use_tools = os.getenv("USE_TOOLS", "true").strip().lower() == "true"

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

topic_research_to_linkedin_crew = Crew(
    agents=[
        web_researcher_agent,
        youtube_researcher_agent,
        linkedin_post_writer_agent,
    ],
    tasks=[
        web_research_task,
        youtube_research_task,
        linkedin_writing_task,
    ],
    process=Process.sequential,
    verbose=True,
)


def run_crew(topic: str):
    return topic_research_to_linkedin_crew.kickoff(inputs={"topic": topic})


def run():
    topic = os.getenv("TOPIC", "AI Agents for Sales Prospecting")
    return run_crew(topic)


if __name__ == "__main__":
    run()
