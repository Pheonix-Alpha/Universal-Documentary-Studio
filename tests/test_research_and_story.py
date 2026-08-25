from __future__ import annotations

from adapters.research.mock_provider import MockResearchProvider
from agents.fact_checker import FactChecker
from agents.research_agent import ResearchAgent
from agents.story_agent import StoryAgent
from core.models import FactCheckStatus, StoryStructure


def test_research_agent_produces_sources_and_claims(sample_topic):
    agent = ResearchAgent(MockResearchProvider())
    research = agent.run(sample_topic, depth="standard")
    assert research.topic == sample_topic
    assert len(research.sources) > 0
    assert len(research.claims) > 0
    for claim in research.claims:
        assert len(claim.source_ids) > 0


def test_research_agent_depth_affects_source_count(sample_topic):
    agent = ResearchAgent(MockResearchProvider())
    quick = agent.run(sample_topic, depth="quick")
    deep = agent.run(sample_topic, depth="deep")
    assert len(deep.sources) > len(quick.sources)


def test_fact_checker_passes_well_sourced_claims(sample_topic):
    research = ResearchAgent(MockResearchProvider()).run(sample_topic)
    report = FactChecker().run(research)
    assert len(report.results) == len(research.claims)
    assert not report.has_critical_failure


def test_fact_checker_flags_claim_with_no_sources():
    from core.models import Claim, Research

    research = Research(topic="X", claims=[Claim(text="Unsupported claim", source_ids=[])], sources=[])
    report = FactChecker().run(research)
    assert report.has_critical_failure
    assert report.results[0].status == FactCheckStatus.FAILED


def test_story_agent_classifies_business_topic(sample_topic):
    research = ResearchAgent(MockResearchProvider()).run(sample_topic)
    story = StoryAgent().run(research)
    assert len(story.structures) >= 1
    assert story.logline


def test_story_agent_defaults_to_chronology_when_no_signal():
    from core.models import Research

    research = Research(topic="Neutral Topic With No Keywords", claims=[], sources=[], context_notes=[])
    story = StoryAgent().run(research)
    assert story.structures == [StoryStructure.CHRONOLOGY]
