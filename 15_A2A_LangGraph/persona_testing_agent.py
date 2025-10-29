"""Persona-Based Testing Agent using Pure OpenAI API (Different Framework).

This agent tests your A2A application by simulating different user personas
with varying expertise levels, goals, and expectations. Unlike the main agent
which uses LangGraph, this testing agent uses the raw OpenAI API directly.

This demonstrates using a DIFFERENT agent framework to test your A2A application.
"""
import asyncio
import logging
from typing import TypedDict, List, Dict, Any
from uuid import uuid4
import os

import httpx
from openai import AsyncOpenAI
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Persona(TypedDict):
    """Definition of a testing persona."""
    name: str
    role: str
    system_prompt: str
    initial_query: str
    goals: List[str]
    max_interactions: int
    expertise_level: str


class TestResult(TypedDict):
    """Result of testing with a persona."""
    persona_name: str
    satisfied: bool
    goals_met: List[str]
    goals_unmet: List[str]
    conversation: List[Dict[str, str]]
    evaluation_summary: str


# Define three different personas for testing
PERSONAS: List[Persona] = [
    {
        "name": "Dr. Sarah Chen",
        "role": "ML Research Scientist",
        "expertise_level": "expert",
        "system_prompt": """You are Dr. Sarah Chen, an expert in Machine Learning with a PhD from MIT.
You want to learn about technical topics but are NOT satisfied with surface-level answers.
You demand:
- Technical depth and accuracy
- Academic sources (research papers, arXiv links)
- Concrete examples and benchmarks
- Verification of claims with sources

You will ask critical follow-up questions if:
- Answers are too vague or general
- No sources are provided
- Technical details are missing
- Claims are not backed up with evidence

Be direct, professional, and intellectually rigorous in your questioning.""",
        "initial_query": "What makes Kimi K2 so incredible? I need technical details and sources to verify the information.",
        "goals": [
            "Obtain technical architecture details about Kimi K2",
            "Get links to research papers or academic sources",
            "Understand specific capabilities with concrete examples",
            "Verify key claims with credible sources"
        ],
        "max_interactions": 3
    },
    {
        "name": "Alex Rivera",
        "role": "Computer Science Undergraduate",
        "expertise_level": "beginner",
        "system_prompt": """You are Alex Rivera, a 3rd-year Computer Science student who is curious about AI.
You are enthusiastic but need concepts explained in accessible, clear terms.
You ask follow-up questions when:
- Technical jargon is used without explanation
- Concepts are unclear
- You want concrete examples to understand better

You appreciate:
- Clear, simple explanations
- Real-world examples
- Step-by-step breakdowns
- Analogies that make complex topics understandable

Be curious, friendly, and willing to admit when you don't understand something.""",
        "initial_query": "I keep hearing about AI agents and LangGraph. Can you explain what they are and why they matter?",
        "goals": [
            "Understand what AI agents are in simple terms",
            "Learn what LangGraph is and its purpose",
            "Get practical examples of how they're used",
            "Understand why these technologies are important"
        ],
        "max_interactions": 3
    },
    {
        "name": "Prof. James Morton",
        "role": "Critical Researcher",
        "expertise_level": "expert",
        "system_prompt": """You are Professor James Morton, a tenured professor known for critical analysis and peer review.
You question claims and demand rigorous evidence.
You prefer:
- Primary sources over secondary summaries
- Peer-reviewed research over blog posts
- Data and benchmarks over anecdotes
- Acknowledgment of limitations

You will challenge:
- Vague or unsupported statements
- Overhyped claims without evidence
- Missing caveats or limitations
- Insufficient source attribution

Be skeptical but fair, professional, and focused on scientific rigor.""",
        "initial_query": "Everyone is talking about the latest AI models. What's the actual evidence for these performance claims?",
        "goals": [
            "Identify specific claims being made",
            "Get evidence for each major claim",
            "Obtain access to primary sources",
            "Understand limitations and caveats"
        ],
        "max_interactions": 3
    }
]


class PersonaTestingAgent:
    """Testing agent that uses different personas to test an A2A agent.

    This uses pure OpenAI API (different from LangGraph) to simulate user personas.
    """

    def __init__(self, a2a_base_url: str = "http://localhost:10000"):
        self.a2a_base_url = a2a_base_url
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def call_a2a_agent(self, query: str) -> str:
        """Make an A2A API call to the agent being tested."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            # Initialize A2A card resolver
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=self.a2a_base_url,
            )

            # Fetch the agent card
            agent_card = await resolver.get_agent_card()

            # Initialize A2A client
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=agent_card
            )

            # Prepare the message
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': query}
                    ],
                    'message_id': uuid4().hex,
                },
            }

            # Create and send request
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )

            response = await client.send_message(request)
            response_data = response.model_dump(mode='json', exclude_none=True)

            # Extract text from artifacts
            result = response_data.get('result', {})
            artifacts = result.get('artifacts', [])

            if artifacts:
                artifact = artifacts[0]
                parts = artifact.get('parts', [])
                text_parts = [p.get('text', '') for p in parts if p.get('kind') == 'text']
                return '\n'.join(text_parts)

            return "No response received from A2A agent."

    async def generate_persona_followup(self, persona: Persona, conversation_history: List[Dict[str, str]]) -> str | None:
        """Use OpenAI to generate a follow-up question based on the persona's goals and conversation history."""
        # Build conversation context
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history
        ])

        goals_text = "\n".join([f"- {goal}" for goal in persona["goals"]])

        prompt = f"""Based on the conversation below and your goals, decide if you need to ask a follow-up question.

Your Goals:
{goals_text}

Conversation So Far:
{conversation_text}

If the response adequately addresses your goals and you are satisfied, respond with only: SATISFIED

If you need more information or the response was insufficient, generate ONE specific follow-up question that will help you achieve your goals. Be direct and focused.

Your response (either 'SATISFIED' or a follow-up question):"""

        messages = [
            {"role": "system", "content": persona["system_prompt"]},
            {"role": "user", "content": prompt}
        ]

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )

        followup = response.choices[0].message.content.strip()

        if "SATISFIED" in followup.upper():
            return None

        return followup

    async def evaluate_persona_satisfaction(self, persona: Persona, conversation_history: List[Dict[str, str]]) -> tuple[bool, List[str], List[str], str]:
        """Evaluate if the persona's goals were met using OpenAI API."""
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history
        ])

        goals_text = "\n".join([f"{i+1}. {goal}" for i, goal in enumerate(persona["goals"])])

        prompt = f"""Evaluate if the following conversation met the user's goals.

User's Goals:
{goals_text}

Conversation:
{conversation_text}

For each goal, determine if it was MET or UNMET based on the conversation.
Then provide an overall assessment.

Respond in this exact format:
GOALS MET:
- [list goals that were met, or "None" if none were met]

GOALS UNMET:
- [list goals that were unmet, or "None" if all were met]

OVERALL: [SATISFIED or UNSATISFIED]

SUMMARY: [1-2 sentence explanation of why the persona would or would not be satisfied]"""

        messages = [
            {"role": "system", "content": "You are an objective evaluator assessing if a conversation met specific user goals."},
            {"role": "user", "content": prompt}
        ]

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        evaluation = response.choices[0].message.content.strip()

        # Parse the evaluation
        satisfied = "OVERALL: SATISFIED" in evaluation

        # Extract goals met/unmet
        goals_met = []
        goals_unmet = []
        summary = ""

        lines = evaluation.split('\n')
        in_met_section = False
        in_unmet_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("GOALS MET:"):
                in_met_section = True
                in_unmet_section = False
            elif line.startswith("GOALS UNMET:"):
                in_unmet_section = True
                in_met_section = False
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                break
            elif line.startswith("-") and in_met_section:
                goals_met.append(line[1:].strip())
            elif line.startswith("-") and in_unmet_section:
                goals_unmet.append(line[1:].strip())

        return satisfied, goals_met, goals_unmet, summary

    async def test_with_persona(self, persona: Persona) -> TestResult:
        """Run a test session with a specific persona."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with Persona: {persona['name']} ({persona['role']})")
        logger.info(f"Expertise Level: {persona['expertise_level']}")
        logger.info(f"{'='*80}\n")

        conversation_history: List[Dict[str, str]] = []

        # Initial query
        logger.info(f"INITIAL QUERY: {persona['initial_query']}\n")
        query = persona['initial_query']

        for interaction in range(persona['max_interactions']):
            logger.info(f"--- Interaction {interaction + 1} ---")

            # Get response from A2A agent
            logger.info(f"Sending to A2A agent: {query}...")
            response = await self.call_a2a_agent(query)
            logger.info(f"Received response ({len(response)} chars)\n")
            logger.info(f"Response: {response}\n")

            # Add to conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response})

            # Check if we should continue
            if interaction < persona['max_interactions'] - 1:
                followup = await self.generate_persona_followup(persona, conversation_history)

                if followup is None:
                    logger.info("Persona is satisfied. No follow-up needed.\n")
                    break
                else:
                    logger.info(f"FOLLOW-UP: {followup}\n")
                    query = followup
            else:
                logger.info("Max interactions reached.\n")

        # Evaluate satisfaction
        logger.info("Evaluating persona satisfaction...")
        satisfied, goals_met, goals_unmet, summary = await self.evaluate_persona_satisfaction(
            persona, conversation_history
        )

        return TestResult(
            persona_name=f"{persona['name']} ({persona['role']})",
            satisfied=satisfied,
            goals_met=goals_met,
            goals_unmet=goals_unmet,
            conversation=conversation_history,
            evaluation_summary=summary
        )

    async def run_all_tests(self) -> List[TestResult]:
        """Run tests with all defined personas."""
        results = []

        print("\n" + "="*80)
        print("PERSONA-BASED A2A AGENT TESTING")
        print("="*80)
        print(f"\nTesting A2A Agent at: {self.a2a_base_url}")
        print(f"Number of Personas: {len(PERSONAS)}")
        print(f"\nThis uses pure OpenAI API (different from LangGraph)")
        print("="*80 + "\n")

        for persona in PERSONAS:
            result = await self.test_with_persona(persona)
            results.append(result)

        # Print summary report
        self.print_test_report(results)

        return results

    def print_test_report(self, results: List[TestResult]):
        """Print a summary report of all test results."""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80 + "\n")

        for i, result in enumerate(results, 1):
            status = "✓ SATISFIED" if result['satisfied'] else "✗ UNSATISFIED"
            print(f"{i}. {result['persona_name']}: {status}")
            print(f"   Goals Met: {len(result['goals_met'])}/{len(result['goals_met']) + len(result['goals_unmet'])}")
            print(f"   Summary: {result['evaluation_summary']}")
            print()

        satisfied_count = sum(1 for r in results if r['satisfied'])
        print(f"Overall: {satisfied_count}/{len(results)} personas satisfied")
        print("="*80 + "\n")


async def main():
    """Main function to run persona-based testing."""
    agent = PersonaTestingAgent()

    try:
        await agent.run_all_tests()
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        print(f"\nERROR: {e}")
        print("\nMake sure your A2A server is running:")
        print("  uv run python -m app")


if __name__ == '__main__':
    asyncio.run(main())
