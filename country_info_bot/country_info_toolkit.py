import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio

# Load environment variables from .env file.
load_dotenv()

# Fetch Gemini API Key.
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# Configure Gemini via OpenAI-compatible API.
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set up the model (Gemini-2.0-Flash).
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Disable tracing for simplicity.
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 1. Capital Agent
capital_agent = Agent(
    name="Capital Agent",
    instructions="Return ONLY the capital city when given a country name. Example: 'France' ➡ 'Paris'",
    model=model
)

# 2. Language Agent
language_agent = Agent(
    name="Language Agent",
    instructions="Return ONLY the official language(s) when given a country name. Example: 'Brazil' ➡ 'Portuguese'",
    model=model
)

# 3. Population Agent
population_agent = Agent(
    name="Population Agent",
    instructions="Return ONLY the approximate population when given a country name. Example: 'Germany' ➡ '83 million'",
    model=model
)

# Orchestrator Agent
orchestrator = Agent(
    name="Country Orchestrator",
    instructions="""When asked about a country:
    1. Use translate_to_capital for capital city
    2. Use translate_to_language for official language
    3. Use translate_to_population for population
    Return all information in this format:
    Country: [name]
    Capital: [capital]
    Language: [language]
    Population: [population]""",
    tools=[
        capital_agent.as_tool(
            tool_name="translate_to_capital",
            tool_description="Translates country name to its capital city"
        ),
        language_agent.as_tool(
            tool_name="translate_to_language",
            tool_description="Translates country name to its official language"
        ),
        population_agent.as_tool(
            tool_name="translate_to_population",
            tool_description="Translates country name to its approximate population"
        )
    ],
    model=model
)

async def main():
    while True:
        country = input("\nEnter country name (or 'quit'): ")
        if country.lower() == 'quit':
            break

        result = await Runner.run(
            orchestrator,
            f"Tell me about {country}",
            run_config=config
        )
        print(f"\n{result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())