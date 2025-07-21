import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio

# Load environment variables from .env file.
load_dotenv()

# Fetch Gemini API key.
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

async def main():
    # Agent 1: Mood Analyzer
    mood_agent = Agent(
        name="Mood Analyzer",
        instructions="""
        Analyze the user's message and return ONLY one word:
        - "happy" (if the user expresses joy/excitement),
        - "sad" (if the user expresses grief/loneliness),
        - "stressed" (if the user mentions anxiety/pressure),
        - "neutral" (for physical symptoms like "headache" or non-emotional statements).
        Examples:
        - "I lost my cat" ➡ "sad"
        - "I have a headache" ➡ "neutral"
        - "Work is overwhelming" ➡ "stressed"
        """,
        model=model
    )

    # Agent 2: Activity Suggester (only for sad/stressed/neutral)
    activity_agent = Agent(
        name="Activity Suggester",
        instructions="""
        Suggest a compassionate activity based on WHY the user is sad/stressed/neutral. 
        For pet loss, focus on memorializing or gentle comforts.
        For other sadness, suggest uplifting activities.
        Examples:
        - "I lost my cat" ➡ "Try: Creating a photo album of your cat. Honoring memories can bring comfort."
        - "I'm stressed at work" ➡ "Try: A 10-minute walk outside. Nature reduces stress hormones."
        - General sadness ➡ "Try: Talking to a close friend. Connection eases loneliness."
        """,
        model=model
    )

    # Get User input.
    user_query = input("How are you feeling today? ")

    # Run Agent 1: Detect Mood
    mood_result = await Runner.run(mood_agent, user_query, run_config=config)
    mood = mood_result.final_output.lower().strip('"')
    print("Mood Detected:", mood)

    # Handoff to Agent 2 if mood is sad/stressed/neutral.
    if "sad" in mood or "stressed" in mood or "neutral" in mood:
        activity_result = await Runner.run(
            activity_agent,
            f"User is feeling {mood} because: '{user_query}'. Suggest an activity.",
            run_config=config
        )
        print("Recommendation:", activity_result.final_output)
    else:
        print("No recommendation needed. Keep enjoying your day!")

if __name__ == "__main__":
    asyncio.run(main())