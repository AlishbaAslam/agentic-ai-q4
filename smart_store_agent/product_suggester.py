import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv
from agents.run import RunConfig
import asyncio

# Load environment variables from .env file.
load_dotenv()

# Fetch Gemini API key.
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# Configure Gemini via OpenAI compatible API.
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set up the model (Gemini 2.0 Flash).
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

# Define the Product Suggester Agent.
async def main():
    agent = Agent(
        name="Smart Store Assistant",
        instructions="""
        You are a helpful product recommendation agent for a pharmacy/store.
        - If the user describes a symptom (e.g., headache, cough), suggest a suitable product.
        - Always explain why the product is recommended.
        - Be concise and friendly.
        Example:
        User: "I have a headache."
        You: "You can take Paracetamol (500mg). It helps relieve headaches and reduce fever."
        """,
        model=model
    )
    
    # Get User input and Run the Agent.
    user_query = input("What do you need help with? (e.g: 'I have a headache'): ")

    result = await Runner.run(
    agent,
    user_query,
    run_config=config
    )
    print("\nRecommendation:", result.final_output)

if __name__ == "__main__":
    asyncio.run(main())