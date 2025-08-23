import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper, function_tool
from dotenv import load_dotenv
from agents.run import RunConfig
from pydantic import BaseModel
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

# Define a simple context using a BaseModel
class UserId(BaseModel):
    id : int

# A tool function that accesses local context via the wrapper
@function_tool
async def check_id(wrapper: RunContextWrapper[UserId]) -> str:
    """Check the user has valid ID."""
    user = wrapper.context
    if user.id == 123:
        return "you are allowed"
    else:
        return "you are not allowed"
    
# Define the Simple Agent.
async def main():
    # Create your context object
    user_context = UserId(id=123)

    agent = Agent(
        name="Friendly Assistant",
        instructions= "You are a helpful, friendly assistant who answers questions clearly and politely.",
        model=model,
        tools=[check_id]
    )
    
    # Get User input and Run the Agent.

    user_query = input("Enter your query here: ")

    result = await Runner.run(
    agent,
    input=user_query,
    run_config=config,
    context=user_context
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())